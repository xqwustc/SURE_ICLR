import argparse
import csv
import json
import os
from collections import Counter, defaultdict


VALID_WINNERS = {"assistant_1", "assistant_2"}


def read_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def write_jsonl(path, records):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_tsv(path, rows, fieldnames):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def normalize(value):
    return str(value or "").strip().lower()


def is_valid(item):
    return normalize(item.get("router_winner")) in VALID_WINNERS


def is_correct_value(value):
    if isinstance(value, bool):
        return value
    return normalize(value) in {"true", "1", "yes"}


def row_stats(records):
    total = len(records)
    valid = [item for item in records if is_valid(item)]
    with_gt = [item for item in valid if normalize(item.get("ground_truth")) in VALID_WINNERS]
    gt_agree = [item for item in with_gt if normalize(item.get("router_winner")) == normalize(item.get("ground_truth"))]
    gt_disagree = [item for item in with_gt if normalize(item.get("router_winner")) != normalize(item.get("ground_truth"))]
    original_wrong_teacher_right = [
        item
        for item in gt_agree
        if "is_correct" in item and not is_correct_value(item.get("is_correct"))
    ]
    original_right_teacher_wrong = [
        item
        for item in gt_disagree
        if "is_correct" in item and is_correct_value(item.get("is_correct"))
    ]
    return {
        "total": total,
        "valid": len(valid),
        "invalid": total - len(valid),
        "with_gt": len(with_gt),
        "gt_agree": len(gt_agree),
        "gt_disagree": len(gt_disagree),
        "gt_agree_rate": len(gt_agree) / len(with_gt) if with_gt else 0.0,
        "original_wrong_teacher_right": len(original_wrong_teacher_right),
        "original_right_teacher_wrong": len(original_right_teacher_wrong),
    }


def filter_records(records, mode, min_confidence):
    filtered = []
    for item in records:
        if not is_valid(item):
            continue
        confidence = item.get("router_confidence")
        if min_confidence > 0 and (confidence is None or float(confidence) < min_confidence):
            continue
        winner = normalize(item.get("router_winner"))
        gt = normalize(item.get("ground_truth"))
        has_gt = gt in VALID_WINNERS
        if mode == "all":
            filtered.append(item)
        elif mode == "agree_gt" and has_gt and winner == gt:
            filtered.append(item)
        elif mode == "disagree_gt" and has_gt and winner != gt:
            filtered.append(item)
        elif mode == "original_wrong_teacher_right" and has_gt and winner == gt and not is_correct_value(item.get("is_correct")):
            filtered.append(item)
    return filtered


def main():
    parser = argparse.ArgumentParser(description="Audit strong-judge labels before PM training.")
    parser.add_argument(
        "--input_path",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        required=True,
    )
    parser.add_argument(
        "--filter_mode",
        choices=["all", "agree_gt", "disagree_gt", "original_wrong_teacher_right"],
        default="agree_gt",
    )
    parser.add_argument("--min_confidence", type=float, default=0.0)
    args = parser.parse_args()

    records = read_jsonl(args.input_path)
    stats = row_stats(records)
    winner_counts = Counter(normalize(item.get("router_winner")) or "missing" for item in records)

    domain_rows = []
    by_domain = defaultdict(list)
    for item in records:
        by_domain[str(item.get("domain") or "unknown")].append(item)
    for domain, items in sorted(by_domain.items()):
        row = {"domain": domain, **row_stats(items)}
        domain_rows.append(row)

    filtered = filter_records(records, args.filter_mode, args.min_confidence)
    os.makedirs(args.output_dir, exist_ok=True)
    filtered_path = os.path.join(args.output_dir, f"train_{args.filter_mode}.jsonl")
    write_jsonl(filtered_path, filtered)

    summary = {
        "input_path": args.input_path,
        "filter_mode": args.filter_mode,
        "min_confidence": args.min_confidence,
        "filtered_count": len(filtered),
        "winner_counts": dict(sorted(winner_counts.items())),
        **stats,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    write_tsv(
        os.path.join(args.output_dir, "domain_summary.tsv"),
        domain_rows,
        [
            "domain",
            "total",
            "valid",
            "invalid",
            "with_gt",
            "gt_agree",
            "gt_disagree",
            "gt_agree_rate",
            "original_wrong_teacher_right",
            "original_right_teacher_wrong",
        ],
    )

    print(f"Loaded {len(records)} judgements from {args.input_path}")
    print("Winner counts:")
    for winner, count in sorted(winner_counts.items()):
        print(f"  {winner}: {count}")
    print(
        f"valid={stats['valid']} invalid={stats['invalid']} "
        f"gt_agree={stats['gt_agree']} gt_disagree={stats['gt_disagree']} "
        f"gt_agree_rate={stats['gt_agree_rate']:.6f}"
    )
    print(
        f"original_wrong_teacher_right={stats['original_wrong_teacher_right']} "
        f"original_right_teacher_wrong={stats['original_right_teacher_wrong']}"
    )
    print(f"Saved filtered training judgements: {filtered_path}")
    print(f"filtered_count={len(filtered)} filter_mode={args.filter_mode}")
    print(f"Saved audit summary to {args.output_dir}")


if __name__ == "__main__":
    main()
