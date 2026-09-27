#!/usr/bin/env python3
import argparse
import json
import os
import random
from collections import defaultdict


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--per_domain", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    groups = defaultdict(list)
    winner_counts = defaultdict(int)
    for row in read_jsonl(args.input):
        winner = row.get("router_winner")
        if winner not in {"assistant_1", "assistant_2"}:
            raise ValueError(
                f"Invalid judge router_winner={winner!r} for pair_index={row.get('pair_index')}; "
                "ground_truth fallback is forbidden"
            )
        winner_counts[winner] += 1
        groups[str(row.get("domain", "unknown"))].append(row)
    if len(groups) < 2:
        raise ValueError(f"Expected multiple domains, found {sorted(groups)}")

    rng = random.Random(args.seed)
    dev, holdout, pool, counts = [], [], [], {}
    for domain, rows in sorted(groups.items()):
        rng.shuffle(rows)
        # Low-budget selections can contain fewer rows than per_domain. Keep at
        # least one holdout example whenever the domain has enough data for it.
        n_dev = min(args.per_domain, max(1, len(rows) - 1))
        dev.extend(rows[:n_dev])
        holdout.extend(rows[n_dev:])
        pool.extend(rows)
        counts[domain] = {"total": len(rows), "dev": n_dev, "holdout": len(rows) - n_dev}

    rng.shuffle(dev)
    rng.shuffle(holdout)
    rng.shuffle(pool)
    os.makedirs(args.output_dir, exist_ok=True)
    write_jsonl(os.path.join(args.output_dir, "evolution_dev.jsonl"), dev)
    write_jsonl(os.path.join(args.output_dir, "evolution_pool.jsonl"), pool)
    write_jsonl(os.path.join(args.output_dir, "holdout.jsonl"), holdout)
    with open(os.path.join(args.output_dir, "split_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "seed": args.seed,
                "per_domain": args.per_domain,
                "label_source": "router_winner",
                "winner_counts": dict(sorted(winner_counts.items())),
                "counts": counts,
            },
            handle,
            indent=2,
        )
        handle.write("\n")
    print(f"Saved stratified evolution dev={len(dev)} holdout={len(holdout)}")
    print(f"Judge winner counts={dict(sorted(winner_counts.items()))}")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
