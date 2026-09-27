#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import re
import shutil
from datetime import datetime

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk


DOMAIN_ALIASES = {
    "chat": ["chat", "conversation", "general"],
    "code": ["code", "coding", "programming"],
    "math": ["math", "mathematical", "calculation"],
    "safety-refuse": ["safety-refuse", "refuse", "refusal", "safety"],
    "safety-response": ["safety-response", "safe response", "safety"],
}


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


def latest_rubric_dir(rubric_root):
    if not os.path.isdir(rubric_root):
        raise FileNotFoundError(f"Rubric root does not exist: {rubric_root}")
    candidates = [
        os.path.join(rubric_root, name)
        for name in os.listdir(rubric_root)
        if os.path.isdir(os.path.join(rubric_root, name))
    ]
    if not candidates:
        raise FileNotFoundError(f"No rubric directories found in {rubric_root}")
    return max(candidates, key=os.path.getmtime)


def read_text(path):
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as handle:
        return handle.read().strip()


def load_any_dataset(path_or_name):
    if os.path.exists(path_or_name):
        return load_from_disk(path_or_name)
    return load_dataset(path_or_name)


def get_split(dataset, split):
    if isinstance(dataset, DatasetDict):
        if split not in dataset:
            raise KeyError(f"Split {split!r} not found. Available splits: {list(dataset.keys())}")
        return dataset[split]
    return dataset


def select_subset(dataset, max_samples, seed):
    if max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    return dataset.shuffle(seed=seed).select(range(max_samples))


def split_markdown_sections(text):
    sections = []
    current_title = ""
    current_lines = []
    for line in text.splitlines():
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if match:
            if current_title or current_lines:
                sections.append((current_title, "\n".join(current_lines).strip()))
            current_title = match.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_title or current_lines:
        sections.append((current_title, "\n".join(current_lines).strip()))
    return [(title, body) for title, body in sections if body]


def build_domain_rubric_map(domain_text):
    sections = split_markdown_sections(domain_text)
    if not sections:
        return {}

    domain_map = {}
    for domain, aliases in DOMAIN_ALIASES.items():
        matched = []
        for title, body in sections:
            haystack = f"{title}\n{body}".lower()
            if any(alias in haystack for alias in aliases):
                matched.append(f"## {title}\n{body}")
        if matched:
            domain_map[domain] = "\n\n".join(matched)
    return domain_map


def pick_domain_rubric(domain, domain_map, full_domain_rubric, fallback):
    domain = str(domain or "").strip()
    if domain in domain_map:
        return domain_map[domain]
    if domain.startswith("safety") and "safety-refuse" in domain_map:
        return domain_map["safety-refuse"]
    if fallback == "full":
        return full_domain_rubric
    return ""


def normalize_winner(value):
    value = str(value or "").strip().lower()
    if value in {"assistant_1", "assistant 1", "response_1", "a", "1"}:
        return "assistant_1"
    if value in {"assistant_2", "assistant 2", "response_2", "b", "2"}:
        return "assistant_2"
    return ""


def infer_label(item, label_source):
    source_order = [label_source]
    if item.get("strict_label_source"):
        source_order = [label_source]
    elif label_source == "ground_truth":
        source_order.append("router_winner")
    elif label_source == "router_winner":
        source_order.append("ground_truth")

    for source in source_order:
        winner = normalize_winner(item.get(source))
        if winner == "assistant_1":
            # PreferenceLoss treats label=1 as assistant_1 winning in the
            # displayed order, and label=0 as assistant_2 winning.
            return 1.0, source
        if winner == "assistant_2":
            return 0.0, source
    return None, None


def build_user_content(item, general_rubric, domain_rubric):
    parts = [
        "Use the rubric below to compare the two assistant responses.",
        "",
        "[General Rubric]",
        general_rubric,
    ]
    if domain_rubric:
        parts.extend(["", f"[Domain-Specific Rubric: {item.get('domain', 'unknown')}]", domain_rubric])
    parts.extend(["", "[User Request]", item.get("prompt", "")])
    return "\n".join(parts)


def build_unified_rubric_system_content(general_rubric):
    return "\n".join(
        [
            "Use the rubric below to compare the two assistant responses.",
            "",
            "[General Rubric]",
            general_rubric,
        ]
    )


def normalize_replay_example(item, source):
    messages = item.get("context_messages")
    if not messages:
        return None
    label = item.get("label")
    if label is None:
        return None
    strength = item.get("strength", 1.0)
    context = item.get("context", "")
    if not context and messages and messages[0].get("role") == "user":
        context = messages[0].get("content", "")
    return {
        "context": context,
        "context_messages": messages,
        "label": float(label),
        "strength": float(strength),
        "source": source,
        "domain": item.get("domain", "id_replay"),
        "prompt_index": item.get("prompt_index", -1),
        "pair_index": item.get("pair_index", -1),
        "label_source": item.get("label_source", "base_dataset"),
        "rubric_dir": item.get("rubric_dir", ""),
        "pair_order": item.get("pair_order", "replay"),
    }


def build_replay_examples(dataset, max_samples, seed, source):
    subset = select_subset(dataset, max_samples, seed)
    examples = []
    skipped = 0
    for item in subset:
        example = normalize_replay_example(item, source)
        if example is None:
            skipped += 1
        else:
            examples.append(example)
    return examples, skipped


def build_examples(records, general_rubric, domain_map, full_domain_rubric, args):
    examples = []
    skipped = 0
    for item in records:
        assistant_1 = item.get("assistant_1")
        assistant_2 = item.get("assistant_2")
        prompt = item.get("prompt")
        label, label_source_used = infer_label(item, args.label_source)
        if not prompt or not assistant_1 or not assistant_2 or label is None:
            skipped += 1
            continue

        router_confidence = item.get("router_confidence")
        original_pm_score = item.get("score")
        original_pm_probability = None
        if args.blend_router_with_original_score and label_source_used == "router_winner":
            if router_confidence is None or original_pm_score is None:
                raise ValueError(
                    "--blend_router_with_original_score requires router_confidence and score "
                    f"for pair_index={item.get('pair_index')}"
                )
            confidence = float(router_confidence)
            original_pm_score = float(original_pm_score)
            if not 0.0 <= confidence <= 1.0 or not math.isfinite(original_pm_score):
                raise ValueError(
                    f"Invalid confidence/score for pair_index={item.get('pair_index')}: "
                    f"confidence={confidence}, score={original_pm_score}"
                )
            if original_pm_score >= 0:
                original_pm_probability = 1.0 / (1.0 + math.exp(-original_pm_score))
            else:
                exp_score = math.exp(original_pm_score)
                original_pm_probability = exp_score / (1.0 + exp_score)
            label = confidence * label + (1.0 - confidence) * original_pm_probability
        elif args.soft_router_labels and label_source_used == "router_winner":
            confidence = 1.0 if router_confidence is None else float(router_confidence)
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(f"router_confidence must be in [0, 1], got {confidence}")
            confidence = max(0.5, confidence)
            label = confidence if label == 1.0 else 1.0 - confidence

        original_context = item.get("context_messages")
        if args.no_training_rubric:
            if not isinstance(original_context, list) or not original_context:
                original_context = [{"role": "user", "content": prompt}]
            else:
                original_context = [dict(message) for message in original_context]
        elif args.unified_rubric:
            if not isinstance(original_context, list) or not original_context:
                original_context = [{"role": "user", "content": prompt}]
            else:
                original_context = [dict(message) for message in original_context]
            original_context = [
                {"role": "system", "content": build_unified_rubric_system_content(general_rubric)}
            ] + original_context
        else:
            domain_rubric = ""
            domain_rubric = pick_domain_rubric(
                item.get("domain"),
                domain_map,
                full_domain_rubric,
                args.domain_rubric_fallback,
            )
            original_context = [{"role": "user", "content": build_user_content(item, general_rubric, domain_rubric)}]

        base_example = {
            "context": prompt,
            "context_messages": original_context + [
                {"role": "assistant_1", "content": assistant_1},
                {"role": "assistant_2", "content": assistant_2},
            ],
            "label": label,
            "strength": float(args.strength),
            "source": "domain_rubric_pm_continue",
            "domain": item.get("domain", ""),
            "prompt_index": item.get("prompt_index"),
            "pair_index": item.get("pair_index"),
            "label_source": label_source_used,
            "router_confidence": router_confidence,
            "original_pm_score": original_pm_score,
            "original_pm_probability": original_pm_probability,
            "rubric_dir": args.rubric_dir,
            "pair_order": "forward",
        }
        examples.append(base_example)
        if args.add_reverse_pairs:
            reverse_example = dict(base_example)
            reverse_example.update(
                {
                    "context_messages": original_context + [
                        {"role": "assistant_1", "content": assistant_2},
                        {"role": "assistant_2", "content": assistant_1},
                    ],
                    "label": 1.0 - label,
                    "source": "domain_rubric_pm_continue_reverse",
                    "pair_order": "reverse",
                }
            )
            examples.append(reverse_example)
    if not examples:
        raise ValueError(f"No usable examples found. skipped={skipped}")
    return examples, skipped


def main():
    parser = argparse.ArgumentParser(description="Build a domain-rubric-augmented preference-model dataset.")
    parser.add_argument("--teacher_pairs_jsonl", required=True)
    parser.add_argument("--output_dataset", required=True)
    parser.add_argument("--rubric_root", default="./artifacts/rubrics")
    parser.add_argument("--rubric_dir", default="")
    parser.add_argument("--label_source", choices=["ground_truth", "router_winner"], default="ground_truth")
    parser.add_argument(
        "--strict_label_source",
        action="store_true",
        default=False,
        help="Skip examples whose requested label source is missing instead of falling back to another label.",
    )
    parser.add_argument("--domain_rubric_fallback", choices=["full", "none"], default="full")
    parser.add_argument(
        "--no_training_rubric",
        action="store_true",
        help="Keep selected examples in the original prompt format; rubrics are then inference-only.",
    )
    parser.add_argument(
        "--unified_rubric",
        action="store_true",
        help="Use general_rubric.md as a system prefix for every example without domain-specific rules.",
    )
    parser.add_argument(
        "--add_reverse_pairs",
        action="store_true",
        help="Add response-order-reversed selected examples with flipped labels.",
    )
    parser.add_argument(
        "--soft_router_labels",
        action="store_true",
        help="Use router confidence as a soft target: c for assistant_1 and 1-c for assistant_2.",
    )
    parser.add_argument(
        "--blend_router_with_original_score",
        action="store_true",
        help="Blend the router target with sigmoid(score) using router_confidence as the weight.",
    )
    parser.add_argument("--base_dataset", default="")
    parser.add_argument("--base_train_split", default="train")
    parser.add_argument("--base_eval_split", default="validation")
    parser.add_argument("--base_train_max_samples", type=int, default=0)
    parser.add_argument("--base_eval_max_samples", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--validation_ratio", type=float, default=0.05)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.no_training_rubric and args.unified_rubric:
        raise ValueError("--unified_rubric cannot be combined with --no_training_rubric")
    if args.soft_router_labels and args.blend_router_with_original_score:
        raise ValueError(
            "Use only one of --soft_router_labels and --blend_router_with_original_score"
        )

    general_rubric = ""
    full_domain_rubric = ""
    if args.no_training_rubric:
        args.rubric_dir = ""
    else:
        args.rubric_dir = args.rubric_dir or latest_rubric_dir(args.rubric_root)
        general_rubric = read_text(os.path.join(args.rubric_dir, "general_rubric.md"))
        full_domain_rubric = read_text(os.path.join(args.rubric_dir, "domain_specific_rubric.md"))
        if not general_rubric:
            raise FileNotFoundError(f"Missing general_rubric.md in {args.rubric_dir}")
        if not args.unified_rubric and not full_domain_rubric:
            raise FileNotFoundError(f"Missing domain_specific_rubric.md in {args.rubric_dir}")

    records = read_jsonl(args.teacher_pairs_jsonl)
    if args.strict_label_source:
        for item in records:
            item["strict_label_source"] = True
    if args.limit > 0:
        records = records[: args.limit]

    domain_map = build_domain_rubric_map(full_domain_rubric)
    if args.add_reverse_pairs:
        shuffled_records = list(records)
        random.Random(args.seed).shuffle(shuffled_records)
        n_eval_records = (
            max(1, int(len(shuffled_records) * args.validation_ratio))
            if args.validation_ratio > 0 and len(shuffled_records) > 1
            else 0
        )
        eval_records = shuffled_records[:n_eval_records] if n_eval_records > 0 else []
        train_records = shuffled_records[n_eval_records:]
        train_examples, train_skipped = build_examples(
            train_records,
            general_rubric,
            domain_map,
            full_domain_rubric,
            args,
        )
        if eval_records:
            eval_examples, eval_skipped = build_examples(
                eval_records,
                general_rubric,
                domain_map,
                full_domain_rubric,
                args,
            )
        else:
            eval_examples, eval_skipped = [], 0
        selected_examples = train_examples + eval_examples
        skipped = train_skipped + eval_skipped
    else:
        selected_examples, skipped = build_examples(records, general_rubric, domain_map, full_domain_rubric, args)
        random.Random(args.seed).shuffle(selected_examples)
        if args.validation_ratio <= 0:
            train_examples = selected_examples
            eval_examples = []
        else:
            n_eval = max(1, int(len(selected_examples) * args.validation_ratio)) if len(selected_examples) > 1 else 0
            train_examples = selected_examples[n_eval:]
            eval_examples = selected_examples[:n_eval] if n_eval > 0 else selected_examples

    replay_train = []
    replay_eval = []
    replay_skipped = 0
    if args.base_dataset:
        base = load_any_dataset(args.base_dataset)
        base_train = get_split(base, args.base_train_split)
        replay_train, train_skipped = build_replay_examples(
            base_train,
            args.base_train_max_samples,
            args.seed,
            "id_replay_train",
        )
        replay_skipped += train_skipped
        if isinstance(base, DatasetDict) and args.base_eval_split in base:
            base_eval = get_split(base, args.base_eval_split)
            replay_eval, eval_skipped = build_replay_examples(
                base_eval,
                args.base_eval_max_samples,
                args.seed + 1,
                "id_replay_eval",
            )
            replay_skipped += eval_skipped

    train_examples = replay_train + train_examples
    eval_examples = replay_eval + eval_examples
    random.Random(args.seed).shuffle(train_examples)
    random.Random(args.seed + 1).shuffle(eval_examples)

    output_splits = {"train": Dataset.from_list(train_examples)}
    if eval_examples:
        output_splits["validation"] = Dataset.from_list(eval_examples)
    output = DatasetDict(output_splits)

    if os.path.exists(args.output_dataset):
        shutil.rmtree(args.output_dataset)
    os.makedirs(os.path.dirname(args.output_dataset), exist_ok=True)
    output.save_to_disk(args.output_dataset)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "teacher_pairs_jsonl": args.teacher_pairs_jsonl,
        "output_dataset": args.output_dataset,
        "rubric_dir": args.rubric_dir,
        "label_source": args.label_source,
        "domain_rubric_fallback": args.domain_rubric_fallback,
        "training_prompt_mode": "raw" if args.no_training_rubric else "rubric",
        "reverse_pairs_enabled": args.add_reverse_pairs,
        "soft_router_labels_enabled": args.soft_router_labels,
        "router_original_blend_enabled": args.blend_router_with_original_score,
        "records_loaded": len(records),
        "selected_examples": len(selected_examples),
        "replay_train_examples": len(replay_train),
        "replay_eval_examples": len(replay_eval),
        "skipped": skipped,
        "replay_skipped": replay_skipped,
        "train": len(train_examples),
        "validation": len(eval_examples),
        "domains_with_specific_rubric": sorted(domain_map),
    }
    with open(os.path.join(args.output_dataset, "build_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print(f"Saved domain-rubric PM dataset to {args.output_dataset}")
    print(f"rubric_dir={args.rubric_dir}")
    print(
        f"selected_examples={len(selected_examples)} replay_train={len(replay_train)} "
        f"replay_eval={len(replay_eval)} train={len(train_examples)} validation={len(eval_examples)} "
        f"skipped={skipped} replay_skipped={replay_skipped}"
    )
    selected_labels = [float(item["label"]) for item in selected_examples]
    print(
        f"selected_label_min={min(selected_labels):.6f} "
        f"selected_label_mean={sum(selected_labels) / len(selected_labels):.6f} "
        f"selected_label_max={max(selected_labels):.6f}"
    )
    print(f"domains_with_specific_rubric={','.join(sorted(domain_map)) or 'none'}")


if __name__ == "__main__":
    main()
