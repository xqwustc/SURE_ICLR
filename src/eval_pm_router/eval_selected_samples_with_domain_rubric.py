#!/usr/bin/env python3
import argparse
import json
import os
import re
from collections import defaultdict
from datetime import datetime

import torch
import torch.distributed as dist
from tqdm import tqdm

from openrlhf.models import get_llm_for_sequence_regression
from openrlhf.utils import get_tokenizer


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


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)


def write_jsonl(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


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
    domain_map = {}
    for title, body in split_markdown_sections(domain_text):
        haystack = f"{title}\n{body}".lower()
        for domain, aliases in DOMAIN_ALIASES.items():
            if any(alias in haystack for alias in aliases):
                domain_map.setdefault(domain, []).append(f"## {title}\n{body}")
    return {domain: "\n\n".join(parts) for domain, parts in domain_map.items()}


def pick_domain_rubric(domain, domain_map):
    domain = str(domain or "").strip()
    if domain in domain_map:
        return domain_map[domain]
    if domain.startswith("safety") and "safety-refuse" in domain_map:
        return domain_map["safety-refuse"]
    return ""


def build_user_content(item, general_rubric, domain_map, use_rubric):
    if not use_rubric:
        return item.get("prompt", "")
    domain_rubric = pick_domain_rubric(item.get("domain"), domain_map)
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


def build_pair_texts(tokenizer, prompt, assistant_1, assistant_2):
    forward = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant_1", "content": assistant_1},
            {"role": "assistant_2", "content": assistant_2},
        ],
        tokenize=False,
    )
    reverse = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant_1", "content": assistant_2},
            {"role": "assistant_2", "content": assistant_1},
        ],
        tokenize=False,
    )
    return forward, reverse


def bool_value(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    return str(value).strip().lower() in {"true", "1", "yes"}


def main():
    parser = argparse.ArgumentParser(description="Evaluate selected RM-Bench samples and export remaining errors for active learning.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--rubric_root", default="./artifacts/rubrics")
    parser.add_argument("--rubric_dir", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=8192)
    parser.add_argument("--value_head_prefix", default="score")
    parser.add_argument("--use_laplace", action="store_true", default=True)
    parser.add_argument("--laplace_ridge", type=float, default=0.001)
    parser.add_argument("--laplace_amplitude", type=float, default=0.1)
    parser.add_argument("--tokenizer_chat_template", default=None)
    parser.add_argument("--no_rubric", action="store_true", default=False)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank() if local_rank != -1 else 0
    world_size = dist.get_world_size() if local_rank != -1 else 1
    device = torch.device(f"cuda:{local_rank}" if local_rank != -1 else ("cuda" if torch.cuda.is_available() else "cpu"))

    rubric_dir = args.rubric_dir or latest_rubric_dir(args.rubric_root)
    general_rubric = read_text(os.path.join(rubric_dir, "general_rubric.md"))
    domain_map = build_domain_rubric_map(read_text(os.path.join(rubric_dir, "domain_specific_rubric.md")))
    if not args.no_rubric and not general_rubric:
        raise FileNotFoundError(f"Missing general_rubric.md in {rubric_dir}")

    model = get_llm_for_sequence_regression(
        args.model_path,
        "preference",
        bf16=torch.cuda.is_available(),
        use_flash_attention_2=torch.cuda.is_available(),
        value_head_prefix=args.value_head_prefix,
        use_laplace=args.use_laplace,
        laplace_ridge=args.laplace_ridge,
        laplace_amplitude=args.laplace_amplitude,
        normalize_reward=False,
    ).to(device)
    model.eval()
    tokenizer = get_tokenizer(args.model_path, model, "left", None, use_fast=True)
    if args.tokenizer_chat_template:
        tokenizer.chat_template = args.tokenizer_chat_template

    records = read_jsonl(args.input_path)
    if args.limit > 0:
        records = records[: args.limit]
    local_records = records[rank::world_size]

    local_results = []
    with torch.no_grad():
        for start in tqdm(range(0, len(local_records), args.batch_size), desc=f"Selected eval rank {rank}", disable=(rank != 0)):
            batch = local_records[start : start + args.batch_size]
            forward_texts = []
            reverse_texts = []
            for item in batch:
                prompt = build_user_content(item, general_rubric, domain_map, not args.no_rubric)
                forward, reverse = build_pair_texts(tokenizer, prompt, item.get("assistant_1", ""), item.get("assistant_2", ""))
                forward_texts.append(forward)
                reverse_texts.append(reverse)

            inputs_1 = tokenizer(forward_texts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_len)
            inputs_2 = tokenizer(reverse_texts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_len)
            inputs_1 = {key: value.to(device) for key, value in inputs_1.items()}
            inputs_2 = {key: value.to(device) for key, value in inputs_2.items()}
            scores_1, var_1 = model.predict(inputs_1["input_ids"], inputs_1["attention_mask"])
            scores_2, var_2 = model.predict(inputs_2["input_ids"], inputs_2["attention_mask"])
            scores = ((scores_1 - scores_2) / 2).detach().float().cpu().tolist()
            uncertainties = (var_1 + var_2).detach().float().cpu().tolist()

            for item, score, uncertainty in zip(batch, scores, uncertainties):
                prediction = "assistant_1" if score >= 0 else "assistant_2"
                result = dict(item)
                result["active_eval_score"] = float(score)
                result["active_eval_uncertainty"] = float(uncertainty)
                result["active_eval_prediction"] = prediction
                result["active_eval_is_correct"] = prediction == result.get("ground_truth")
                result["previous_is_correct"] = bool_value(result.get("is_correct"))
                result["rubric_dir"] = rubric_dir
                local_results.append(result)

    if local_rank != -1:
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, local_results)
        if rank == 0:
            results = [item for rows in gathered for item in rows]
        else:
            dist.destroy_process_group()
            return
    else:
        results = local_results

    results.sort(key=lambda item: item.get("pair_index", 0))
    wrong = [item for item in results if not item["active_eval_is_correct"]]
    fixed = [item for item in results if item.get("previous_is_correct") is False and item["active_eval_is_correct"]]
    domain_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for item in results:
        domain_stats[item.get("domain", "unknown")]["total"] += 1
        domain_stats[item.get("domain", "unknown")]["correct"] += int(item["active_eval_is_correct"])

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_path": args.model_path,
        "input_path": args.input_path,
        "rubric_dir": rubric_dir,
        "use_rubric": not args.no_rubric,
        "domains_with_specific_rubric": sorted(domain_map),
        "accuracy": (len(results) - len(wrong)) / len(results) if results else 0.0,
        "correct": len(results) - len(wrong),
        "wrong": len(wrong),
        "total": len(results),
        "previous_wrong_fixed": len(fixed),
        "domain_accuracy": {
            domain: {
                "accuracy": stats["correct"] / stats["total"] if stats["total"] else 0.0,
                "correct": stats["correct"],
                "total": stats["total"],
            }
            for domain, stats in sorted(domain_stats.items())
        },
    }

    write_jsonl(os.path.join(args.save_path, "selected_pair_results.jsonl"), results)
    write_jsonl(os.path.join(args.save_path, "wrong_samples.jsonl"), wrong)
    write_jsonl(os.path.join(args.save_path, "fixed_previous_wrong_samples.jsonl"), fixed)
    write_json(os.path.join(args.save_path, "summary.json"), summary)

    if rank == 0:
        print(f"selected_accuracy={summary['accuracy']:.6f} correct={summary['correct']} wrong={summary['wrong']} total={summary['total']}")
        print(f"previous_wrong_fixed={summary['previous_wrong_fixed']}")
        for domain, stats in summary["domain_accuracy"].items():
            print(f"{domain}: accuracy={stats['accuracy']:.6f} correct={stats['correct']} total={stats['total']}")
        print(f"Saved to {args.save_path}")

    if local_rank != -1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
