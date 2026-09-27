#!/usr/bin/env python3
import argparse
import json
import os
import re
from collections import defaultdict
from datetime import timedelta

if os.environ.get("RMBENCH_OFFLINE", "1") != "0":
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
import torch.distributed as dist
from datasets import DownloadMode, load_dataset, load_from_disk
from tqdm import tqdm

from openrlhf.models import get_llm_for_sequence_regression
from openrlhf.utils import get_tokenizer
from openrlhf.utils.rmbench_utils import compute_accuracy


EXCEL_COLUMNS = [
    "model",
    "chat",
    "math",
    "code",
    "safety",
    "hard_acc",
    "normal_acc",
    "easy_acc",
    "total_avg_acc",
    "ece",
    "uncertainty_correlation",
    "total_samples",
    "total_prompts",
]


DOMAIN_ALIASES = {
    "chat": ["chat", "conversation", "general"],
    "code": ["code", "coding", "programming"],
    "math": ["math", "mathematical", "calculation"],
    "safety-refuse": ["safety-refuse", "refuse", "refusal", "safety"],
    "safety-response": ["safety-response", "safe response", "safety"],
}


def _excel_value(value):
    if value is None:
        return "NA"
    try:
        import numpy as np

        if isinstance(value, np.generic):
            value = value.item()
    except Exception:
        pass
    if isinstance(value, float):
        return "NA" if not torch.isfinite(torch.tensor(value)) else f"{value:.6f}"
    if isinstance(value, int):
        return str(value)
    return str(value)


def infer_model_name(model_path):
    normalized = os.path.normpath(model_path)
    name = os.path.basename(normalized)
    parent = os.path.basename(os.path.dirname(normalized))
    if name.startswith("round_") and parent:
        return f"{parent}/{name}"
    return name


def print_excel_frame(results, model_path, save_path):
    model_name = infer_model_name(model_path)
    row = {"model": model_name, **results}
    header = "\t".join(EXCEL_COLUMNS)
    values = "\t".join(_excel_value(row.get(column)) for column in EXCEL_COLUMNS)
    frame = f"{header}\n{values}"

    print("\nExcel TSV frame (copy and paste into Excel):")
    print(frame)

    os.makedirs(save_path, exist_ok=True)
    frame_path = os.path.join(save_path, "excel_frame.tsv")
    with open(frame_path, "w", encoding="utf-8", newline="") as handle:
        handle.write(frame + "\n")
    print(f"Saved Excel TSV frame to {frame_path}")


def load_rmbench_split(dataset_ref, split, cache_dir):
    if os.path.isdir(dataset_ref):
        dataset = load_from_disk(dataset_ref)
        if hasattr(dataset, "keys"):
            return dataset[split]
        if split != "train":
            raise KeyError(f"Local RM-Bench dataset has no split {split!r}")
        return dataset

    return load_dataset(
        dataset_ref,
        split=split,
        cache_dir=cache_dir or None,
        download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
    )


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
    sections = split_markdown_sections(domain_text)
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


def pick_domain_rubric(domain, domain_map):
    domain = str(domain or "").strip()
    if domain in domain_map:
        return domain_map[domain]
    if domain.startswith("safety") and "safety-refuse" in domain_map:
        return domain_map["safety-refuse"]
    return ""


def build_user_content(prompt, domain, general_rubric, domain_map):
    parts = ["Use the rubric below to compare the two assistant responses."]
    if general_rubric:
        parts.extend(["", "[General Rubric]", general_rubric])
    domain_rubric = pick_domain_rubric(domain, domain_map)
    if domain_rubric:
        parts.extend(["", f"[Domain-Specific Rubric: {domain}]", domain_rubric])
    parts.extend(["", "[User Request]", prompt])
    return "\n".join(parts)


def build_pairs(dataset, tokenizer, general_rubric, domain_map):
    pairs = []
    for prompt_index, item in enumerate(dataset):
        prompt = item["prompt"]
        domain = item["domain"]
        for chosen_index, chosen in enumerate(item["chosen"]):
            for rejected_index, rejected in enumerate(item["rejected"]):
                rubric_prompt = build_user_content(prompt, domain, general_rubric, domain_map)
                query = tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": rubric_prompt},
                        {"role": "assistant_1", "content": chosen},
                        {"role": "assistant_2", "content": rejected},
                    ],
                    tokenize=False,
                )
                reverse_query = tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": rubric_prompt},
                        {"role": "assistant_1", "content": rejected},
                        {"role": "assistant_2", "content": chosen},
                    ],
                    tokenize=False,
                )
                pairs.append(
                    {
                        "prompt_index": prompt_index,
                        "pair_index": len(pairs),
                        "domain": domain,
                        "chosen_index": chosen_index,
                        "rejected_index": rejected_index,
                        "query": query,
                        "reverse_query": reverse_query,
                    }
                )
    return pairs


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)


def write_jsonl(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a domain-rubric-continued preference model on RM-Bench.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--dataset", default="THU-KEG/RM-Bench")
    parser.add_argument("--split", default="train")
    parser.add_argument("--cache_dir", default=os.environ.get("RMBENCH_CACHE_DIR", os.path.expanduser("~/.cache/huggingface/datasets")))
    parser.add_argument("--dist_timeout_minutes", type=int, default=120)
    parser.add_argument("--rubric_root", default="./artifacts/rubrics")
    parser.add_argument("--rubric_dir", default="")
    parser.add_argument("--general_only", action="store_true", default=False, help="Use only general_rubric.md and ignore domain-specific criteria.")
    parser.add_argument("--domain_only", action="store_true", default=False, help="Use only domain_specific_rubric.md and ignore general criteria.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=8192)
    parser.add_argument("--max_prompts", type=int, default=0)
    parser.add_argument("--value_head_prefix", default="score")
    parser.add_argument("--use_laplace", action="store_true", default=False)
    parser.add_argument("--laplace_ridge", type=float, default=0.001)
    parser.add_argument("--laplace_amplitude", type=float, default=0.1)
    parser.add_argument("--tokenizer_chat_template", default=None)
    args = parser.parse_args()
    if args.general_only and args.domain_only:
        raise ValueError("--general_only and --domain_only cannot be used together")

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=args.dist_timeout_minutes))
    rank = dist.get_rank() if local_rank != -1 else 0
    world_size = dist.get_world_size() if local_rank != -1 else 1
    device = torch.device(f"cuda:{local_rank}" if local_rank != -1 else ("cuda" if torch.cuda.is_available() else "cpu"))

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

    rubric_dir = args.rubric_dir or latest_rubric_dir(args.rubric_root)
    general_rubric = "" if args.domain_only else read_text(os.path.join(rubric_dir, "general_rubric.md"))
    domain_rubric = "" if args.general_only else read_text(os.path.join(rubric_dir, "domain_specific_rubric.md"))
    if not general_rubric and not domain_rubric:
        raise FileNotFoundError(f"No selected rubric content found in {rubric_dir}")
    domain_map = build_domain_rubric_map(domain_rubric)

    dataset = load_rmbench_split(args.dataset, args.split, args.cache_dir)
    if args.max_prompts > 0:
        dataset = dataset.select(range(min(args.max_prompts, len(dataset))))

    pairs = build_pairs(dataset, tokenizer, general_rubric, domain_map)
    local_pairs = pairs[rank::world_size]

    local_results = []
    with torch.no_grad():
        for start in tqdm(range(0, len(local_pairs), args.batch_size), desc=f"Evaluating rank {rank}", disable=(rank != 0)):
            batch = local_pairs[start : start + args.batch_size]
            inputs_1 = tokenizer(
                [item["query"] for item in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_len,
            )
            inputs_2 = tokenizer(
                [item["reverse_query"] for item in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_len,
            )
            inputs_1 = {key: value.to(device) for key, value in inputs_1.items()}
            inputs_2 = {key: value.to(device) for key, value in inputs_2.items()}
            scores_1, var_1 = model.predict(inputs_1["input_ids"], inputs_1["attention_mask"])
            scores_2, var_2 = model.predict(inputs_2["input_ids"], inputs_2["attention_mask"])
            scores = ((scores_1 - scores_2) / 2).detach().float().cpu().tolist()
            uncertainties = (var_1 + var_2).detach().float().cpu().tolist()

            for item, score, uncertainty in zip(batch, scores, uncertainties):
                prediction = "assistant_1" if score >= 0 else "assistant_2"
                local_results.append(
                    {
                        "prompt_index": item["prompt_index"],
                        "pair_index": item["pair_index"],
                        "domain": item["domain"],
                        "chosen_index": item["chosen_index"],
                        "rejected_index": item["rejected_index"],
                        "score": float(score),
                        "uncertainty": float(uncertainty),
                        "model_prediction": prediction,
                        "ground_truth": "assistant_1",
                        "is_correct": prediction == "assistant_1",
                    }
                )

    if local_rank != -1:
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, local_results)
        if rank == 0:
            results = [item for rank_results in gathered for item in rank_results]
        else:
            dist.destroy_process_group()
            return
    else:
        results = local_results

    results.sort(key=lambda item: item["pair_index"])
    total = len(results)
    correct = sum(1 for item in results if item["is_correct"])
    domain_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for item in results:
        domain_stats[item["domain"]]["total"] += 1
        domain_stats[item["domain"]]["correct"] += int(item["is_correct"])

    prompt_results = []
    by_prompt = defaultdict(list)
    for item in results:
        by_prompt[item["prompt_index"]].append(item)
    for prompt_index, items in by_prompt.items():
        matrix = [[0.0 for _ in range(3)] for _ in range(3)]
        domain = items[0]["domain"]
        for item in items:
            matrix[int(item["chosen_index"])][int(item["rejected_index"])] = item["score"]
        prompt_results.append({"prompt_index": prompt_index, "domain": domain, "reward_diff": matrix})
    rmbench_metrics = compute_accuracy(prompt_results, model_type="preference")

    summary = {
        "model_path": args.model_path,
        "dataset": args.dataset,
        "split": args.split,
        "rubric_dir": rubric_dir,
        "domains_with_specific_rubric": sorted(domain_map),
        "pair_accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "total_prompts": len(prompt_results),
        "rmbench_official_metrics": rmbench_metrics,
        "domain_accuracy": {
            domain: {
                "accuracy": stats["correct"] / stats["total"] if stats["total"] else 0.0,
                "correct": stats["correct"],
                "total": stats["total"],
            }
            for domain, stats in sorted(domain_stats.items())
        },
    }

    write_jsonl(os.path.join(args.save_path, "pair_results.jsonl"), results)
    write_json(os.path.join(args.save_path, "summary.json"), summary)
    excel_results = dict(rmbench_metrics)
    excel_results["ece"] = None
    excel_results["uncertainty_correlation"] = None
    excel_results["total_samples"] = total
    excel_results["total_prompts"] = len(prompt_results)
    print(f"pair_accuracy={summary['pair_accuracy']:.6f} correct={correct} total={total}")
    print("RM-Bench official-style metrics:")
    for key, value in rmbench_metrics.items():
        print(f"{key}: {value}")
    for domain, stats in summary["domain_accuracy"].items():
        print(f"{domain}: accuracy={stats['accuracy']:.6f} correct={stats['correct']} total={stats['total']}")
    print(f"domains_with_specific_rubric={','.join(sorted(domain_map)) or 'none'}")
    print(f"Saved to {args.save_path}")
    print_excel_frame(excel_results, args.model_path, args.save_path)

    if local_rank != -1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
