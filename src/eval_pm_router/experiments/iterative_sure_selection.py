#!/usr/bin/env python3
"""Re-score unlabelled RM-Bench pairs with an updated rubric-conditioned PM."""

import argparse
from bisect import bisect_left, bisect_right
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import sys
from statistics import fmean

import numpy as np
import torch
import torch.distributed as dist
from datasets import DatasetDict, load_from_disk
from tqdm import tqdm

PPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PPO_ROOT / "src"
for path in (PPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_domain_rubric_pm_dataset import (  # noqa: E402
    build_domain_rubric_map,
    build_user_content,
    pick_domain_rubric,
    read_text,
)
from openrlhf.models import get_llm_for_sequence_regression  # noqa: E402
from openrlhf.utils import get_tokenizer  # noqa: E402


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"No records found in {path}")
    return rows


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_split(dataset, split):
    if isinstance(dataset, DatasetDict):
        if split not in dataset:
            raise KeyError(f"Split {split!r} not found; available: {list(dataset.keys())}")
        return dataset[split]
    return dataset


def unwrap(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def get_backbone(model):
    model = unwrap(model)
    for name in ("model", "base_model", "pretrained_model", "transformer"):
        candidate = getattr(model, name, None)
        if candidate is not None and candidate is not model:
            return candidate
    raise AttributeError("Could not find PM backbone")


def render(tokenizer, messages):
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def tokenize(tokenizer, texts, device, max_len):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    return {key: value.to(device) for key, value in inputs.items()}


def features(backbone, inputs):
    outputs = backbone(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        output_hidden_states=True,
        return_dict=True,
    )
    hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs.hidden_states[-1]
    return hidden[:, -1, :].float().cpu().numpy()


def batched_features(backbone, tokenizer, texts, device, batch_size, max_len, rank, desc):
    rows = []
    for start in tqdm(range(0, len(texts), batch_size), desc=f"{desc} rank {rank}", disable=(rank != 0)):
        inputs = tokenize(tokenizer, texts[start : start + batch_size], device, max_len)
        with torch.no_grad():
            rows.append(features(backbone, inputs))
    return np.concatenate(rows, axis=0)


def batched_scores(model, tokenizer, texts, device, batch_size, max_len, rank, desc):
    scores = []
    for start in tqdm(range(0, len(texts), batch_size), desc=f"{desc} rank {rank}", disable=(rank != 0)):
        inputs = tokenize(tokenizer, texts[start : start + batch_size], device, max_len)
        with torch.no_grad():
            score, _ = model.predict(inputs["input_ids"], inputs["attention_mask"])
        scores.extend(score.float().cpu().tolist())
    return scores


def fit_mahalanobis(anchor, test, ridge, pca_dim):
    mean = anchor.mean(axis=0, keepdims=True)
    scale = np.maximum(anchor.std(axis=0, keepdims=True), 1e-6)
    anchor = (anchor - mean) / scale
    test = (test - mean) / scale
    if 0 < pca_dim < anchor.shape[1]:
        _, _, vt = np.linalg.svd(anchor, full_matrices=False)
        components = vt[:pca_dim].T
        anchor, test = anchor @ components, test @ components
    center = anchor.mean(axis=0)
    covariance = (anchor - center).T @ (anchor - center) / max(1, len(anchor) - 1)
    covariance += ridge * np.eye(covariance.shape[0])
    precision = np.linalg.pinv(covariance)
    return (
        np.einsum("bi,ij,bj->b", anchor - center, precision, anchor - center),
        np.einsum("bi,ij,bj->b", test - center, precision, test - center),
    )


def swap_preference_roles(messages):
    swapped, found = [], set()
    for message in messages:
        row = dict(message)
        if row.get("role") == "assistant_1":
            row["role"] = "assistant_2"
            found.add("assistant_1")
        elif row.get("role") == "assistant_2":
            row["role"] = "assistant_1"
            found.add("assistant_2")
        swapped.append(row)
    if found != {"assistant_1", "assistant_2"}:
        raise ValueError("Every ID anchor must contain assistant_1 and assistant_2 messages")
    return swapped


def anchor_texts(dataset, tokenizer, context_key, max_samples):
    if context_key not in dataset.column_names:
        raise KeyError(f"Anchor dataset is missing {context_key!r}")
    if 0 < max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
    forward = [render(tokenizer, item[context_key]) for item in dataset]
    reverse = [render(tokenizer, swap_preference_roles(item[context_key])) for item in dataset]
    return forward, reverse


def group_ids(path):
    if not path:
        return set()
    return {str(row.get("group", row.get("prompt_index", "unknown"))) for row in read_jsonl(path)}


def pair_messages(item, general_rubric, domain_map, full_domain_rubric):
    domain = str(item.get("domain", "unknown"))
    domain_rubric = pick_domain_rubric(domain, domain_map, full_domain_rubric, "full")
    user_content = build_user_content(item, general_rubric, domain_rubric)
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant_1", "content": item["assistant_1"]},
        {"role": "assistant_2", "content": item["assistant_2"]},
    ]


def reverse_pair_messages(item, general_rubric, domain_map, full_domain_rubric):
    messages = pair_messages(item, general_rubric, domain_map, full_domain_rubric)
    messages[-2]["content"], messages[-1]["content"] = messages[-1]["content"], messages[-2]["content"]
    return messages


def main():
    parser = argparse.ArgumentParser(description="Select a fresh SURE batch from unlabelled RM-Bench prompt groups.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--candidate_samples", required=True)
    parser.add_argument("--excluded_selected", required=True)
    parser.add_argument("--rubric_dir", required=True)
    parser.add_argument("--anchor_dataset", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_fraction", type=float, required=True)
    parser.add_argument("--anchor_split", default="train")
    parser.add_argument("--anchor_context_key", default="context_messages")
    parser.add_argument("--anchor_max_samples", type=int, default=3840)
    parser.add_argument("--group_key", default="prompt_index")
    parser.add_argument("--group_reduce", choices=["max", "mean"], default="max")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--anchor_score_max_len", type=int, default=4096)
    parser.add_argument("--pair_score_max_len", type=int, default=8192)
    parser.add_argument("--feature_max_len", type=int, default=4096)
    parser.add_argument("--pca_dim", type=int, default=256)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--value_head_prefix", default="score")
    parser.add_argument("--tokenizer_chat_template", default=None)
    args = parser.parse_args()
    if not 0 < args.batch_fraction <= 1:
        raise ValueError("--batch_fraction must be in (0, 1]")

    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank >= 0:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
    rank = dist.get_rank() if local_rank >= 0 else 0
    world_size = dist.get_world_size() if local_rank >= 0 else 1
    device = torch.device(f"cuda:{local_rank}" if local_rank >= 0 else "cuda")

    general_rubric = read_text(os.path.join(args.rubric_dir, "general_rubric.md"))
    full_domain_rubric = read_text(os.path.join(args.rubric_dir, "domain_specific_rubric.md"))
    domain_map = build_domain_rubric_map(full_domain_rubric)
    excluded = group_ids(args.excluded_selected)
    candidates = read_jsonl(args.candidate_samples)
    total_groups = {str(row.get(args.group_key, "unknown")) for row in candidates}
    remaining = [row for row in candidates if str(row.get(args.group_key, "unknown")) not in excluded]
    remaining_groups = {str(row.get(args.group_key, "unknown")) for row in remaining}
    select_count = math.ceil(len(total_groups) * args.batch_fraction)
    if not remaining_groups:
        raise ValueError("No prompt groups remain after exclusion")
    if select_count > len(remaining_groups):
        raise ValueError(f"Requested {select_count} fresh groups but only {len(remaining_groups)} remain")

    model = get_llm_for_sequence_regression(
        args.model_path,
        "preference",
        bf16=True,
        use_flash_attention_2=True,
        value_head_prefix=args.value_head_prefix,
        normalize_reward=False,
    ).to(device)
    model.eval()
    tokenizer = get_tokenizer(args.model_path, model, "left", None, use_fast=True)
    if args.tokenizer_chat_template:
        tokenizer.chat_template = args.tokenizer_chat_template
    backbone = get_backbone(model)

    indexed = list(enumerate(remaining))[rank::world_size]
    local_indices = [index for index, _ in indexed]
    local_rows = [row for _, row in indexed]
    pair_text = [render(tokenizer, pair_messages(row, general_rubric, domain_map, full_domain_rubric)) for row in local_rows]
    reverse_text = [render(tokenizer, reverse_pair_messages(row, general_rubric, domain_map, full_domain_rubric)) for row in local_rows]
    anchor = get_split(load_from_disk(args.anchor_dataset), args.anchor_split)
    anchor_text, reverse_anchor_text = anchor_texts(anchor, tokenizer, args.anchor_context_key, args.anchor_max_samples)

    # The HelpSteer anchor serialization intentionally remains unchanged from
    # the established SURE selector; only candidate comparisons use R_t.
    anchor_features = batched_features(backbone, tokenizer, anchor_text, device, args.batch_size, args.feature_max_len, rank, "Scoring ID anchors")
    anchor_forward = batched_scores(model, tokenizer, anchor_text, device, args.batch_size, args.anchor_score_max_len, rank, "Scoring ID margins")
    anchor_reverse = batched_scores(model, tokenizer, reverse_anchor_text, device, args.batch_size, args.anchor_score_max_len, rank, "Scoring ID margins")
    pair_features = batched_features(backbone, tokenizer, pair_text, device, args.batch_size, args.feature_max_len, rank, "Scoring remaining pairs")
    forward = batched_scores(model, tokenizer, pair_text, device, args.batch_size, args.pair_score_max_len, rank, "Scoring PM margins")
    reverse = batched_scores(model, tokenizer, reverse_text, device, args.batch_size, args.pair_score_max_len, rank, "Scoring PM margins")
    anchor_distances, distances = fit_mahalanobis(anchor_features.astype(np.float64), pair_features.astype(np.float64), args.ridge, args.pca_dim)

    if local_rank >= 0:
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, (local_indices, forward, reverse, distances.tolist()))
        if rank != 0:
            dist.destroy_process_group()
            return
        merged = {}
        for indices, left, right, local_distances in gathered:
            for index, left_score, right_score, distance in zip(indices, left, right, local_distances):
                merged[index] = (left_score, right_score, distance)
        forward = [merged[index][0] for index in range(len(remaining))]
        reverse = [merged[index][1] for index in range(len(remaining))]
        distances = [merged[index][2] for index in range(len(remaining))]
    else:
        distances = distances.tolist()

    anchor_scores = [(left - right) / 2.0 for left, right in zip(anchor_forward, anchor_reverse)]
    sorted_anchor_maha = sorted(anchor_distances.tolist())
    sorted_anchor_abs = sorted(abs(score) for score in anchor_scores)
    rows = []
    for item, left, right, distance in zip(remaining, forward, reverse, distances):
        score = (left - right) / 2.0
        maha_risk = bisect_left(sorted_anchor_maha, distance) / len(sorted_anchor_maha)
        margin_risk = 1.0 - bisect_right(sorted_anchor_abs, abs(score)) / len(sorted_anchor_abs)
        row = dict(item)
        row.update(
            score=float(score),
            abs_score=abs(float(score)),
            original_pm_score=float(score),
            epistemic_uncertainty=float(distance),
            mahalanobis_ood_score=float(maha_risk),
            mahalanobis_risk=float(maha_risk),
            margin_risk=float(margin_risk),
            selection_score=float((maha_risk + margin_risk) / 2.0),
            acquisition_method="id_calibrated_mean_mahalanobis_margin",
            scoring_model=args.model_path,
            scoring_rubric=args.rubric_dir,
        )
        rows.append(row)

    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row.get(args.group_key, "unknown"))].append(row)
    groups = []
    for group, group_rows in grouped.items():
        values = [row["selection_score"] for row in group_rows]
        representative = max(group_rows, key=lambda row: row["selection_score"])
        groups.append({
            "group": group,
            "prompt_index": representative.get("prompt_index", group),
            "pair_count": len(group_rows),
            "group_score": max(values) if args.group_reduce == "max" else fmean(values),
            "selection_score_max": max(values),
            "selection_score_mean": fmean(values),
            "mahalanobis_ood_score_max": max(row["mahalanobis_ood_score"] for row in group_rows),
            "mahalanobis_ood_score_mean": fmean(row["mahalanobis_ood_score"] for row in group_rows),
            "margin_risk_max": max(row["margin_risk"] for row in group_rows),
            "margin_risk_mean": fmean(row["margin_risk"] for row in group_rows),
        })
    groups.sort(key=lambda row: row["group_score"], reverse=True)
    selected_groups = {row["group"] for row in groups[:select_count]}
    ordered_rows = sorted(rows, key=lambda row: row["selection_score"], reverse=True)
    for index, row in enumerate(ordered_rows):
        row["uncertainty_rank"] = index
        row["selected_for_judge"] = str(row.get(args.group_key, "unknown")) in selected_groups
    selected_rows = [row for row in ordered_rows if row["selected_for_judge"]]

    write_jsonl(os.path.join(args.output_dir, "scored_samples.jsonl"), ordered_rows)
    write_jsonl(os.path.join(args.output_dir, "selected_samples.jsonl"), selected_rows)
    write_jsonl(os.path.join(args.output_dir, "scored_groups.jsonl"), groups)
    write_jsonl(os.path.join(args.output_dir, "selected_groups.jsonl"), groups[:select_count])
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "selection_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump({
            "model_path": args.model_path,
            "rubric_dir": args.rubric_dir,
            "candidate_samples": args.candidate_samples,
            "excluded_selected": args.excluded_selected,
            "total_groups": len(total_groups),
            "excluded_groups": len(excluded),
            "eligible_groups": len(remaining_groups),
            "selected_groups": select_count,
            "batch_fraction_of_original_pool": args.batch_fraction,
            "acquisition_method": "id_calibrated_mean_mahalanobis_margin",
            "anchor_serialization": "established_helpsteer_context_messages",
        }, handle, ensure_ascii=False, indent=2)
    print(f"Rescored {len(remaining)} pairs from {len(remaining_groups)} remaining prompt groups")
    print(f"Selected {select_count} fresh groups ({len(selected_rows)} pairs) using the updated PM and rubric")
    if local_rank >= 0:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
