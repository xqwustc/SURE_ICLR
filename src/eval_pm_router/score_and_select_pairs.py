#!/usr/bin/env python3
"""Score offline rollout pairs with an original PM and select top Mahalanobis-OOD pairs."""

import argparse
from bisect import bisect_left, bisect_right
from collections import defaultdict
import json
import math
import os
from statistics import fmean

import numpy as np
import torch
import torch.distributed as dist
from datasets import DatasetDict, load_from_disk
from tqdm import tqdm

from openrlhf.models import get_llm_for_sequence_regression
from openrlhf.utils import get_tokenizer


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"No pairs found in {path}")
    return rows


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


def pair_messages(item):
    context = item.get("context_messages")
    if not isinstance(context, list) or not context:
        context = [{"role": "user", "content": item["prompt"]}]
    return context + [
        {"role": "assistant_1", "content": item["assistant_1"]},
        {"role": "assistant_2", "content": item["assistant_2"]},
    ]


def reverse_pair_messages(item):
    context = item.get("context_messages")
    if not isinstance(context, list) or not context:
        context = [{"role": "user", "content": item["prompt"]}]
    return context + [
        {"role": "assistant_1", "content": item["assistant_2"]},
        {"role": "assistant_2", "content": item["assistant_1"]},
    ]

def render(tokenizer, messages):
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def tokenize(tokenizer, texts, device, max_len):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    return {key: value.to(device) for key, value in inputs.items()}


def features(backbone, inputs):
    outputs = backbone(
        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], output_hidden_states=True, return_dict=True
    )
    hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs.hidden_states[-1]
    return hidden[:, -1, :].float().cpu().numpy()


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
    delta = anchor - center
    covariance = delta.T @ delta / max(1, len(anchor) - 1)
    covariance += ridge * np.eye(covariance.shape[0])
    precision = np.linalg.pinv(covariance)
    anchor_distance = anchor - center
    test_distance = test - center
    return (
        np.einsum("bi,ij,bj->b", anchor_distance, precision, anchor_distance),
        np.einsum("bi,ij,bj->b", test_distance, precision, test_distance),
    )


def swap_preference_roles(messages):
    swapped = []
    found = set()
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


def anchor_texts(dataset, tokenizer, context_key, max_samples, seed):
    if context_key not in dataset.column_names:
        raise KeyError(f"Anchor dataset is missing {context_key!r}")
    if max_samples > 0 and max_samples < len(dataset):
        # Match the established anchor dump: take the first fixed ID subset.
        dataset = dataset.select(range(max_samples))
    forward = [render(tokenizer, item[context_key]) for item in dataset]
    reverse = [render(tokenizer, swap_preference_roles(item[context_key])) for item in dataset]
    return forward, reverse


def batched_features(backbone, tokenizer, texts, device, batch_size, max_len, with_scores=False, model=None, rank=0):
    output_features, output_scores = [], []
    for start in tqdm(range(0, len(texts), batch_size), desc=f"Scoring original PM rank {rank}", disable=(rank != 0)):
        inputs = tokenize(tokenizer, texts[start : start + batch_size], device, max_len)
        with torch.no_grad():
            output_features.append(features(backbone, inputs))
            if with_scores:
                score, _ = model.predict(inputs["input_ids"], inputs["attention_mask"])
                output_scores.extend(score.float().cpu().tolist())
    return np.concatenate(output_features, axis=0), output_scores


def batched_scores(model, tokenizer, texts, device, batch_size, max_len, rank):
    scores = []
    for start in tqdm(range(0, len(texts), batch_size), desc=f"Scoring PM margins rank {rank}", disable=(rank != 0)):
        inputs = tokenize(tokenizer, texts[start : start + batch_size], device, max_len)
        with torch.no_grad():
            score, _ = model.predict(inputs["input_ids"], inputs["attention_mask"])
        scores.extend(score.float().cpu().tolist())
    return scores


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Rank rollout pairs by original-PM Mahalanobis epistemic uncertainty.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--anchor_dataset", required=True, help="In-distribution PM training data, e.g. HelpSteer3 comparison.")
    parser.add_argument("--anchor_split", default="train")
    parser.add_argument("--anchor_context_key", default="context_messages")
    parser.add_argument("--anchor_max_samples", type=int, default=3840)
    parser.add_argument("--input_pairs", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--top_fraction", type=float, required=True, help="Fraction of prompt groups to send to the judge.")
    parser.add_argument("--group_key", default="prompt_index")
    parser.add_argument("--group_reduce", choices=["max", "mean"], default="max")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--anchor_score_max_len", type=int, default=4096)
    parser.add_argument("--pair_score_max_len", type=int, default=8192)
    parser.add_argument("--feature_max_len", type=int, default=4096)
    parser.add_argument("--pca_dim", type=int, default=256)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--value_head_prefix", default="score")
    parser.add_argument(
        "--tokenizer_chat_template",
        default="{% for message in messages %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}",
    )
    args = parser.parse_args()
    if not 0.0 < args.top_fraction <= 1.0:
        raise ValueError("--top_fraction must be in (0, 1]")

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank >= 0:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank() if local_rank >= 0 else 0
    world_size = dist.get_world_size() if local_rank >= 0 else 1
    device = torch.device(f"cuda:{local_rank}" if local_rank >= 0 else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = get_llm_for_sequence_regression(
        args.model_path, "preference", bf16=torch.cuda.is_available(), use_flash_attention_2=torch.cuda.is_available(),
        value_head_prefix=args.value_head_prefix, normalize_reward=False,
    ).to(device)
    model.eval()
    tokenizer = get_tokenizer(args.model_path, model, "left", None, use_fast=True)
    tokenizer.chat_template = args.tokenizer_chat_template
    backbone = get_backbone(model)

    pairs = read_jsonl(args.input_pairs)
    indexed_pairs = list(enumerate(pairs))[rank::world_size]
    local_indices = [index for index, _ in indexed_pairs]
    local_pairs = [pair for _, pair in indexed_pairs]
    anchor = get_split(load_from_disk(args.anchor_dataset), args.anchor_split)
    anchor_text, reverse_anchor_text = anchor_texts(
        anchor, tokenizer, args.anchor_context_key, args.anchor_max_samples, args.seed
    )
    pair_text = [render(tokenizer, pair_messages(item)) for item in local_pairs]
    reverse_pair_text = [render(tokenizer, reverse_pair_messages(item)) for item in local_pairs]
    anchor_features, _ = batched_features(
        backbone, tokenizer, anchor_text, device, args.batch_size, args.feature_max_len, rank=rank
    )
    anchor_forward_scores = batched_scores(
        model, tokenizer, anchor_text, device, args.batch_size, args.anchor_score_max_len, rank
    )
    anchor_reverse_scores = batched_scores(
        model, tokenizer, reverse_anchor_text, device, args.batch_size, args.anchor_score_max_len, rank
    )
    pair_features, _ = batched_features(
        backbone, tokenizer, pair_text, device, args.batch_size, args.feature_max_len, rank=rank
    )
    forward_scores = batched_scores(
        model, tokenizer, pair_text, device, args.batch_size, args.pair_score_max_len, rank
    )
    reverse_scores = batched_scores(
        model, tokenizer, reverse_pair_text, device, args.batch_size, args.pair_score_max_len, rank
    )
    anchor_distances, local_distances = fit_mahalanobis(
        anchor_features.astype(np.float64), pair_features.astype(np.float64), args.ridge, args.pca_dim
    )
    local_distances = local_distances.tolist()
    if local_rank >= 0:
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, (local_indices, forward_scores, reverse_scores, local_distances))
        if rank != 0:
            dist.destroy_process_group()
            return
        by_index = {}
        for indices, forward, reverse, distances in gathered:
            for index, forward_score, reverse_score, distance in zip(indices, forward, reverse, distances):
                by_index[index] = (forward_score, reverse_score, distance)
        forward_scores = [by_index[index][0] for index in range(len(pairs))]
        reverse_scores = [by_index[index][1] for index in range(len(pairs))]
        distances = [by_index[index][2] for index in range(len(pairs))]
    else:
        distances = local_distances

    # Match the established selector: both risks are calibrated against the
    # fixed ID anchor distribution, not against this rollout batch.
    sorted_anchor_mahalanobis = sorted(anchor_distances.tolist())
    anchor_scores = [
        (forward - reverse_) / 2.0 for forward, reverse_ in zip(anchor_forward_scores, anchor_reverse_scores)
    ]
    sorted_anchor_abs_scores = sorted(abs(score) for score in anchor_scores)
    symmetric_scores = [(forward - reverse_) / 2.0 for forward, reverse_ in zip(forward_scores, reverse_scores)]
    mahalanobis_risk = [
        bisect_left(sorted_anchor_mahalanobis, distance) / len(sorted_anchor_mahalanobis) for distance in distances
    ]
    margin_risk = [
        1.0 - bisect_right(sorted_anchor_abs_scores, abs(score)) / len(sorted_anchor_abs_scores)
        for score in symmetric_scores
    ]
    scored_rows = []
    for item, score, distance, maha_risk, pair_margin_risk in zip(
        pairs, symmetric_scores, distances, mahalanobis_risk, margin_risk
    ):
        row = dict(item)
        row["score"] = float(score)
        row["abs_score"] = abs(float(score))
        row["original_pm_score"] = float(score)
        row["epistemic_uncertainty"] = float(distance)
        row["mahalanobis_ood_score"] = float(maha_risk)
        row["mahalanobis_risk"] = float(maha_risk)
        row["margin_risk"] = float(pair_margin_risk)
        row["selection_score"] = float((maha_risk + pair_margin_risk) / 2.0)
        row["acquisition_method"] = "id_calibrated_mean_mahalanobis_margin"
        scored_rows.append(row)

    # Match the established selection unit: a prompt is selected if its
    # highest-risk response pair is in the top fraction of prompt groups.
    groups = defaultdict(list)
    for row in scored_rows:
        groups[str(row.get(args.group_key, "unknown"))].append(row)
    group_rows = []
    for group, rows in groups.items():
        scores = [row["selection_score"] for row in rows]
        group_score = max(scores) if args.group_reduce == "max" else fmean(scores)
        representative = max(rows, key=lambda row: row["selection_score"])
        group_rows.append(
            {
                "group": group,
                "group_score": group_score,
                "pair_count": len(rows),
                "prompt_index": representative.get("prompt_index", group),
                "representative_pair_index": representative.get("pair_index"),
                "selection_score_max": max(scores),
                "selection_score_mean": fmean(scores),
                "mahalanobis_ood_score_max": max(row["mahalanobis_ood_score"] for row in rows),
                "mahalanobis_ood_score_mean": fmean(row["mahalanobis_ood_score"] for row in rows),
                "margin_risk_max": max(row["margin_risk"] for row in rows),
                "margin_risk_mean": fmean(row["margin_risk"] for row in rows),
            }
        )
    ordered_groups = sorted(group_rows, key=lambda row: row["group_score"], reverse=True)
    selected_group_count = max(1, math.ceil(len(ordered_groups) * args.top_fraction))
    selected_groups = {row["group"] for row in ordered_groups[:selected_group_count]}
    for row in scored_rows:
        row["selected_for_judge"] = str(row.get(args.group_key, "unknown")) in selected_groups

    ordered = sorted(scored_rows, key=lambda item: item["selection_score"], reverse=True)
    for rank, item in enumerate(ordered):
        item["uncertainty_rank"] = rank
    selected = [row for row in scored_rows if row["selected_for_judge"]]
    write_jsonl(os.path.join(args.output_dir, "scored_pairs.jsonl"), ordered)
    write_jsonl(os.path.join(args.output_dir, "selected_pairs.jsonl"), selected)
    write_jsonl(os.path.join(args.output_dir, "scored_groups.jsonl"), ordered_groups)
    write_jsonl(os.path.join(args.output_dir, "selected_groups.jsonl"), ordered_groups[:selected_group_count])
    metadata = {
        "original_pm": args.model_path, "anchor_dataset": args.anchor_dataset, "anchor_split": args.anchor_split,
        "anchor_max_samples": args.anchor_max_samples, "candidate_pairs": len(pairs), "selected_pairs": len(selected),
        "candidate_groups": len(ordered_groups), "selected_groups": selected_group_count,
        "top_fraction": args.top_fraction, "group_key": args.group_key, "group_reduce": args.group_reduce,
        "acquisition_method": "id_calibrated_mean_mahalanobis_margin", "pca_dim": args.pca_dim, "ridge": args.ridge,
        "anchor_score_max_len": args.anchor_score_max_len, "pair_score_max_len": args.pair_score_max_len,
        "feature_max_len": args.feature_max_len, "anchor_selection": "first_n_train_examples",
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "selection_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(f"Scored {len(ordered)} candidate pairs against {len(anchor_text)} ID anchors")
    print(f"Selected {selected_group_count}/{len(ordered_groups)} prompt groups ({len(selected)} pairs) by dual-rank Mahalanobis + low-margin uncertainty")
    print(f"Saved to {args.output_dir}")
    if local_rank >= 0:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
