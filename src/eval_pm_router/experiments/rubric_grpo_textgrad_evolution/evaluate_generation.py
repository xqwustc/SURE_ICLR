#!/usr/bin/env python3
import argparse
import json
import math
import os
import shutil
from collections import defaultdict
from functools import lru_cache

import torch
import torch.distributed as dist
from tqdm import tqdm

from openrlhf.models import get_llm_for_sequence_regression
from openrlhf.utils import get_tokenizer
from src.eval_pm_router.eval_selected_samples_with_domain_rubric import (
    build_domain_rubric_map,
    build_pair_texts,
    build_user_content,
    read_jsonl,
    read_text,
)


def rubric_dirs(root, evaluation_only):
    result = {}
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, "general_rubric.md")):
            if not evaluation_only and name != "parent" and not name.startswith("candidate_"):
                continue
            result[name] = path
    if "parent" not in result:
        raise FileNotFoundError(f"Missing parent rubric in {root}")
    if not evaluation_only and len(result) < 2:
        raise ValueError(f"No rollout candidates found in {root}")
    return result


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def truncate_text(text, limit):
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + " ..."


def candidate_metrics(rows):
    by_domain = defaultdict(lambda: [0, 0])
    for row in rows:
        stats = by_domain[str(row["domain"])]
        stats[0] += int(row["is_correct"])
        stats[1] += 1
    domain_accuracy = {domain: correct / total for domain, (correct, total) in sorted(by_domain.items())}
    return {
        "macro_accuracy": sum(domain_accuracy.values()) / len(domain_accuracy),
        "pair_accuracy": sum(int(row["is_correct"]) for row in rows) / len(rows),
        "domain_accuracy": domain_accuracy,
        "correct": sum(int(row["is_correct"]) for row in rows),
        "total": len(rows),
    }


def row_key(row):
    return (row.get("prompt_index"), row.get("pair_index"))


def add_fix_regress_metrics(all_metrics, all_results):
    if "parent" not in all_results:
        return
    parent = {row_key(row): row for row in all_results["parent"]}
    for name, rows in all_results.items():
        if name == "parent":
            all_metrics[name].update(
                {
                    "fix_count": 0,
                    "regress_count": 0,
                    "net_fix": 0,
                    "fix_regress_reward": 0.0,
                    "domain_fix_regress": {},
                }
            )
            continue
        fixed = 0
        regressed = 0
        by_domain = defaultdict(lambda: {"fixed": 0, "regressed": 0, "total": 0})
        for row in rows:
            parent_row = parent.get(row_key(row))
            if parent_row is None:
                continue
            domain_stats = by_domain[str(row.get("domain"))]
            domain_stats["total"] += 1
            parent_correct = bool(parent_row["is_correct"])
            candidate_correct = bool(row["is_correct"])
            if not parent_correct and candidate_correct:
                fixed += 1
                domain_stats["fixed"] += 1
            elif parent_correct and not candidate_correct:
                regressed += 1
                domain_stats["regressed"] += 1
        domain_fix_regress = {}
        domain_rewards = []
        for domain, stats in sorted(by_domain.items()):
            net = stats["fixed"] - stats["regressed"]
            reward = net / stats["total"] if stats["total"] else 0.0
            stats["net"] = net
            stats["reward"] = reward
            domain_fix_regress[domain] = stats
            domain_rewards.append(reward)
        reward = sum(domain_rewards) / len(domain_rewards) if domain_rewards else 0.0
        all_metrics[name].update(
            {
                "fix_count": fixed,
                "regress_count": regressed,
                "net_fix": fixed - regressed,
                "fix_regress_reward": reward,
                "domain_fix_regress": domain_fix_regress,
            }
        )


def compact_feedback_row(row):
    return {
        "domain": row.get("domain"),
        "prompt": truncate_text(row.get("prompt"), 220),
        "prediction": row.get("prediction"),
        "expected": row.get("judge_winner"),
        "reason": truncate_text(row.get("judge_reason"), 240),
    }


def feedback_case_lines(title, rows):
    if not rows:
        return []
    lines = [title]
    for row in rows:
        lines.append(json.dumps(compact_feedback_row(row), ensure_ascii=False))
    return lines


def build_advantage_strategy(metrics, rubrics):
    names = [name for name in metrics if name != "parent"]
    sections = sorted(rubrics["parent"])
    strategy = []
    for section in sections:
        rewards = [
            metrics[name]["fix_regress_reward"] if section == "general"
            else metrics[name].get("domain_fix_regress", {}).get(section, {}).get("reward", 0.0)
            for name in names
        ]
        mean = sum(rewards) / len(rewards)
        std = math.sqrt(sum((reward - mean) ** 2 for reward in rewards) / len(rewards))
        advantages = [(reward - mean) / std if std > 0 else 0.0 for reward in rewards]
        parent_rules = {line.strip() for line in rubrics["parent"][section].splitlines()
                        if line.strip().startswith("- ")}
        for name, reward, advantage in zip(names, rewards, advantages):
            rules = {line.strip() for line in rubrics[name].get(section, "").splitlines()
                     if line.strip().startswith("- ")}
            strategy.append({
                "candidate": name, "section": section, "reward": reward,
                "advantage": advantage,
                "action": "reinforce" if advantage > 0 and reward > 0
                else "avoid" if advantage < 0 else "neutral",
                "added_rules": sorted(rules - parent_rules)[:4],
                "removed_rules": sorted(parent_rules - rules)[:4],
            })
    return strategy


def build_textgrad_feedback(summary):
    candidates = summary["candidates"]
    best = candidates[summary["best_candidate"]]
    parent = candidates["parent"]
    domain_stats = best.get("domain_fix_regress", {})
    lines = [
        "Variable-specific textual gradients for the next rubric update.",
        "general_rubric is the cross-domain variable. Each domain subsection is a separate domain variable.",
        "Update general_rubric only for repeated cross-domain patterns or global priority/tie-break issues.",
        "Update each domain variable only from cases and feedback for that same domain.",
        f"parent_macro={parent.get('macro_accuracy')} best_macro={best.get('macro_accuracy')} accepted={summary.get('accepted')}",
        "",
        "[Advantage-guided evolution strategy]",
        "Each section uses relative advantage A=(reward-group_mean)/group_std.",
        "Use the sign and magnitude of advantage to compare candidate rule changes.",
        "Reinforce positive-advantage changes only when their reward also improves the parent.",
        "Avoid negative-advantage strategies after checking regressions; do not blindly reverse every rule.",
        "Neutral changes provide no directional evidence. Rule effects are associative, not isolated causal estimates.",
    ]
    strategy = sorted(summary.get("advantage_strategy", []),
                      key=lambda item: (item["section"] != "general", -abs(item["advantage"])))
    lines.extend(json.dumps(item, ensure_ascii=False) for item in strategy)
    lines.extend([
        "",
        "[general_rubric gradient]",
        "- Strengthen cross-domain priority rules when fixes/regressions show repeated failure modes.",
        "- Prefer observable scoring criteria over broad quality labels.",
        "- Preserve rules that prevent answer-order, verbosity, polish, and formatting bias.",
    ])
    errors = summary.get("representative_errors", {})
    fixes = summary.get("representative_fixes", {})
    regressions = summary.get("representative_regressions", {})
    for domain in sorted(domain_stats):
        stats = domain_stats[domain]
        lines.extend([
            "",
            f"[{domain}_rubric gradient]",
            f"- fixed={stats.get('fixed', 0)} regressed={stats.get('regressed', 0)} net={stats.get('net', 0)} reward={stats.get('reward', 0.0)}",
        ])
        lines.extend(feedback_case_lines("- Remaining mistakes:", errors.get(domain, [])))
        lines.extend(feedback_case_lines("- Useful fixes to preserve:", fixes.get(domain, [])))
        lines.extend(feedback_case_lines("- Regressions to avoid:", regressions.get(domain, [])))
    return "\n".join(lines)


@lru_cache(maxsize=1)
def load_evaluator(path, prefix, device):
    model = get_llm_for_sequence_regression(
        path, "preference", bf16=True, use_flash_attention_2=True,
        value_head_prefix=prefix, use_laplace=False, normalize_reward=False,
    ).to(device)
    model.eval()
    return model, get_tokenizer(path, model, "left", None, use_fast=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--generation_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=8192)
    parser.add_argument("--value_head_prefix", default="score")
    parser.add_argument("--feedback_errors_per_domain", type=int, default=4)
    parser.add_argument("--tokenizer_chat_template", default=None)
    parser.add_argument("--evaluation_only", action="store_true")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group("nccl")
    rank = dist.get_rank() if local_rank != -1 else 0
    world_size = dist.get_world_size() if local_rank != -1 else 1
    device = torch.device(f"cuda:{local_rank}" if local_rank != -1 else "cuda")

    model, tokenizer = load_evaluator(args.model_path, args.value_head_prefix, device)
    if args.tokenizer_chat_template:
        tokenizer.chat_template = args.tokenizer_chat_template

    records = read_jsonl(args.input)
    for row in records:
        if row.get("router_winner") not in {"assistant_1", "assistant_2"}:
            raise ValueError(f"Missing binary judge router_winner for pair_index={row.get('pair_index')}")
    local_records = records[rank::world_size]
    candidates = rubric_dirs(args.generation_dir, args.evaluation_only)
    all_metrics = {}
    all_results = {}

    for name, path in candidates.items():
        general = read_text(os.path.join(path, "general_rubric.md"))
        domain_map = build_domain_rubric_map(read_text(os.path.join(path, "domain_specific_rubric.md")))
        local_rows = []
        with torch.no_grad():
            starts = range(0, len(local_records), args.batch_size)
            for start in tqdm(starts, desc=f"{name} rank {rank}", disable=rank != 0):
                batch = local_records[start:start + args.batch_size]
                forward, reverse = [], []
                for row in batch:
                    prompt = build_user_content(row, general, domain_map, True)
                    text_1, text_2 = build_pair_texts(
                        tokenizer, prompt, row.get("assistant_1", ""), row.get("assistant_2", "")
                    )
                    forward.append(text_1)
                    reverse.append(text_2)
                inputs_1 = tokenizer(forward, return_tensors="pt", padding=True, truncation=True, max_length=args.max_len)
                inputs_2 = tokenizer(reverse, return_tensors="pt", padding=True, truncation=True, max_length=args.max_len)
                inputs_1 = {key: value.to(device) for key, value in inputs_1.items()}
                inputs_2 = {key: value.to(device) for key, value in inputs_2.items()}
                score_1, _ = model.predict(inputs_1["input_ids"], inputs_1["attention_mask"])
                score_2, _ = model.predict(inputs_2["input_ids"], inputs_2["attention_mask"])
                scores = ((score_1 - score_2) / 2).detach().float().cpu().tolist()
                for row, score in zip(batch, scores):
                    prediction = "assistant_1" if score >= 0 else "assistant_2"
                    judge_winner = row["router_winner"]
                    local_rows.append(
                        {
                            "prompt_index": row.get("prompt_index"),
                            "pair_index": row.get("pair_index"),
                            "domain": row.get("domain"),
                            "prompt": row.get("prompt", ""),
                            "assistant_1": row.get("assistant_1", ""),
                            "assistant_2": row.get("assistant_2", ""),
                            "judge_winner": judge_winner,
                            "judge_reason": row.get("router_reason", ""),
                            "prediction": prediction,
                            "score": float(score),
                            "is_correct": prediction == judge_winner,
                        }
                    )

        if local_rank != -1:
            gathered = [None] * world_size
            dist.all_gather_object(gathered, local_rows)
            rows = [row for part in gathered for row in part] if rank == 0 else []
        else:
            rows = local_rows
        if rank == 0:
            rows.sort(key=lambda row: (row["prompt_index"], row["pair_index"]))
            metrics = candidate_metrics(rows)
            all_results[name] = rows
            all_metrics[name] = metrics
            write_json(os.path.join(path, "dev_metrics.json"), metrics)
            write_jsonl(os.path.join(path, "dev_predictions.jsonl"), rows)

    if rank == 0 and args.evaluation_only:
        summary = {
            "model_path": args.model_path,
            "input": args.input,
            "evaluation": all_metrics,
        }
        write_json(os.path.join(args.generation_dir, "evaluation_summary.json"), summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    if rank == 0 and not args.evaluation_only:
        add_fix_regress_metrics(all_metrics, all_results)
        rollout_names = [name for name in all_metrics if name != "parent"]
        rewards = [all_metrics[name]["fix_regress_reward"] for name in rollout_names]
        mean = sum(rewards) / len(rewards)
        std = math.sqrt(sum((reward - mean) ** 2 for reward in rewards) / len(rewards))
        for name, reward in zip(rollout_names, rewards):
            all_metrics[name]["grpo_advantage"] = (reward - mean) / std if std > 0 else 0.0
        best_name = max(
            rollout_names,
            key=lambda name: (
                all_metrics[name]["fix_regress_reward"],
                all_metrics[name]["macro_accuracy"],
                all_metrics[name]["pair_accuracy"],
            ),
        )
        parent_reward = all_metrics["parent"]["fix_regress_reward"]
        accepted_name = best_name if all_metrics[best_name]["fix_regress_reward"] > parent_reward else "parent"

        worst_name = min(
            rollout_names,
            key=lambda name: (
                all_metrics[name]["fix_regress_reward"],
                all_metrics[name]["macro_accuracy"],
                all_metrics[name]["pair_accuracy"],
            ),
        )
        accepted_dir = os.path.join(args.generation_dir, "accepted_rubric")
        if os.path.exists(accepted_dir):
            shutil.rmtree(accepted_dir)
        shutil.copytree(candidates[accepted_name], accepted_dir)
        errors_by_domain = defaultdict(list)
        for row in all_results[best_name]:
            if not row["is_correct"] and len(errors_by_domain[row["domain"]]) < args.feedback_errors_per_domain:
                errors_by_domain[row["domain"]].append(row)
        parent_by_key = {row_key(row): row for row in all_results["parent"]}
        fixes_by_domain = defaultdict(list)
        regressions_by_domain = defaultdict(list)
        for row in all_results[best_name]:
            parent_row = parent_by_key.get(row_key(row))
            if not parent_row:
                continue
            if not parent_row["is_correct"] and row["is_correct"] and len(fixes_by_domain[row["domain"]]) < args.feedback_errors_per_domain:
                fixes_by_domain[row["domain"]].append(row)
            if parent_row["is_correct"] and not row["is_correct"] and len(regressions_by_domain[row["domain"]]) < args.feedback_errors_per_domain:
                regressions_by_domain[row["domain"]].append(row)
        summary = {
            "model_path": args.model_path,
            "input": args.input,
            "reward": "domain_balanced_fix_minus_regress_against_parent",
            "rollout_mean": mean,
            "rollout_std": std,
            "parent_fix_regress_reward": parent_reward,
            "parent_macro_accuracy": all_metrics["parent"]["macro_accuracy"],
            "best_candidate": best_name,
            "best_candidate_rubric": {
                "general": read_text(os.path.join(candidates[best_name], "general_rubric.md")),
                "domain_specific": read_text(os.path.join(candidates[best_name], "domain_specific_rubric.md")),
            },
            "worst_candidate": worst_name,
            "worst_candidate_rubric": {
                "general": read_text(os.path.join(candidates[worst_name], "general_rubric.md")),
                "domain_specific": read_text(os.path.join(candidates[worst_name], "domain_specific_rubric.md")),
            },
            "accepted": accepted_name,
            "improved": accepted_name != "parent",
            "candidates": all_metrics,
            "representative_errors": dict(errors_by_domain),
            "representative_fixes": dict(fixes_by_domain),
            "representative_regressions": dict(regressions_by_domain),
        }
        rubric_sections = {}
        for name, path in candidates.items():
            rubric_sections[name] = {
                "general": read_text(os.path.join(path, "general_rubric.md")),
                **build_domain_rubric_map(read_text(os.path.join(path, "domain_specific_rubric.md"))),
            }
        summary["advantage_strategy"] = build_advantage_strategy(all_metrics, rubric_sections)
        write_json(os.path.join(args.generation_dir, "generation_summary.json"), summary)
        with open(os.path.join(args.generation_dir, "textgrad_feedback.txt"), "w", encoding="utf-8") as handle:
            handle.write(build_textgrad_feedback(summary))
            handle.write("\n")
        print(json.dumps({key: value for key, value in summary.items() if key != "representative_errors"}, indent=2))

    if local_rank != -1 and not os.environ.get("RUBRIC_PERSISTENT_WORKER"):
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
