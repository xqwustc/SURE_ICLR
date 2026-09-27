#!/usr/bin/env python3
import argparse
import json
import os


def read_text(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read().strip()


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--generations", required=True, type=int)
    args = parser.parse_args()

    rows = []
    initial_dir = os.path.join(args.run_root, "trajectory", "round_00")
    if not os.path.isdir(initial_dir):
        initial_dir = os.path.join(args.run_root, "generation_01", "parent")
    rows.append(
        {
            "round": 0,
            "role": "initial",
            "accepted": "initial",
            "improved": None,
            "macro_accuracy": None,
            "pair_accuracy": None,
            "general_rubric": read_text(os.path.join(initial_dir, "general_rubric.md")),
            "domain_specific_rubric": read_text(os.path.join(initial_dir, "domain_specific_rubric.md")),
        }
    )

    for generation in range(1, args.generations + 1):
        round_dir = os.path.join(args.run_root, "trajectory", f"round_{generation:02d}")
        generation_dir = os.path.join(args.run_root, f"generation_{generation:02d}")
        rubric_dir = round_dir if os.path.isdir(round_dir) else os.path.join(generation_dir, "accepted_rubric")
        summary_path = os.path.join(round_dir, "generation_summary.json")
        if not os.path.isfile(summary_path):
            summary_path = os.path.join(generation_dir, "generation_summary.json")
        summary = read_json(summary_path)
        accepted = summary["accepted"]
        metrics = summary["candidates"][accepted]
        rows.append(
            {
                "round": generation,
                "role": "accepted",
                "accepted": accepted,
                "best_candidate": summary["best_candidate"],
                "improved": summary["improved"],
                "macro_accuracy": metrics["macro_accuracy"],
                "pair_accuracy": metrics["pair_accuracy"],
                "domain_accuracy": metrics["domain_accuracy"],
                "grpo_advantage": metrics.get("grpo_advantage"),
                "general_rubric": read_text(os.path.join(rubric_dir, "general_rubric.md")),
                "domain_specific_rubric": read_text(os.path.join(rubric_dir, "domain_specific_rubric.md")),
            }
        )

    output = os.path.join(args.run_root, "rubric_trajectory.jsonl")
    with open(output, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved rubric trajectory with {len(rows)} snapshots to {output}")


if __name__ == "__main__":
    main()
