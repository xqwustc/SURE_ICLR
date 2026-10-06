# SURE: Evolving Pairwise Reward Models with Rubrics from Uncertain Comparisons

![ICLR 2027](https://img.shields.io/badge/ICLR-2027%20Submission-blue)
![Anonymous](https://img.shields.io/badge/review-anonymous-lightgrey)

This repository contains the anonymous implementation accompanying an ICLR 2027 submission.

SURE closes the loop between pairwise reward-model uncertainty and reusable preference knowledge:

1. calibrate low-margin and representation-space OOD uncertainty against an in-distribution anchor set;
2. route the most uncertain response pairs to a stronger judge;
3. train a pairwise preference model from the judge comparisons and their reversed counterparts;
4. evolve a two-level rubric using errors made by the current preference model; and
5. optionally use the resulting pairwise model as a group-relative reward for policy optimization.

## Repository Layout

```text
SURE_ICLR/
├── configs/example_rubric/       # Minimal two-level rubric format
├── openrlhf_patch/               # OpenRLHF integration for pairwise PM and GRPO
├── scripts/                      # Portable stage launchers
└── src/
    ├── build_domain_rubric_pm_dataset.py
    └── eval_pm_router/
        ├── score_and_select_pairs.py
        ├── judge_selected_pairs.py
        ├── audit_judgements.py
        ├── eval_rmbench_with_domain_rubric.py
        └── experiments/rubric_grpo_textgrad_evolution/
```

## Installation

The code expects a standard OpenRLHF checkout and the packages listed in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

git clone https://github.com/OpenRLHF/OpenRLHF.git third_party/OpenRLHF
cp -r openrlhf_patch/openrlhf/* third_party/OpenRLHF/openrlhf/
export PYTHONPATH="$(pwd):$(pwd)/third_party/OpenRLHF:${PYTHONPATH:-}"
```

## Input Format

Candidate pairs are stored as JSON Lines. The method uses no benchmark label during selection or strong-judge supervision.

```json
{
  "prompt_index": 0,
  "pair_index": 0,
  "domain": "math",
  "prompt": "...",
  "assistant_1": "...",
  "assistant_2": "..."
}
```

The anchor dataset is a Hugging Face dataset saved with `save_to_disk` and contains a `context_messages` column with `user`, `assistant_1`, and `assistant_2` roles.

## 1. SURE Selection

Both uncertainty components are calibrated against the same in-distribution anchor distribution:

```text
OOD risk    = empirical CDF of anchor Mahalanobis distances
margin risk = 1 - empirical CDF of anchor absolute symmetric margins
SURE score  = 0.5 * OOD risk + 0.5 * margin risk
```

The forward/reverse symmetric PM margin is `(score_forward - score_reverse) / 2`.

```bash
torchrun --nproc_per_node=4 \
  src/eval_pm_router/score_and_select_pairs.py \
  --model_path artifacts/base_pm \
  --anchor_dataset data/id_anchor \
  --input_pairs data/candidate_pairs.jsonl \
  --output_dir artifacts/selection \
  --top_fraction 0.10 \
  --batch_size 4
```

Selection is performed over prompt groups. By default, a group is ranked by its maximum pair-level SURE score.

## 2. Strong-Judge Annotation

Any OpenAI-compatible judge endpoint can be used. Credentials are read only from environment variables.

```bash
export JUDGE_MODEL=<judge-model>
export JUDGE_API_KEY=<api-key>
export JUDGE_BASE_URL=<openai-compatible-endpoint>

python src/eval_pm_router/judge_selected_pairs.py \
  --input_path artifacts/selection/selected_pairs.jsonl \
  --output_path artifacts/judgements.jsonl \
  --workers 8 \
  --resume
```

The output stores `router_winner` as `assistant_1` or `assistant_2`, together with the judge reason and confidence. Parse failures are retained for auditing and excluded from strict training/evolution inputs.

## 3. Initial Rubric and Pairwise PM Training

Generate the initial two-level rubric from valid judge comparisons:

```bash
python src/eval_pm_router/audit_judgements.py \
  --input_path artifacts/judgements.jsonl \
  --output_dir artifacts/audit \
  --filter_mode all

CUDA_VISIBLE_DEVICES=0 python \
  src/eval_pm_router/experiments/rubric_grpo_textgrad_evolution/generate_initial_rubric.py \
  --model_path models/rubric_generator \
  --input artifacts/audit/train_all.jsonl \
  --output_dir artifacts/initial_rubric \
  --examples_per_domain 50 \
  --batch_examples_per_domain 10 \
  --seed 44
```

Build forward and reversed training comparisons, then train for one epoch:

```bash
PRETRAIN_CKPT=artifacts/base_pm \
JUDGEMENTS=artifacts/judgements.jsonl \
RUBRIC_DIR=artifacts/initial_rubric \
bash scripts/train_pm.sh
```

The training label always comes from `router_winner`; benchmark labels are not used as fallback supervision.

## 4. Rubric Evolution

Rubric candidates are generated from current-model errors and accepted according to pairwise agreement with the strong judge.

```bash
PM_MODEL=artifacts/sure_pm \
EVOLVER_MODEL=models/rubric_generator \
SELECTED_PATH=artifacts/audit/train_all.jsonl \
INITIAL_RUBRIC_DIR=artifacts/initial_rubric \
EVOLVER_GPUS=0 \
EVAL_GPUS=1,2,3 \
bash scripts/evolve_rubric.sh
```

Each accepted rubric is saved under `artifacts/rubric_evolution/trajectory/round_XX`. The trajectory is directly evaluable without regenerating candidates.

## 5. Evaluation

```bash
torchrun --nproc_per_node=4 \
  src/eval_pm_router/eval_rmbench_with_domain_rubric.py \
  --model_path artifacts/sure_pm \
  --rubric_dir artifacts/rubric_evolution/trajectory/round_40 \
  --save_path artifacts/rmbench_eval \
  --batch_size 1 \
  --max_len 8192
```

## 6. Optional Group-Relative Policy Optimization

`openrlhf_patch` contains the exact integration used to:

- send structured groups of prompt-response records to a frozen pairwise PM;
- convert pairwise preferences into centered or standardized group-relative advantages;
- apply PPO-style clipped actor updates without a critic; and
- synchronize the actor with vLLM rollout engines.

After overlaying the patch onto OpenRLHF, start the pairwise PM and policy training with the portable launchers:

```bash
PM_CHECKPOINT=artifacts/sure_pm \
RUBRIC_PATH=artifacts/rubric_evolution/trajectory/round_40/general_rubric.md \
bash scripts/serve_pairwise_pm.sh

POLICY_PRETRAIN=models/instruct_model \
PROMPT_DATASET=data/policy_prompts \
REMOTE_RM_URL=http://127.0.0.1:5000/get_reward \
bash scripts/train_policy_vllm.sh
```

## Reproducibility Notes

- Default random seed: `44`.
- Default acquisition weight: equal weighting of calibrated OOD and margin risks.
- PM training includes forward and reversed pairs.
- Rubric evolution uses only valid strong-judge comparisons.
- RM-Bench trajectory reporting uses round 40; RewardBench trajectory reporting uses round 44 in the paper experiments.

