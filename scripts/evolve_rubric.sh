#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
OPENRLHF_ROOT=${OPENRLHF_ROOT:-${REPO_ROOT}/third_party/OpenRLHF}
EVOLUTION_DIR="${REPO_ROOT}/src/eval_pm_router/experiments/rubric_grpo_textgrad_evolution"
export PYTHONPATH="${REPO_ROOT}:${OPENRLHF_ROOT}:${EVOLUTION_DIR}:${PYTHONPATH:-}"

: "${PM_MODEL:?Set PM_MODEL to the fixed pairwise PM checkpoint.}"
: "${EVOLVER_MODEL:?Set EVOLVER_MODEL to the local rubric-generator checkpoint.}"
: "${SELECTED_PATH:?Set SELECTED_PATH to valid strong-judge comparisons.}"
: "${INITIAL_RUBRIC_DIR:?Set INITIAL_RUBRIC_DIR to the initial two-level rubric.}"

RUN_ROOT=${RUN_ROOT:-${REPO_ROOT}/artifacts/rubric_evolution}
FINAL_RUBRIC_DIR=${FINAL_RUBRIC_DIR:-${RUN_ROOT}/final_rubric}
EVOLVER_GPUS=${EVOLVER_GPUS:-0}
EVAL_GPUS=${EVAL_GPUS:-1,2,3}
NPROC_PER_NODE=${NPROC_PER_NODE:-3}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-29520}
GENERATIONS=${GENERATIONS:-40}
ROLLOUTS=${ROLLOUTS:-8}
PER_DOMAIN=${PER_DOMAIN:-50}
SEED=${SEED:-44}
TEMPERATURE=${TEMPERATURE:-1.2}
TOP_P=${TOP_P:-0.98}
MAX_SIMILARITY=${MAX_SIMILARITY:-0.8}
MAX_INPUT_TOKENS=${MAX_INPUT_TOKENS:-30000}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-4096}
MAX_CASE_CHARS=${MAX_CASE_CHARS:-12000}
MAX_FEEDBACK_CHARS=${MAX_FEEDBACK_CHARS:-8000}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1}
MAX_LEN=${MAX_LEN:-12288}

mkdir -p "${RUN_ROOT}/trajectory/round_00"
cp "${INITIAL_RUBRIC_DIR}/general_rubric.md" "${RUN_ROOT}/trajectory/round_00/general_rubric.md"
cp "${INITIAL_RUBRIC_DIR}/domain_specific_rubric.md" "${RUN_ROOT}/trajectory/round_00/domain_specific_rubric.md"

if [[ ! -d "${RUN_ROOT}/data" ]]; then
  python "${EVOLUTION_DIR}/prepare_subset.py" \
    --input "${SELECTED_PATH}" \
    --output_dir "${RUN_ROOT}/data" \
    --per_domain "${PER_DOMAIN}" \
    --seed "${SEED}"
fi

QWEN_PM_CHAT_TEMPLATE="{% for message in messages %}<|im_start|>{{ message['role'] }}
{{ message['content'] }}<|im_end|>
{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"

export PM_MODEL EVOLVER_MODEL EVOLVER_GPUS EVAL_GPUS NPROC_PER_NODE MASTER_PORT_BASE
export RUN_ROOT FINAL_RUBRIC_DIR GENERATIONS ROLLOUTS PER_DOMAIN SEED TEMPERATURE TOP_P
export MAX_SIMILARITY MAX_INPUT_TOKENS MAX_NEW_TOKENS MAX_CASE_CHARS MAX_FEEDBACK_CHARS
export EVAL_BATCH_SIZE MAX_LEN QWEN_PM_CHAT_TEMPLATE

python "${EVOLUTION_DIR}/persistent_run.py"

