#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
OPENRLHF_ROOT=${OPENRLHF_ROOT:-${REPO_ROOT}/third_party/OpenRLHF}
export PYTHONPATH="${REPO_ROOT}:${OPENRLHF_ROOT}:${PYTHONPATH:-}"

: "${PM_CHECKPOINT:?Set PM_CHECKPOINT to the trained pairwise PM.}"
: "${RUBRIC_PATH:?Set RUBRIC_PATH to general_rubric.md.}"

PM_GPUS=${PM_GPUS:-0,1}
WORLD_SIZE=${WORLD_SIZE:-2}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29540}
PORT=${PORT:-5000}
MAX_LEN=${MAX_LEN:-8192}
PM_BATCH_SIZE=${PM_BATCH_SIZE:-8}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-4}

IFS=',' read -r -a PM_GPU_LIST <<< "${PM_GPUS}"
if (( ${#PM_GPU_LIST[@]} != WORLD_SIZE )); then
  echo "WORLD_SIZE must match the number of entries in PM_GPUS." >&2
  exit 2
fi

MASTER_ADDR="${MASTER_ADDR}" MASTER_PORT="${MASTER_PORT}" \
CUDA_VISIBLE_DEVICES="${PM_GPUS}" python -m openrlhf.cli.serve_pm \
  --reward_pretrain "${PM_CHECKPOINT}" \
  --model_type preference \
  --value_head_prefix score \
  --max_len "${MAX_LEN}" \
  --batch_size "${PM_BATCH_SIZE}" \
  --n_samples_per_prompt "${N_SAMPLES_PER_PROMPT}" \
  --world_size "${WORLD_SIZE}" \
  --rubric_path "${RUBRIC_PATH}" \
  --bf16 \
  --flash_attn \
  --port "${PORT}"

