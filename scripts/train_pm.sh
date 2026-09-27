#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
OPENRLHF_ROOT=${OPENRLHF_ROOT:-${REPO_ROOT}/third_party/OpenRLHF}
export PYTHONPATH="${REPO_ROOT}:${OPENRLHF_ROOT}:${PYTHONPATH:-}"

: "${PRETRAIN_CKPT:?Set PRETRAIN_CKPT to the initial pairwise PM checkpoint.}"
: "${JUDGEMENTS:?Set JUDGEMENTS to strong-judge JSONL output.}"
: "${RUBRIC_DIR:?Set RUBRIC_DIR to a directory containing both rubric Markdown files.}"

RUN_DIR=${RUN_DIR:-${REPO_ROOT}/artifacts/pm_training}
SAVE_PATH=${SAVE_PATH:-${REPO_ROOT}/artifacts/sure_pm}
TRAIN_DATASET=${TRAIN_DATASET:-${RUN_DIR}/pm_dataset}
INCLUDE_GPUS=${INCLUDE_GPUS:-localhost:0,1,2,3}
MASTER_PORT=${MASTER_PORT:-29501}
SEED=${SEED:-44}
MAX_LEN=${MAX_LEN:-8192}
MICRO_TRAIN_BATCH_SIZE=${MICRO_TRAIN_BATCH_SIZE:-2}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-96}
LEARNING_RATE=${LEARNING_RATE:-2e-6}
WARMUP_RATIO=${WARMUP_RATIO:-0.10}
ZERO_STAGE=${ZERO_STAGE:-3}
STRENGTH=${STRENGTH:-1.5}

mkdir -p "${RUN_DIR}" "${SAVE_PATH}"

python "${REPO_ROOT}/src/eval_pm_router/audit_judgements.py" \
  --input_path "${JUDGEMENTS}" \
  --output_dir "${RUN_DIR}/judgement_audit" \
  --filter_mode all

python "${REPO_ROOT}/src/build_domain_rubric_pm_dataset.py" \
  --teacher_pairs_jsonl "${RUN_DIR}/judgement_audit/train_all.jsonl" \
  --output_dataset "${TRAIN_DATASET}" \
  --rubric_root "$(dirname "${RUBRIC_DIR}")" \
  --rubric_dir "${RUBRIC_DIR}" \
  --label_source router_winner \
  --strict_label_source \
  --strength "${STRENGTH}" \
  --validation_ratio 0 \
  --seed "${SEED}" \
  --add_reverse_pairs

CHAT_TEMPLATE="{% for message in messages %}<|im_start|>{{ message['role'] }}
{{ message['content'] }}<|im_end|>
{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"

python -m deepspeed.launcher.runner \
  --master_port "${MASTER_PORT}" \
  --include="${INCLUDE_GPUS}" \
  --module openrlhf.cli.train_pm \
  --pretrain "${PRETRAIN_CKPT}" \
  --save_path "${SAVE_PATH}" \
  --dataset "${TRAIN_DATASET}" \
  --train_split train \
  --eval_split train \
  --context_key context_messages \
  --label_key label \
  --strength_key strength \
  --apply_chat_template \
  --tokenizer_chat_template "${CHAT_TEMPLATE}" \
  --value_head_prefix score \
  --seed "${SEED}" \
  --max_len "${MAX_LEN}" \
  --max_epochs 1 \
  --loss scaled_bt \
  --learning_rate "${LEARNING_RATE}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --micro_train_batch_size "${MICRO_TRAIN_BATCH_SIZE}" \
  --train_batch_size "${TRAIN_BATCH_SIZE}" \
  --save_steps -1 \
  --eval_steps 999999 \
  --zero_stage "${ZERO_STAGE}" \
  --bf16 \
  --flash_attn \
  --gradient_checkpointing

