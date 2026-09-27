#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
OPENRLHF_ROOT=${OPENRLHF_ROOT:-${REPO_ROOT}/third_party/OpenRLHF}
export PYTHONPATH="${REPO_ROOT}:${OPENRLHF_ROOT}:${PYTHONPATH:-}"

: "${POLICY_PRETRAIN:?Set POLICY_PRETRAIN to the initial policy checkpoint.}"
: "${PROMPT_DATASET:?Set PROMPT_DATASET to a non-benchmark prompt dataset.}"
: "${REMOTE_RM_URL:?Set REMOTE_RM_URL to the pairwise PM /get_reward endpoint.}"

TRAIN_GPUS=${TRAIN_GPUS:-2,3,4,5,6,7}
ACTOR_GPUS=${ACTOR_GPUS:-4}
VLLM_ENGINES=${VLLM_ENGINES:-2}
VLLM_TP_SIZE=${VLLM_TP_SIZE:-1}
RAY_PORT=${RAY_PORT:-29560}
RAY_CPUS=${RAY_CPUS:-48}
SAVE_PATH=${SAVE_PATH:-${REPO_ROOT}/artifacts/policy}
CKPT_PATH=${CKPT_PATH:-${SAVE_PATH}/training_state}
MAX_SAMPLES=${MAX_SAMPLES:-38912}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-4}
MICRO_ROLLOUT_BATCH_SIZE=${MICRO_ROLLOUT_BATCH_SIZE:-16}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-128}
MICRO_TRAIN_BATCH_SIZE=${MICRO_TRAIN_BATCH_SIZE:-1}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
PROMPT_MAX_LEN=${PROMPT_MAX_LEN:-512}
GENERATE_MAX_LEN=${GENERATE_MAX_LEN:-512}
SEED=${SEED:-44}

IFS=',' read -r -a TRAIN_GPU_LIST <<< "${TRAIN_GPUS}"
if (( ${#TRAIN_GPU_LIST[@]} != 6 )); then
  echo "TRAIN_GPUS must contain exactly six GPU indices." >&2
  exit 2
fi
if (( ACTOR_GPUS + VLLM_ENGINES * VLLM_TP_SIZE != 6 )); then
  echo "This launcher reserves six training GPUs: actor + vLLM must equal 6." >&2
  exit 2
fi
if (( ROLLOUT_BATCH_SIZE % ACTOR_GPUS != 0 )); then
  echo "ROLLOUT_BATCH_SIZE must be divisible by ACTOR_GPUS." >&2
  exit 2
fi
if (( TRAIN_BATCH_SIZE % (MICRO_TRAIN_BATCH_SIZE * ACTOR_GPUS) != 0 )); then
  echo "TRAIN_BATCH_SIZE must be divisible by MICRO_TRAIN_BATCH_SIZE * ACTOR_GPUS." >&2
  exit 2
fi
if (( MICRO_ROLLOUT_BATCH_SIZE % N_SAMPLES_PER_PROMPT != 0 )); then
  echo "MICRO_ROLLOUT_BATCH_SIZE must be divisible by N_SAMPLES_PER_PROMPT." >&2
  exit 2
fi

curl --silent --show-error --fail --max-time 10 \
  "${REMOTE_RM_URL%/get_reward}/health" >/dev/null

mkdir -p "${SAVE_PATH}" "${CKPT_PATH}" "${REPO_ROOT}/artifacts/ray"
RAY_LOG="${REPO_ROOT}/artifacts/ray/ray.log"
RAY_PID=""

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM
  if [[ -n "${RAY_PID}" ]]; then
    kill -TERM -- "-${RAY_PID}" 2>/dev/null || true
    wait "${RAY_PID}" 2>/dev/null || true
  fi
  exit "${exit_code}"
}
trap cleanup EXIT INT TERM

setsid env CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" \
  ray start --head --block \
  --node-ip-address=127.0.0.1 \
  --port="${RAY_PORT}" \
  --num-cpus="${RAY_CPUS}" \
  --num-gpus=6 \
  --temp-dir="${REPO_ROOT}/artifacts/ray" \
  --disable-usage-stats \
  --include-dashboard=false \
  >"${RAY_LOG}" 2>&1 &
RAY_PID=$!
export RAY_ADDRESS="127.0.0.1:${RAY_PORT}"

for _ in $(seq 1 120); do
  if ray status --address="${RAY_ADDRESS}" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "${RAY_PID}" 2>/dev/null; then
    tail -n 200 "${RAY_LOG}" >&2 || true
    exit 1
  fi
  sleep 2
done
ray status --address="${RAY_ADDRESS}" >/dev/null 2>&1

CHAT_TEMPLATE="{% for message in messages %}<|im_start|>{{ message['role'] }}
{{ message['content'] }}<|im_end|>
{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"

python -u -m openrlhf.cli.train_ppo_ray \
  --actor_num_nodes 1 --actor_num_gpus_per_node "${ACTOR_GPUS}" \
  --ref_num_nodes 1 --ref_num_gpus_per_node "${ACTOR_GPUS}" --colocate_actor_ref \
  --vllm_num_engines "${VLLM_ENGINES}" --vllm_tensor_parallel_size "${VLLM_TP_SIZE}" \
  --vllm_sync_backend gloo \
  --pretrain "${POLICY_PRETRAIN}" \
  --remote_rm_url "${REMOTE_RM_URL}" --reward_model_type preference \
  --save_path "${SAVE_PATH}" --ckpt_path "${CKPT_PATH}" \
  --save_steps 20 --max_ckpt_num -1 --logging_steps 1 --eval_steps -1 \
  --micro_train_batch_size "${MICRO_TRAIN_BATCH_SIZE}" --train_batch_size "${TRAIN_BATCH_SIZE}" \
  --micro_rollout_batch_size "${MICRO_ROLLOUT_BATCH_SIZE}" --rollout_batch_size "${ROLLOUT_BATCH_SIZE}" \
  --n_samples_per_prompt "${N_SAMPLES_PER_PROMPT}" \
  --policy_gradient_style grpo --disable_critic --lambd 1 --gamma 1 --max_epochs 1 \
  --preference_advantage_mode standardized \
  --prompt_max_len "${PROMPT_MAX_LEN}" --generate_max_len "${GENERATE_MAX_LEN}" \
  --zero_stage 2 --bf16 --actor_learning_rate 5e-7 --init_kl_coef 0.01 \
  --temperature 1.0 --top_p 1.0 \
  --prompt_data "${PROMPT_DATASET}" --input_key context_messages \
  --apply_chat_template --tokenizer_chat_template "${CHAT_TEMPLATE}" \
  --max_samples "${MAX_SAMPLES}" --seed "${SEED}" \
  --flash_attn --gradient_checkpointing --perf

