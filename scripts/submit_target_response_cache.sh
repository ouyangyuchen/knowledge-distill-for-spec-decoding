#!/bin/bash
# Submit one Run:AI job to cache target-generated responses with vLLM.
#
# Run from the repo root on a machine with runai configured. Defaults target the
# Qwen2.5 UltraChat-50k response cache on one 40GB A100. Override env vars, e.g.
#   DATA=ultrachat_10k REQUEST_BATCH_SIZE=512 ./scripts/submit_target_response_cache.sh
#   MODEL=qwen3 TARGET_ID=Qwen/Qwen3-14B DATA=ultrachat_50k ./scripts/submit_target_response_cache.sh

set -euo pipefail

MODEL="${MODEL:-qwen25}"
DATA="${DATA:-ultrachat_50k}"
TARGET_ID="${TARGET_ID:-}"
SEED="${SEED:-42}"
SPLITS="${SPLITS:-[train,val]}"
PREPARE_DATA="${PREPARE_DATA:-true}"
LIMIT="${LIMIT:-}"

BACKEND="${BACKEND:-vllm}"
REQUEST_BATCH_SIZE="${REQUEST_BATCH_SIZE:-1024}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
MODE="${MODE:-greedy}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
SWAP_SPACE="${SWAP_SPACE:-0}"
ENFORCE_EAGER="${ENFORCE_EAGER:-false}"

REPO_BRANCH="${REPO_BRANCH:-codex/vllm-generate}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-cache_${MODEL}_${DATA}_seed${SEED}}"
WANDB_MODE="${WANDB_MODE:-disabled}"

TARGET_OVERRIDE=""
if [[ -n "${TARGET_ID}" ]]; then
  TARGET_OVERRIDE="model.target=${TARGET_ID}"
fi

LIMIT_OVERRIDE=""
if [[ -n "${LIMIT}" ]]; then
  LIMIT_OVERRIDE="+data.limit=${LIMIT}"
fi

read -r -d '' run_command <<EOF || true
set -euo pipefail

echo ">>> Target response cache job: ${EXPERIMENT_NAME}"
echo ">>> model config: ${MODEL}"
echo ">>> target override: ${TARGET_ID:-<config default>}"
echo ">>> data: ${DATA}"
echo ">>> splits: ${SPLITS}"
echo ">>> prepare data first: ${PREPARE_DATA}"
echo ">>> backend: ${BACKEND}"
echo ">>> request batch size: ${REQUEST_BATCH_SIZE}"
echo ">>> max new tokens: ${MAX_NEW_TOKENS}"
echo ">>> mode/temp/top_p: ${MODE}/${TEMPERATURE}/${TOP_P}"
echo ">>> max_model_len: ${MAX_MODEL_LEN}"
echo ">>> gpu memory utilization: ${GPU_MEMORY_UTILIZATION}"

if [[ "${PREPARE_DATA}" == "true" || "${PREPARE_DATA}" == "1" ]]; then
  echo ">>> Preparing processed data for ${DATA}"
  python scripts/prepare_data.py \
model=${MODEL} data=${DATA} ${TARGET_OVERRIDE} ${LIMIT_OVERRIDE} seed=${SEED}
else
  echo ">>> PREPARE_DATA=${PREPARE_DATA}; skipping prepare_data.py"
fi

echo ">>> Generating target responses"
python scripts/generate_target_responses.py \
model=${MODEL} data=${DATA} ${TARGET_OVERRIDE} ${LIMIT_OVERRIDE} seed=${SEED} \
"data.target_generation.splits=${SPLITS}" \
data.target_generation.backend=${BACKEND} \
data.target_generation.request_batch_size=${REQUEST_BATCH_SIZE} \
data.target_generation.max_new_tokens=${MAX_NEW_TOKENS} \
data.target_generation.mode=${MODE} \
data.target_generation.temperature=${TEMPERATURE} \
data.target_generation.top_p=${TOP_P} \
data.target_generation.max_model_len=${MAX_MODEL_LEN} \
data.target_generation.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
data.target_generation.tensor_parallel_size=${TENSOR_PARALLEL_SIZE} \
data.target_generation.swap_space=${SWAP_SPACE} \
data.target_generation.enforce_eager=${ENFORCE_EAGER}

echo ">>> Target response cache finished"
EOF

echo ">>> Submitting target response cache job: ${EXPERIMENT_NAME}"
REPO_BRANCH="${REPO_BRANCH}" \
RUN_NAME="${EXPERIMENT_NAME}" \
WANDB_MODE="${WANDB_MODE}" \
RUN_COMMAND="${run_command}" \
./rcp_support/submit_train.sh "tcache-${MODEL}"
