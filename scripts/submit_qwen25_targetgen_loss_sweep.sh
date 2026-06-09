#!/bin/bash
# Submit one Run:AI job that runs a Qwen2.5 target-generated loss sweep.
#
# The heavy training/eval logic lives in run_qwen25_targetgen_loss_sweep.sh so
# the RunAI RUN_COMMAND stays short enough for environment-variable limits.
# Override env vars as needed, e.g.
#   LOSSES="fkl rkl jsd ce" ./scripts/submit_qwen25_targetgen_loss_sweep.sh
#   DRY_RUN=true ./scripts/submit_qwen25_targetgen_loss_sweep.sh

set -euo pipefail

TARGET_ID="${TARGET_ID:-Qwen/Qwen2.5-3B-Instruct}"
DRAFT_ID="${DRAFT_ID:-Qwen/Qwen2.5-0.5B-Instruct}"
DATA="${DATA:-ultrachat_50k_target_gen}"
SOURCE_PROCESSED_DIR="${SOURCE_PROCESSED_DIR:-/scratch/cs552-data/processed/ultrachat_50k}"
TARGET_CACHE_DIR="${TARGET_CACHE_DIR:-/scratch/cs552-data/target_generated/ultrachat_50k_qwen25_3b}"

LOSSES="${LOSSES:-fkl rkl jsd ce}"
STEPS="${STEPS:-8000}" # Set max_steps=0 to use num_train_epochs.
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
LR="${LR:-2e-5}"
ALPHA="${ALPHA:-1.0}" # Only used for non-CE losses.
TEMP="${TEMP:-1.0}"
SEED="${SEED:-42}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
KD_CHUNK_SIZE="${KD_CHUNK_SIZE:-}"
COMPILE_TARGET="${COMPILE_TARGET:-false}"
EVAL_REPORTING_STEPS="${EVAL_REPORTING_STEPS:-2000}"
SAVE_STEPS="${SAVE_STEPS:-2000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-4}"
LOAD_BEST_MODEL_AT_END="${LOAD_BEST_MODEL_AT_END:-true}"
SAVE_BEST_MODEL="${SAVE_BEST_MODEL:-true}"
VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
TRAIN_REPORT_TO_WANDB="${TRAIN_REPORT_TO_WANDB:-true}"

RUN_EVAL="${RUN_EVAL:-true}"
EVAL_PRETRAINED_BASELINE="${EVAL_PRETRAINED_BASELINE:-true}"
EVAL_PROMPTS_JSONL="${EVAL_PROMPTS_JSONL:-${SOURCE_PROCESSED_DIR}/eval.jsonl}"
EVAL_PROMPTS_LIMIT="${EVAL_PROMPTS_LIMIT:-50}"
EVAL_GAMMA="${EVAL_GAMMA:-4}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-256}"
EVAL_WARMUP="${EVAL_WARMUP:-1}"
EVAL_REPEATS="${EVAL_REPEATS:-1}"
EVAL_BACKEND="${EVAL_BACKEND:-manual}"
EVAL_REPORT_TO_WANDB="${EVAL_REPORT_TO_WANDB:-true}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen25_3btarget_0p5b_targetgen50k_seed${SEED}}"
WANDB_GROUP="${WANDB_GROUP:-${EXPERIMENT_NAME}}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-qwen25_3btarget_0p5b}"

REPO_BRANCH="${REPO_BRANCH:-codex/vllm-eval}"
WANDB_MODE="${WANDB_MODE:-online}"
DRY_RUN="${DRY_RUN:-false}"

quote() {
  printf "%q" "$1"
}

run_command="TARGET_ID=$(quote "${TARGET_ID}")"
run_command+=" DRAFT_ID=$(quote "${DRAFT_ID}")"
run_command+=" DATA=$(quote "${DATA}")"
run_command+=" SOURCE_PROCESSED_DIR=$(quote "${SOURCE_PROCESSED_DIR}")"
run_command+=" TARGET_CACHE_DIR=$(quote "${TARGET_CACHE_DIR}")"
run_command+=" LOSSES=$(quote "${LOSSES}")"
run_command+=" STEPS=$(quote "${STEPS}")"
run_command+=" EPOCHS=$(quote "${EPOCHS}")"
run_command+=" BATCH_SIZE=$(quote "${BATCH_SIZE}")"
run_command+=" GRAD_ACCUM_STEPS=$(quote "${GRAD_ACCUM_STEPS}")"
run_command+=" LR=$(quote "${LR}")"
run_command+=" ALPHA=$(quote "${ALPHA}")"
run_command+=" TEMP=$(quote "${TEMP}")"
run_command+=" SEED=$(quote "${SEED}")"
run_command+=" MAX_SEQ_LEN=$(quote "${MAX_SEQ_LEN}")"
run_command+=" KD_CHUNK_SIZE=$(quote "${KD_CHUNK_SIZE}")"
run_command+=" COMPILE_TARGET=$(quote "${COMPILE_TARGET}")"
run_command+=" EVAL_REPORTING_STEPS=$(quote "${EVAL_REPORTING_STEPS}")"
run_command+=" VLLM_WORKER_MULTIPROC_METHOD=$(quote "${VLLM_WORKER_MULTIPROC_METHOD}")"
run_command+=" PYTORCH_CUDA_ALLOC_CONF=$(quote "${PYTORCH_CUDA_ALLOC_CONF}")"
run_command+=" TRAIN_REPORT_TO_WANDB=$(quote "${TRAIN_REPORT_TO_WANDB}")"
run_command+=" RUN_EVAL=$(quote "${RUN_EVAL}")"
run_command+=" EVAL_PRETRAINED_BASELINE=$(quote "${EVAL_PRETRAINED_BASELINE}")"
run_command+=" EVAL_PROMPTS_JSONL=$(quote "${EVAL_PROMPTS_JSONL}")"
run_command+=" EVAL_PROMPTS_LIMIT=$(quote "${EVAL_PROMPTS_LIMIT}")"
run_command+=" EVAL_GAMMA=$(quote "${EVAL_GAMMA}")"
run_command+=" EVAL_MAX_NEW_TOKENS=$(quote "${EVAL_MAX_NEW_TOKENS}")"
run_command+=" EVAL_WARMUP=$(quote "${EVAL_WARMUP}")"
run_command+=" EVAL_REPEATS=$(quote "${EVAL_REPEATS}")"
run_command+=" EVAL_BACKEND=$(quote "${EVAL_BACKEND}")"
run_command+=" EVAL_REPORT_TO_WANDB=$(quote "${EVAL_REPORT_TO_WANDB}")"
run_command+=" EXPERIMENT_NAME=$(quote "${EXPERIMENT_NAME}")"
run_command+=" WANDB_GROUP=$(quote "${WANDB_GROUP}")"
run_command+=" RUN_NAME_PREFIX=$(quote "${RUN_NAME_PREFIX}")"
run_command+=" bash scripts/run_qwen25_targetgen_loss_sweep.sh"

echo ">>> RUN_COMMAND chars: ${#run_command} (kept short for RunAI env limit)"
echo ">>> Submitting Qwen2.5 target-generated loss sweep: ${EXPERIMENT_NAME}"
echo ">>> target cache: ${TARGET_CACHE_DIR}"

if [[ "${DRY_RUN}" == "true" || "${DRY_RUN}" == "1" ]]; then
  echo ">>> DRY_RUN=${DRY_RUN}; not submitting. RUN_COMMAND:"
  printf '%s\n' "${run_command}"
  exit 0
fi

REPO_BRANCH="${REPO_BRANCH}" \
RUN_NAME="${EXPERIMENT_NAME}" \
RUN_COMMAND="${run_command}" \
WANDB_MODE="${WANDB_MODE}" \
./rcp_support/submit_train.sh "qwen25-tgen-loss-sweep"
