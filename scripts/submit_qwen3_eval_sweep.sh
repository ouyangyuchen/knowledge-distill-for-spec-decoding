#!/bin/bash
# Submit one Run:AI job that evaluates Qwen3 pretrained + trained checkpoints.
# Run this from the repo root on the laptop/cluster login machine where runai is
# configured. It mirrors submit_qwen3_loss_sweep.sh but does not train.
#
# Common overrides:
#   LOSSES="fkl rkl jsd ce" ./scripts/submit_qwen3_eval_sweep.sh
#   DRAFT_SIZE=1.7b LOSSES="ce" ./scripts/submit_qwen3_eval_sweep.sh
#   FORCE_RERUN=true EVAL_PROMPTS_LIMIT=200 ./scripts/submit_qwen3_eval_sweep.sh

set -euo pipefail

DRAFT_SIZE="${DRAFT_SIZE:-0.6b}"  # 0.6b or 1.7b
DATA="${DATA:-ultrachat_50k}"
SEED="${SEED:-42}"
TARGET_ID="${TARGET_ID:-}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

EVAL_PRETRAINED_BASELINE="${EVAL_PRETRAINED_BASELINE:-true}"
EVAL_PROMPTS_JSONL="${EVAL_PROMPTS_JSONL:-/scratch/cs552-data/processed/${DATA}/eval.jsonl}"
EVAL_PROMPTS_LIMIT="${EVAL_PROMPTS_LIMIT:-256}"
EVAL_GAMMA="${EVAL_GAMMA:-4}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-256}"
EVAL_WARMUP="${EVAL_WARMUP:-3}"
EVAL_REPEATS="${EVAL_REPEATS:-5}"
EVAL_BACKEND="${EVAL_BACKEND:-manual}"
EVAL_MODE="${EVAL_MODE:-sampling}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-1.0}"
EVAL_TOP_P="${EVAL_TOP_P:-0.9}"
EVAL_REPORT_TO_WANDB="${EVAL_REPORT_TO_WANDB:-true}"
EVAL_REPORT_CACHED_TO_WANDB="${EVAL_REPORT_CACHED_TO_WANDB:-true}"
FORCE_RERUN="${FORCE_RERUN:-false}"

RESULTS_ROOT="${RESULTS_ROOT:-/scratch/cs552-results}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints}"
HYDRA_ROOT="${HYDRA_ROOT:-outputs/qwen3-eval-sweep}"
PRETRAINED_CHECKPOINT_ROOT="${PRETRAINED_CHECKPOINT_ROOT:-${CHECKPOINT_ROOT}/pretrained}"

REPO_BRANCH="${REPO_BRANCH:-codex/qwen3}"

case "${DRAFT_SIZE}" in
  0.6b|0_6b)
    DRAFT_TAG="0p6b"
    LOSSES="${LOSSES:-fkl rkl jsd ce}"
    ;;
  1.7b|1_7b)
    DRAFT_TAG="1p7b"
    LOSSES="${LOSSES:-ce}"
    ;;
  *)
    echo "ERROR: DRAFT_SIZE must be 0.6b or 1.7b, got '${DRAFT_SIZE}'." >&2
    exit 1
    ;;
esac

EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_${DRAFT_TAG}_${DATA}_seed${SEED}}"
WANDB_GROUP="${WANDB_GROUP:-${EXPERIMENT_NAME}}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-qwen3_${DRAFT_TAG}}"

quote() {
  printf "%q" "$1"
}

run_command="DRAFT_SIZE=$(quote "${DRAFT_SIZE}")"
run_command+=" DATA=$(quote "${DATA}")"
run_command+=" SEED=$(quote "${SEED}")"
run_command+=" TARGET_ID=$(quote "${TARGET_ID}")"
run_command+=" PYTORCH_CUDA_ALLOC_CONF=$(quote "${PYTORCH_CUDA_ALLOC_CONF}")"
run_command+=" EVAL_PRETRAINED_BASELINE=$(quote "${EVAL_PRETRAINED_BASELINE}")"
run_command+=" EVAL_PROMPTS_JSONL=$(quote "${EVAL_PROMPTS_JSONL}")"
run_command+=" EVAL_PROMPTS_LIMIT=$(quote "${EVAL_PROMPTS_LIMIT}")"
run_command+=" EVAL_GAMMA=$(quote "${EVAL_GAMMA}")"
run_command+=" EVAL_MAX_NEW_TOKENS=$(quote "${EVAL_MAX_NEW_TOKENS}")"
run_command+=" EVAL_WARMUP=$(quote "${EVAL_WARMUP}")"
run_command+=" EVAL_REPEATS=$(quote "${EVAL_REPEATS}")"
run_command+=" EVAL_BACKEND=$(quote "${EVAL_BACKEND}")"
run_command+=" EVAL_MODE=$(quote "${EVAL_MODE}")"
run_command+=" EVAL_TEMPERATURE=$(quote "${EVAL_TEMPERATURE}")"
run_command+=" EVAL_TOP_P=$(quote "${EVAL_TOP_P}")"
run_command+=" EVAL_REPORT_TO_WANDB=$(quote "${EVAL_REPORT_TO_WANDB}")"
run_command+=" EVAL_REPORT_CACHED_TO_WANDB=$(quote "${EVAL_REPORT_CACHED_TO_WANDB}")"
run_command+=" FORCE_RERUN=$(quote "${FORCE_RERUN}")"
run_command+=" RESULTS_ROOT=$(quote "${RESULTS_ROOT}")"
run_command+=" CHECKPOINT_ROOT=$(quote "${CHECKPOINT_ROOT}")"
run_command+=" HYDRA_ROOT=$(quote "${HYDRA_ROOT}")"
run_command+=" PRETRAINED_CHECKPOINT_ROOT=$(quote "${PRETRAINED_CHECKPOINT_ROOT}")"
run_command+=" LOSSES=$(quote "${LOSSES}")"
run_command+=" EXPERIMENT_NAME=$(quote "${EXPERIMENT_NAME}")"
run_command+=" WANDB_GROUP=$(quote "${WANDB_GROUP}")"
run_command+=" RUN_NAME_PREFIX=$(quote "${RUN_NAME_PREFIX}")"
run_command+=" bash scripts/run_qwen3_eval_sweep.sh"
echo ">>> RUN_COMMAND chars: ${#run_command} (kept short for RunAI env limit)"

echo ">>> Submitting Qwen3 eval-only job: ${EXPERIMENT_NAME}"
echo ">>> Checkpoint root inside pod: ${CHECKPOINT_ROOT}"
echo ">>> Results root inside pod: ${RESULTS_ROOT}"
REPO_BRANCH="${REPO_BRANCH}" RUN_NAME="${EXPERIMENT_NAME}-eval" RUN_COMMAND="${run_command}" ./rcp_support/submit_train.sh "qwen3-${DRAFT_TAG}-eval"
