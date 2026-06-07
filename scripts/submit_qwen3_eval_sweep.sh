#!/bin/bash
# Submit one Run:AI job that evaluates every complete final Qwen3 checkpoint
# under /scratch/cs552-checkpoints. Run this from the repo root on the
# laptop/cluster login machine where runai is configured.
#
# Common overrides:
#   FORCE_RERUN=true EVAL_PROMPTS_LIMIT=200 ./scripts/submit_qwen3_eval_sweep.sh
#   CHECKPOINT_ROOT=/scratch/other-checkpoints ./scripts/submit_qwen3_eval_sweep.sh
#   SUMMARY_CSV=/scratch/cs552-results/my_qwen3_eval.csv ./scripts/submit_qwen3_eval_sweep.sh

set -euo pipefail

DATA="${DATA:-ultrachat_50k}"
SEED="${SEED:-42}"
TARGET_ID="${TARGET_ID:-}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

EVAL_PRETRAINED_BASELINE="${EVAL_PRETRAINED_BASELINE:-false}"
EVAL_PROMPTS_JSONL="${EVAL_PROMPTS_JSONL:-}"
EVAL_PROMPTS_LIMIT="${EVAL_PROMPTS_LIMIT:-256}"
EVAL_GAMMA="${EVAL_GAMMA:-4}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-256}"
EVAL_WARMUP="${EVAL_WARMUP:-1}"
EVAL_REPEATS="${EVAL_REPEATS:-1}"
EVAL_BACKEND="${EVAL_BACKEND:-vllm}"
EVAL_MODE="${EVAL_MODE:-greedy}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.0}"
EVAL_TOP_P="${EVAL_TOP_P:-1.0}"
EVAL_RUN_VANILLA_BASELINE="${EVAL_RUN_VANILLA_BASELINE:-true}"
EVAL_REPORT_TO_WANDB="${EVAL_REPORT_TO_WANDB:-true}"
EVAL_REPORT_CACHED_TO_WANDB="${EVAL_REPORT_CACHED_TO_WANDB:-true}"
FORCE_RERUN="${FORCE_RERUN:-true}"

RESULTS_ROOT="${RESULTS_ROOT:-/scratch/cs552-results}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/scratch/cs552-checkpoints}"
HYDRA_ROOT="${HYDRA_ROOT:-outputs/qwen3-eval-sweep}"
PRETRAINED_CHECKPOINT_ROOT="${PRETRAINED_CHECKPOINT_ROOT:-${CHECKPOINT_ROOT}/pretrained}"
SUMMARY_CSV="${SUMMARY_CSV:-${RESULTS_ROOT}/qwen3_checkpoint_eval_summary.csv}"

REPO_BRANCH="${REPO_BRANCH:-codex/vllm-eval}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_checkpoint_eval_sweep}"
WANDB_GROUP="${WANDB_GROUP:-${EXPERIMENT_NAME}}"

quote() {
  printf "%q" "$1"
}

run_command="DATA=$(quote "${DATA}")"
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
run_command+=" EVAL_RUN_VANILLA_BASELINE=$(quote "${EVAL_RUN_VANILLA_BASELINE}")"
run_command+=" EVAL_REPORT_TO_WANDB=$(quote "${EVAL_REPORT_TO_WANDB}")"
run_command+=" EVAL_REPORT_CACHED_TO_WANDB=$(quote "${EVAL_REPORT_CACHED_TO_WANDB}")"
run_command+=" FORCE_RERUN=$(quote "${FORCE_RERUN}")"
run_command+=" RESULTS_ROOT=$(quote "${RESULTS_ROOT}")"
run_command+=" CHECKPOINT_ROOT=$(quote "${CHECKPOINT_ROOT}")"
run_command+=" HYDRA_ROOT=$(quote "${HYDRA_ROOT}")"
run_command+=" PRETRAINED_CHECKPOINT_ROOT=$(quote "${PRETRAINED_CHECKPOINT_ROOT}")"
run_command+=" SUMMARY_CSV=$(quote "${SUMMARY_CSV}")"
run_command+=" EXPERIMENT_NAME=$(quote "${EXPERIMENT_NAME}")"
run_command+=" WANDB_GROUP=$(quote "${WANDB_GROUP}")"
run_command+=" bash scripts/run_qwen3_eval_sweep.sh"
echo ">>> RUN_COMMAND chars: ${#run_command} (kept short for RunAI env limit)"

echo ">>> Submitting Qwen3 checkpoint eval job: ${EXPERIMENT_NAME}"
echo ">>> Checkpoint root inside pod: ${CHECKPOINT_ROOT}"
echo ">>> Results root inside pod: ${RESULTS_ROOT}"
echo ">>> Summary CSV inside pod: ${SUMMARY_CSV}"
echo ">>> Eval backend/mode/temp/top_p: ${EVAL_BACKEND}/${EVAL_MODE}/${EVAL_TEMPERATURE}/${EVAL_TOP_P}"
echo ">>> W&B enabled: ${EVAL_REPORT_TO_WANDB}"
REPO_BRANCH="${REPO_BRANCH}" RUN_NAME="${EXPERIMENT_NAME}-eval" RUN_COMMAND="${run_command}" ./rcp_support/submit_train.sh "qwen3-eval"
