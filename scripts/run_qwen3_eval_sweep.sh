#!/bin/bash
# Evaluate every complete final Qwen3 checkpoint under CHECKPOINT_ROOT.
# This script runs inside the RunAI pod from the checked-out repo.

set -euo pipefail

DATA="${DATA:-ultrachat_50k}"
SEED="${SEED:-42}"
TARGET_ID="${TARGET_ID:-}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# The checkpoint sweep is dynamic; pretrained baselines are intentionally off by
# default so this job evaluates the shared checkpoint folder only.
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
FORCE_RERUN="${FORCE_RERUN:-false}"

RESULTS_ROOT="${RESULTS_ROOT:-/scratch/cs552-results}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/scratch/cs552-checkpoints}"
HYDRA_ROOT="${HYDRA_ROOT:-outputs/qwen3-eval-sweep}"
PRETRAINED_CHECKPOINT_ROOT="${PRETRAINED_CHECKPOINT_ROOT:-${CHECKPOINT_ROOT}/pretrained}"
SUMMARY_CSV="${SUMMARY_CSV:-${RESULTS_ROOT}/qwen3_checkpoint_eval_summary.csv}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_checkpoint_eval_sweep}"
WANDB_GROUP="${WANDB_GROUP:-${EXPERIMENT_NAME}}"

export PYTORCH_CUDA_ALLOC_CONF
mkdir -p "${RESULTS_ROOT}" "${HYDRA_ROOT}" "${PRETRAINED_CHECKPOINT_ROOT}"

if [[ "${EVAL_PRETRAINED_BASELINE}" == "true" || "${EVAL_PRETRAINED_BASELINE}" == "1" ]]; then
  echo ">>> WARNING: EVAL_PRETRAINED_BASELINE is ignored by this dynamic checkpoint sweep." >&2
fi

echo ">>> Qwen3 eval sweep: ${EXPERIMENT_NAME}"
echo ">>> W&B group: ${WANDB_GROUP}"
echo ">>> checkpoint root: ${CHECKPOINT_ROOT}"
echo ">>> results root: ${RESULTS_ROOT}"
echo ">>> hydra root: ${HYDRA_ROOT}"
echo ">>> summary csv: ${SUMMARY_CSV}"
echo ">>> eval prompts override: ${EVAL_PROMPTS_JSONL:-<checkpoint config>}"
echo ">>> eval prompts limit: ${EVAL_PROMPTS_LIMIT}"
echo ">>> eval gamma/max_new: ${EVAL_GAMMA}/${EVAL_MAX_NEW_TOKENS}"
echo ">>> eval warmup/repeats: ${EVAL_WARMUP}/${EVAL_REPEATS}"
echo ">>> eval backend: ${EVAL_BACKEND}"
echo ">>> eval mode/temp/top_p: ${EVAL_MODE}/${EVAL_TEMPERATURE}/${EVAL_TOP_P}"
echo ">>> run vanilla baseline: ${EVAL_RUN_VANILLA_BASELINE}"
echo ">>> eval report to W&B: ${EVAL_REPORT_TO_WANDB}"
echo ">>> report cached evals to W&B: ${EVAL_REPORT_CACHED_TO_WANDB}"
echo ">>> force rerun: ${FORCE_RERUN}"

python scripts/qwen3_eval_sweep.py run \
  --checkpoint-root "${CHECKPOINT_ROOT}" \
  --results-root "${RESULTS_ROOT}" \
  --hydra-root "${HYDRA_ROOT}" \
  --summary-csv "${SUMMARY_CSV}" \
  --target-id "${TARGET_ID}" \
  --pretrained-checkpoint-root "${PRETRAINED_CHECKPOINT_ROOT}" \
  --prompts-jsonl-override "${EVAL_PROMPTS_JSONL}" \
  --prompts-limit "${EVAL_PROMPTS_LIMIT}" \
  --gamma "${EVAL_GAMMA}" \
  --max-new-tokens "${EVAL_MAX_NEW_TOKENS}" \
  --warmup "${EVAL_WARMUP}" \
  --repeats "${EVAL_REPEATS}" \
  --backend "${EVAL_BACKEND}" \
  --mode "${EVAL_MODE}" \
  --temperature "${EVAL_TEMPERATURE}" \
  --top-p "${EVAL_TOP_P}" \
  --run-vanilla-baseline "${EVAL_RUN_VANILLA_BASELINE}" \
  --report-to-wandb "${EVAL_REPORT_TO_WANDB}" \
  --report-cached-to-wandb "${EVAL_REPORT_CACHED_TO_WANDB}" \
  --force-rerun "${FORCE_RERUN}" \
  --wandb-group "${WANDB_GROUP}"
