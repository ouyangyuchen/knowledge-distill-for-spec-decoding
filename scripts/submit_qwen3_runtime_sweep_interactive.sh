#!/bin/bash
# Run one foreground job that evaluates one target-generated JSD Qwen3 checkpoint
# across runtime parameters. Run this from the repo root inside the RunAI pod.
#
# Common overrides:
#   CHECKPOINT_RUN=qwen3_8btarget_0p6b_tgen_jsd_ultrachat_50k_target_gen_seed42 bash scripts/submit_qwen3_runtime_sweep_interactive.sh
#   GAMMAS="1 2 4 6 8" TEMPERATURES="1.0" EVAL_MAX_NEW_TOKENS_VALUES="256" bash scripts/submit_qwen3_runtime_sweep_interactive.sh
#   DRAFT_PATH=/scratch/cs552-checkpoints/my_run/model bash scripts/submit_qwen3_runtime_sweep_interactive.sh

set -euo pipefail

# Runtime settings.
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_DIR="${WANDB_DIR:-/scratch/wandb-syz}"
HYDRA_OUTPUTS_DIR="${HYDRA_OUTPUTS_DIR:-/scratch/cs552-hydra-outputs-syz}"

DRAFT_SIZE="${DRAFT_SIZE:-0.6b}"
DATA="${DATA:-ultrachat_50k_target_gen}"
BASE_DATA="${BASE_DATA:-${DATA%_target_gen}}"
SEED="${SEED:-42}"
TARGET_ID="${TARGET_ID:-Qwen/Qwen3-8B}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
KDSD_VENV="${KDSD_VENV:-/scratch/venvs/kdsd-vllm}"

GAMMAS="${GAMMAS:-1 2 4 6 8}"
TEMPERATURES="${TEMPERATURES:-1.0}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-256}"
EVAL_MAX_NEW_TOKENS_VALUES="${EVAL_MAX_NEW_TOKENS_VALUES:-256}"
EVAL_WARMUP="${EVAL_WARMUP:-1}"
EVAL_REPEATS="${EVAL_REPEATS:-3}"
EVAL_BACKEND="${EVAL_BACKEND:-vllm}"
EVAL_MODE="${EVAL_MODE:-auto}"
EVAL_TOP_P="${EVAL_TOP_P:-0.9}"
EVAL_RUN_VANILLA_BASELINE="${EVAL_RUN_VANILLA_BASELINE:-true}"
EVAL_REPORT_TO_WANDB="${EVAL_REPORT_TO_WANDB:-true}"
FORCE_RERUN="${FORCE_RERUN:-true}"

RESULTS_ROOT="${RESULTS_ROOT:-/scratch/cs552-results-syz}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/scratch/cs552-checkpoints}"
HYDRA_ROOT="${HYDRA_ROOT:-outputs/qwen3-runtime-sweep}"
PRETRAINED_CHECKPOINT_ROOT="${PRETRAINED_CHECKPOINT_ROOT:-${CHECKPOINT_ROOT}/pretrained}"
EVAL_PROMPTS_JSONL="${EVAL_PROMPTS_JSONL:-/scratch/cs552-data/processed/${BASE_DATA}/eval.jsonl}"
EVAL_PROMPTS_LIMIT="${EVAL_PROMPTS_LIMIT:-50}"

if [[ "${WANDB_MODE}" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
  echo "ERROR: set WANDB_API_KEY before running online W&B, or run with WANDB_MODE=offline." >&2
  exit 1
fi

case "${DRAFT_SIZE}" in
  0.6b|0_6b)
    DRAFT_TAG="0p6b"
    ;;
  1.7b|1_7b)
    DRAFT_TAG="1p7b"
    ;;
  *)
    echo "ERROR: DRAFT_SIZE must be 0.6b or 1.7b, got '${DRAFT_SIZE}'." >&2
    exit 1
    ;;
esac

if [[ -z "${CHECKPOINT_RUN:-}" ]]; then
  if [[ "${DATA}" == *_target_gen ]]; then
    CHECKPOINT_RUN="qwen3_8btarget_${DRAFT_TAG}_tgen_jsd_${DATA}_seed${SEED}"
  else
    CHECKPOINT_RUN="qwen3_8btarget_${DRAFT_TAG}_jsd_${DATA}_seed${SEED}"
  fi
fi
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${CHECKPOINT_ROOT}/${CHECKPOINT_RUN}}"
DRAFT_PATH="${DRAFT_PATH:-${CHECKPOINT_DIR}/model}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_runtime_sweep_${CHECKPOINT_RUN}}"
WANDB_GROUP="${WANDB_GROUP:-${EXPERIMENT_NAME}}"
SUMMARY_CSV="${SUMMARY_CSV:-${RESULTS_ROOT}/${EXPERIMENT_NAME}.csv}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

HF_HOME="${HF_HOME:-/scratch/hf_cache-syz}"
mkdir -p "${HF_HOME}" "${WANDB_DIR}" "${CHECKPOINT_ROOT}" "${RESULTS_ROOT}" "${HYDRA_OUTPUTS_DIR}"

export HF_HOME
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export WANDB_DIR WANDB_MODE
export WANDB_PROJECT="${WANDB_PROJECT:-cs552-kdsd}"
export WANDB_NAME="${EXPERIMENT_NAME}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export CHECKPOINT_ROOT RESULTS_ROOT HYDRA_OUTPUTS_DIR
export DRAFT_SIZE DATA BASE_DATA SEED TARGET_ID
export PYTORCH_CUDA_ALLOC_CONF KDSD_VENV
export GAMMAS TEMPERATURES
export EVAL_MAX_NEW_TOKENS EVAL_MAX_NEW_TOKENS_VALUES
export EVAL_WARMUP EVAL_REPEATS EVAL_BACKEND EVAL_MODE EVAL_TOP_P
export EVAL_RUN_VANILLA_BASELINE EVAL_REPORT_TO_WANDB FORCE_RERUN
export HYDRA_ROOT PRETRAINED_CHECKPOINT_ROOT EVAL_PROMPTS_JSONL EVAL_PROMPTS_LIMIT
export CHECKPOINT_RUN CHECKPOINT_DIR DRAFT_PATH
export EXPERIMENT_NAME WANDB_GROUP SUMMARY_CSV

echo ">>> Running Qwen3 runtime sweep in foreground: ${EXPERIMENT_NAME}"
echo ">>> Fixed checkpoint: ${CHECKPOINT_RUN}"
echo ">>> Draft path: ${DRAFT_PATH}"
echo ">>> Results root: ${RESULTS_ROOT}"
echo ">>> Summary CSV: ${SUMMARY_CSV}"
echo ">>> Gammas: ${GAMMAS}"
echo ">>> Temperatures: ${TEMPERATURES}"
echo ">>> Max new token lengths: ${EVAL_MAX_NEW_TOKENS_VALUES}"
echo ">>> Eval backend/mode/top_p: ${EVAL_BACKEND}/${EVAL_MODE}/${EVAL_TOP_P}"
echo ">>> Python venv: ${KDSD_VENV}"
echo ">>> W&B enabled: ${EVAL_REPORT_TO_WANDB}"

if [[ ! -f "scripts/run_qwen3_runtime_sweep_interactive.sh" ]]; then
  echo "ERROR: scripts/run_qwen3_runtime_sweep_interactive.sh not found in ${ROOT}" >&2
  exit 1
fi

bash scripts/run_qwen3_runtime_sweep_interactive.sh
