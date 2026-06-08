#!/bin/bash
# Submit one Run:AI job that evaluates the Qwen2.5 target-generated loss-sweep
# checkpoints with vLLM speculative decoding.
#
# Defaults match scripts/submit_qwen25_targetgen_loss_sweep.sh:
#   checkpoints: /scratch/cs552-checkpoints/qwen25_3btarget_0p5b_{loss}_ultrachat_50k_target_gen_seed42
#   gamma: 4
#   decoding: greedy
#   max_new_tokens: 256
#
# Common overrides:
#   LOSSES="jsd ce" ./scripts/submit_qwen25_targetgen_vllm_eval_sweep.sh
#   CHECKPOINT_RUNS="run_a run_b" ./scripts/submit_qwen25_targetgen_vllm_eval_sweep.sh
#   EVAL_PROMPTS_LIMIT=200 FORCE_RERUN=true ./scripts/submit_qwen25_targetgen_vllm_eval_sweep.sh

set -euo pipefail

TARGET_ID="${TARGET_ID:-Qwen/Qwen2.5-3B-Instruct}"
DATA="${DATA:-ultrachat_50k_target_gen}"
BASE_DATA="${BASE_DATA:-${DATA%_target_gen}}"
SEED="${SEED:-42}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-qwen25_3btarget_0p5b}"
LOSSES="${LOSSES:-fkl rkl jsd ce}"
CHECKPOINT_RUNS="${CHECKPOINT_RUNS:-}"

VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
KDSD_VENV="${KDSD_VENV:-/scratch/venvs/kdsd-vllm}"

EVAL_BACKEND="${EVAL_BACKEND:-vllm}"
EVAL_MODE="${EVAL_MODE:-greedy}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.0}"
EVAL_TOP_P="${EVAL_TOP_P:-1.0}"
EVAL_GAMMA="${EVAL_GAMMA:-4}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-256}"
EVAL_WARMUP="${EVAL_WARMUP:-1}"
EVAL_REPEATS="${EVAL_REPEATS:-3}"
EVAL_RUN_VANILLA_BASELINE="${EVAL_RUN_VANILLA_BASELINE:-true}"
EVAL_REPORT_TO_WANDB="${EVAL_REPORT_TO_WANDB:-false}"
FORCE_RERUN="${FORCE_RERUN:-false}"

RESULTS_ROOT="${RESULTS_ROOT:-/scratch/cs552-results}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/scratch/cs552-checkpoints}"
HYDRA_ROOT="${HYDRA_ROOT:-outputs/qwen25-targetgen-vllm-eval-sweep}"
PRETRAINED_CHECKPOINT_ROOT="${PRETRAINED_CHECKPOINT_ROOT:-${CHECKPOINT_ROOT}/pretrained}"
EVAL_PROMPTS_JSONL="${EVAL_PROMPTS_JSONL:-/scratch/cs552-data/processed/${BASE_DATA}/eval.jsonl}"
EVAL_PROMPTS_LIMIT="${EVAL_PROMPTS_LIMIT:-50}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen25_targetgen_vllm_eval_g${EVAL_GAMMA}_max${EVAL_MAX_NEW_TOKENS}_seed${SEED}}"
WANDB_GROUP="${WANDB_GROUP:-${EXPERIMENT_NAME}}"
SUMMARY_CSV="${SUMMARY_CSV:-${RESULTS_ROOT}/${EXPERIMENT_NAME}.csv}"

REPO_BRANCH="${REPO_BRANCH:-codex/vllm-eval}"
WANDB_MODE="${WANDB_MODE:-disabled}"
DRY_RUN="${DRY_RUN:-false}"

quote() {
  printf "%q" "$1"
}

run_command="TARGET_ID=$(quote "${TARGET_ID}")"
run_command+=" DATA=$(quote "${DATA}")"
run_command+=" BASE_DATA=$(quote "${BASE_DATA}")"
run_command+=" SEED=$(quote "${SEED}")"
run_command+=" RUN_NAME_PREFIX=$(quote "${RUN_NAME_PREFIX}")"
run_command+=" LOSSES=$(quote "${LOSSES}")"
run_command+=" CHECKPOINT_RUNS=$(quote "${CHECKPOINT_RUNS}")"
run_command+=" VLLM_WORKER_MULTIPROC_METHOD=$(quote "${VLLM_WORKER_MULTIPROC_METHOD}")"
run_command+=" PYTORCH_CUDA_ALLOC_CONF=$(quote "${PYTORCH_CUDA_ALLOC_CONF}")"
run_command+=" KDSD_VENV=$(quote "${KDSD_VENV}")"
run_command+=" EVAL_BACKEND=$(quote "${EVAL_BACKEND}")"
run_command+=" EVAL_MODE=$(quote "${EVAL_MODE}")"
run_command+=" EVAL_TEMPERATURE=$(quote "${EVAL_TEMPERATURE}")"
run_command+=" EVAL_TOP_P=$(quote "${EVAL_TOP_P}")"
run_command+=" EVAL_GAMMA=$(quote "${EVAL_GAMMA}")"
run_command+=" EVAL_MAX_NEW_TOKENS=$(quote "${EVAL_MAX_NEW_TOKENS}")"
run_command+=" EVAL_WARMUP=$(quote "${EVAL_WARMUP}")"
run_command+=" EVAL_REPEATS=$(quote "${EVAL_REPEATS}")"
run_command+=" EVAL_RUN_VANILLA_BASELINE=$(quote "${EVAL_RUN_VANILLA_BASELINE}")"
run_command+=" EVAL_REPORT_TO_WANDB=$(quote "${EVAL_REPORT_TO_WANDB}")"
run_command+=" FORCE_RERUN=$(quote "${FORCE_RERUN}")"
run_command+=" RESULTS_ROOT=$(quote "${RESULTS_ROOT}")"
run_command+=" CHECKPOINT_ROOT=$(quote "${CHECKPOINT_ROOT}")"
run_command+=" HYDRA_ROOT=$(quote "${HYDRA_ROOT}")"
run_command+=" PRETRAINED_CHECKPOINT_ROOT=$(quote "${PRETRAINED_CHECKPOINT_ROOT}")"
run_command+=" EVAL_PROMPTS_JSONL=$(quote "${EVAL_PROMPTS_JSONL}")"
run_command+=" EVAL_PROMPTS_LIMIT=$(quote "${EVAL_PROMPTS_LIMIT}")"
run_command+=" EXPERIMENT_NAME=$(quote "${EXPERIMENT_NAME}")"
run_command+=" WANDB_GROUP=$(quote "${WANDB_GROUP}")"
run_command+=" SUMMARY_CSV=$(quote "${SUMMARY_CSV}")"
run_command+=" bash scripts/run_qwen25_targetgen_vllm_eval_sweep.sh"

echo ">>> RUN_COMMAND chars: ${#run_command} (kept short for RunAI env limit)"
echo ">>> Submitting Qwen2.5 target-generated vLLM eval sweep: ${EXPERIMENT_NAME}"
echo ">>> Checkpoint root inside pod: ${CHECKPOINT_ROOT}"
echo ">>> Results root inside pod: ${RESULTS_ROOT}"
echo ">>> Summary CSV inside pod: ${SUMMARY_CSV}"
echo ">>> Losses/checkpoint runs: ${LOSSES} / ${CHECKPOINT_RUNS:-<derived from losses>}"
echo ">>> Eval backend/mode/temp/top_p: ${EVAL_BACKEND}/${EVAL_MODE}/${EVAL_TEMPERATURE}/${EVAL_TOP_P}"
echo ">>> Eval gamma/max_new: ${EVAL_GAMMA}/${EVAL_MAX_NEW_TOKENS}"
echo ">>> Eval prompts: ${EVAL_PROMPTS_JSONL} limit=${EVAL_PROMPTS_LIMIT}"
echo ">>> vLLM worker multiprocessing method: ${VLLM_WORKER_MULTIPROC_METHOD}"
echo ">>> W&B mode/enabled: ${WANDB_MODE}/${EVAL_REPORT_TO_WANDB}"

if [[ "${DRY_RUN}" == "true" || "${DRY_RUN}" == "1" ]]; then
  echo ">>> DRY_RUN=${DRY_RUN}; not submitting. RUN_COMMAND:"
  printf '%s\n' "${run_command}"
  exit 0
fi

REPO_BRANCH="${REPO_BRANCH}" \
RUN_NAME="${EXPERIMENT_NAME}-eval" \
RUN_COMMAND="${run_command}" \
WANDB_MODE="${WANDB_MODE}" \
./rcp_support/submit_train.sh "qwen25-vllm-eval"
