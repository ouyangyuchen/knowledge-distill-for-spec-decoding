#!/bin/bash
# Submit one Run:AI job for OPD training over all losses and selected data arms.
#
# Defaults:
#   draft  = Qwen/Qwen3-0.6B
#   target = Qwen/Qwen3-8B
#   data   = UltraChat source and target-generated arms
#   losses = fkl rkl jsd ce
#
# Override env vars as needed, e.g.
#   DATA_VARIANTS="ultrachat_50k" STEPS=1000 ./scripts/submit_opd_qwen3_8b_loss_sweep.sh
#   DRAFT_ID=Qwen/Qwen3-0.6B DRAFT_TAG=0p6b ./scripts/submit_opd_qwen3_8b_loss_sweep.sh
#   RUN_EVAL=false LOSSES="jsd fkl" ./scripts/submit_opd_qwen3_8b_loss_sweep.sh

set -euo pipefail

TARGET_ID="${TARGET_ID:-Qwen/Qwen3-8B}"
DRAFT_ID="${DRAFT_ID:-Qwen/Qwen3-0.6B}"
DRAFT_TAG="${DRAFT_TAG:-0p4b}"
DATA_VARIANTS="${DATA_VARIANTS:-ultrachat_50k ultrachat_50k_target_gen}"
LOSSES="${LOSSES:-fkl rkl jsd ce}"
STEPS="${STEPS:-0}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
LR="${LR:-2e-5}"
ALPHA="${ALPHA:-1.0}"
TEMP="${TEMP:-1.0}"
SEED="${SEED:-42}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
KD_CHUNK_SIZE="${KD_CHUNK_SIZE:-128}"
COMPILE_TARGET="${COMPILE_TARGET:-false}"
EVAL_REPORTING_STEPS="${EVAL_REPORTING_STEPS:-100}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OPD_GAMMA="${OPD_GAMMA:-4}"
OPD_ROLLOUT_MAX_NEW_TOKENS="${OPD_ROLLOUT_MAX_NEW_TOKENS:-128}"
OPD_MAX_PROMPT_TOKENS="${OPD_MAX_PROMPT_TOKENS:-}"
OPD_MODE="${OPD_MODE:-greedy}"
OPD_TEMPERATURE="${OPD_TEMPERATURE:-1.0}"
OPD_TOP_P="${OPD_TOP_P:-1.0}"
OPD_MAX_REPLAY_EXAMPLES="${OPD_MAX_REPLAY_EXAMPLES:-1}"

RUN_EVAL="${RUN_EVAL:-true}"
EVAL_PRETRAINED_BASELINE="${EVAL_PRETRAINED_BASELINE:-true}"
EVAL_PROMPTS_JSONL="${EVAL_PROMPTS_JSONL:-}"
EVAL_PROMPTS_LIMIT="${EVAL_PROMPTS_LIMIT:-50}"
EVAL_GAMMA="${EVAL_GAMMA:-4}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-256}"
EVAL_WARMUP="${EVAL_WARMUP:-1}"
EVAL_REPEATS="${EVAL_REPEATS:-3}"
EVAL_REPORT_TO_WANDB="${EVAL_REPORT_TO_WANDB:-true}"

REPO_BRANCH="${REPO_BRANCH:-opd}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-opd_qwen3_8btarget_${DRAFT_TAG}_seed${SEED}}"
WANDB_GROUP="${WANDB_GROUP:-${EXPERIMENT_NAME}}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-opd_qwen3_8btarget_${DRAFT_TAG}}"

quote() {
  printf "%q" "$1"
}

run_command="TARGET_ID=$(quote "${TARGET_ID}")"
run_command+=" DRAFT_ID=$(quote "${DRAFT_ID}")"
run_command+=" DRAFT_TAG=$(quote "${DRAFT_TAG}")"
run_command+=" DATA_VARIANTS=$(quote "${DATA_VARIANTS}")"
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
run_command+=" PYTORCH_CUDA_ALLOC_CONF=$(quote "${PYTORCH_CUDA_ALLOC_CONF}")"
run_command+=" OPD_GAMMA=$(quote "${OPD_GAMMA}")"
run_command+=" OPD_ROLLOUT_MAX_NEW_TOKENS=$(quote "${OPD_ROLLOUT_MAX_NEW_TOKENS}")"
run_command+=" OPD_MAX_PROMPT_TOKENS=$(quote "${OPD_MAX_PROMPT_TOKENS}")"
run_command+=" OPD_MODE=$(quote "${OPD_MODE}")"
run_command+=" OPD_TEMPERATURE=$(quote "${OPD_TEMPERATURE}")"
run_command+=" OPD_TOP_P=$(quote "${OPD_TOP_P}")"
run_command+=" OPD_MAX_REPLAY_EXAMPLES=$(quote "${OPD_MAX_REPLAY_EXAMPLES}")"
run_command+=" RUN_EVAL=$(quote "${RUN_EVAL}")"
run_command+=" EVAL_PRETRAINED_BASELINE=$(quote "${EVAL_PRETRAINED_BASELINE}")"
run_command+=" EVAL_PROMPTS_JSONL=$(quote "${EVAL_PROMPTS_JSONL}")"
run_command+=" EVAL_PROMPTS_LIMIT=$(quote "${EVAL_PROMPTS_LIMIT}")"
run_command+=" EVAL_GAMMA=$(quote "${EVAL_GAMMA}")"
run_command+=" EVAL_MAX_NEW_TOKENS=$(quote "${EVAL_MAX_NEW_TOKENS}")"
run_command+=" EVAL_WARMUP=$(quote "${EVAL_WARMUP}")"
run_command+=" EVAL_REPEATS=$(quote "${EVAL_REPEATS}")"
run_command+=" EVAL_REPORT_TO_WANDB=$(quote "${EVAL_REPORT_TO_WANDB}")"
run_command+=" EXPERIMENT_NAME=$(quote "${EXPERIMENT_NAME}")"
run_command+=" WANDB_GROUP=$(quote "${WANDB_GROUP}")"
run_command+=" RUN_NAME_PREFIX=$(quote "${RUN_NAME_PREFIX}")"
run_command+=" bash scripts/run_opd_qwen3_8b_loss_sweep.sh"

echo ">>> Submitting OPD Qwen3 8B-target loss/data sweep: ${EXPERIMENT_NAME}"
echo ">>> target=${TARGET_ID}"
echo ">>> draft=${DRAFT_ID}"
echo ">>> data variants=${DATA_VARIANTS}"
echo ">>> losses=${LOSSES}"
echo ">>> RUN_COMMAND chars: ${#run_command} (kept short for RunAI env limit)"

REPO_BRANCH="${REPO_BRANCH}" \
RUN_NAME="${EXPERIMENT_NAME}" \
RUN_COMMAND="${run_command}" \
./rcp_support/submit_train.sh "opd-qwen3-8b-${DRAFT_TAG}"
