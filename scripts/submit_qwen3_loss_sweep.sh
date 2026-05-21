#!/bin/bash
# Submit one Run:AI job that runs one Qwen3 training run per loss.
# Run this from the repo root on the laptop/cluster login machine where runai is
# configured. Override env vars as needed, e.g.
#   LOSSES="ce fkl rkl jsd" WANDB_GROUP=qwen3_0p6b_sweep ./scripts/submit_qwen3_loss_sweep.sh
#   DRAFT_SIZE=1.7b LOSSES="ce" ./scripts/submit_qwen3_loss_sweep.sh

set -euo pipefail

DRAFT_SIZE="${DRAFT_SIZE:-0.6b}"  # 0.6b or 1.7b
DATA="${DATA:-ultrachat_10k}"
STEPS="${STEPS:-400}"
LR="${LR:-1e-5}"
ALPHA="${ALPHA:-1.0}" # only used for non-CE losses; ignored when LOSSES includes only "ce".
TEMP="${TEMP:-1.0}"
SEED="${SEED:-42}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
KD_CHUNK_SIZE="${KD_CHUNK_SIZE:-128}"
COMPILE_TARGET="${COMPILE_TARGET:-false}"
TARGET_ID="${TARGET_ID:-}" # optional override, e.g. Qwen/Qwen3-8B
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

REPO_BRANCH="${REPO_BRANCH:-codex/qwen3}"

case "${DRAFT_SIZE}" in
  0.6b|0_6b)
    DRAFT_ID="Qwen/Qwen3-0.6B"
    DRAFT_TAG="0p6b"
    LOSSES="${LOSSES:-ce fkl rkl jsd}"
    ;;
  1.7b|1_7b)
    DRAFT_ID="Qwen/Qwen3-1.7B"
    DRAFT_TAG="1p7b"
    LOSSES="${LOSSES:-ce}"
    cat >&2 <<'EOF'
WARNING: Qwen3-1.7B full fine-tuning with a resident Qwen3-14B KD target is
likely to OOM on a 40GB A100 for fkl/rkl/jsd. This script defaults to CE only
for DRAFT_SIZE=1.7b. Set ALLOW_QWEN3_1_7B_FULL_KD=1 and LOSSES="fkl rkl jsd"
only if you intentionally want to try those full-finetune jobs.
EOF
    ;;
  *)
    echo "ERROR: DRAFT_SIZE must be 0.6b or 1.7b, got '${DRAFT_SIZE}'." >&2
    exit 1
    ;;
esac

EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_${DRAFT_TAG}_${DATA}_seed${SEED}}"
WANDB_GROUP="${WANDB_GROUP:-${EXPERIMENT_NAME}}"
TARGET_OVERRIDE=""
if [[ -n "${TARGET_ID}" ]]; then
  TARGET_OVERRIDE="model.target=${TARGET_ID}"
fi

for loss in ${LOSSES}; do
  if [[ "${DRAFT_TAG}" == "1p7b" && "${loss}" != "ce" && "${ALLOW_QWEN3_1_7B_FULL_KD:-0}" != "1" ]]; then
    echo "ERROR: refusing likely-OOM Qwen3-1.7B KD loss '${loss}' without ALLOW_QWEN3_1_7B_FULL_KD=1." >&2
    exit 1
  fi
done

read -r -d '' run_command <<EOF || true
set -euo pipefail

LOSSES="${LOSSES}"

echo ">>> Qwen3 experiment: ${EXPERIMENT_NAME}"
echo ">>> W&B group: ${WANDB_GROUP}"
echo ">>> Losses: \${LOSSES}"
echo ">>> max_seq_len: ${MAX_SEQ_LEN}"
echo ">>> KD chunk size: ${KD_CHUNK_SIZE}"
echo ">>> compile_target: ${COMPILE_TARGET}"
echo ">>> target override: ${TARGET_ID:-<config default>}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}"

for loss in \${LOSSES}; do
  run_name="qwen3_${DRAFT_TAG}_\${loss}_${DATA}_seed${SEED}"

  if [[ "\${loss}" == "ce" ]]; then
    loss_overrides="loss=ce"
  else
    loss_overrides="loss=\${loss} loss.alpha=${ALPHA} loss.temperature=${TEMP}"
  fi

  echo ">>> Starting \${run_name}"
  WANDB_GROUP="${WANDB_GROUP}" \
  WANDB_NAME="\${run_name}" \
  WANDB_JOB_TYPE="train" \
python scripts/train.py \
model=qwen3 train=a100_40gb_qwen3 data=${DATA} ${TARGET_OVERRIDE} \${loss_overrides} \
train.draft_init=${DRAFT_ID} \
train.max_steps=${STEPS} \
train.learning_rate=${LR} \
train.compile_target=${COMPILE_TARGET} \
data.max_seq_len=${MAX_SEQ_LEN} \
loss.chunk_size=${KD_CHUNK_SIZE} \
seed=${SEED} \
run_name=\${run_name}

  echo ">>> Finished \${run_name}"
done
EOF

echo ">>> Submitting one sequential Qwen3 experiment job: ${EXPERIMENT_NAME}"
REPO_BRANCH="${REPO_BRANCH}" RUN_NAME="${EXPERIMENT_NAME}" RUN_COMMAND="${run_command}" ./rcp_support/submit_train.sh "qwen3-${DRAFT_TAG}-sweep"
