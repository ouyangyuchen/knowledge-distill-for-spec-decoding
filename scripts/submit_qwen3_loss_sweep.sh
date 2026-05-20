#!/bin/bash
# Submit one Run:AI job per Qwen3 loss using rcp_support/submit_train.sh.
# Run this from the repo root on the laptop/cluster login machine where runai is
# configured. Override env vars as needed, e.g.
#   DRAFT_SIZE=1.7b LOSSES="ce" ./scripts/submit_qwen3_loss_sweep.sh

set -euo pipefail

DRAFT_SIZE="${DRAFT_SIZE:-0.6b}"  # 0.6b or 1.7b
DATA="${DATA:-ultrachat_10k}"
STEPS="${STEPS:-4000}"
LR="${LR:-2e-5}"
ALPHA="${ALPHA:-0.5}"
TEMP="${TEMP:-1.0}"
SEED="${SEED:-42}"

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

for loss in ${LOSSES}; do
  if [[ "${DRAFT_TAG}" == "1p7b" && "${loss}" != "ce" && "${ALLOW_QWEN3_1_7B_FULL_KD:-0}" != "1" ]]; then
    echo "ERROR: refusing likely-OOM Qwen3-1.7B KD loss '${loss}' without ALLOW_QWEN3_1_7B_FULL_KD=1." >&2
    exit 1
  fi

  run_name="qwen3_${DRAFT_TAG}_${loss}_${DATA}_seed${SEED}"

  if [[ "${loss}" == "ce" ]]; then
    loss_overrides="loss=ce"
  else
    loss_overrides="loss=${loss} loss.alpha=${ALPHA} loss.temperature=${TEMP}"
  fi

  run_command="python scripts/train.py \
model=qwen3 train=a100_40gb_qwen3 data=${DATA} ${loss_overrides} \
train.draft_init=${DRAFT_ID} \
train.max_steps=${STEPS} \
train.learning_rate=${LR} \
seed=${SEED} \
run_name=${run_name}"

  echo ">>> Submitting ${run_name}"
  RUN_NAME="${run_name}" RUN_COMMAND="${run_command}" ./rcp_support/submit_train.sh "qwen3-${DRAFT_TAG}-${loss}"
done
