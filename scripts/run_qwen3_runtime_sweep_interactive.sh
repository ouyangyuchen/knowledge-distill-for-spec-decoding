#!/bin/bash
# Interactive wrapper for the Qwen3 runtime sweep used in Siyuan's notebook.
#
# This keeps the original run_qwen3_runtime_sweep.sh implementation intact, but
# changes the default checkpoint family from target-generated RKL to
# target-generated JSD.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

DRAFT_SIZE="${DRAFT_SIZE:-0.6b}"
DATA="${DATA:-ultrachat_50k_target_gen}"
SEED="${SEED:-42}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/scratch/cs552-checkpoints}"

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
    export CHECKPOINT_RUN="qwen3_8btarget_${DRAFT_TAG}_tgen_jsd_${DATA}_seed${SEED}"
  else
    export CHECKPOINT_RUN="qwen3_8btarget_${DRAFT_TAG}_jsd_${DATA}_seed${SEED}"
  fi
fi

export CHECKPOINT_DIR="${CHECKPOINT_DIR:-${CHECKPOINT_ROOT}/${CHECKPOINT_RUN}}"
export DRAFT_PATH="${DRAFT_PATH:-${CHECKPOINT_DIR}/model}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_runtime_sweep_${CHECKPOINT_RUN}}"

bash scripts/run_qwen3_runtime_sweep.sh
