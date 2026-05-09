#!/bin/bash
# CS-552 — example Run:AI training job launcher.
#
# This is NOT a deliverable script. It is a helper/example for longer
# compute runs where you want the job to execute a training command and
# exit when the command finishes.
#
# For grading, submit the interactive submit.sh next to your notebook.
# Use this file only as a starting point for your own training runs.
#
# Training jobs are lower priority than interactive jobs and can be
# preempted/restarted by the scheduler. Your code must write checkpoints
# to /scratch and resume from them.

set -euo pipefail

# ============== EDIT THESE LINES ==============
RUN_NAME="fkl-20k-source"  # <-- A name for this training run, used in W&B and output dirs.
GASPAR="gaspar"              # <-- YOUR GASPAR EPFL username.
GROUP="g67"                  # <-- YOUR TEAM, e.g. g07.
WANDB_MODE="offline"              # <-- Optional: W&B mode (online, offline, or disabled).

# Edit this for your project. Keep outputs/checkpoints under /scratch.
REPO="/scratch/cs552-mnlp-${GASPAR}" # download code here at first
RUN_COMMAND="python scripts/train.py \
  run_name=${RUN_NAME} \
  loss=fkl \
  data=ultrachat_10k \
  train.max_steps=1500 \
  train.learning_rate=2e-5 \
  train.warmup_ratio=0.05 \
  train.lr_scheduler_type=constant_with_warmup \
  train.gradient_accumulation_steps=4 \
  train.logging_steps=50 \
  train.eval_steps=250 \
  train.save_steps=500 \
  train.report_to_wandb=true
"
TRAIN_COMMAND="cd ${REPO} && ${RUN_COMMAND}"
# ==============================================

# W&B runs are non-interactive in this background job, so authenticate with an
# environment variable before submitting:
#   export WANDB_API_KEY=...
# Optional overrides:
#   RUN_NAME=my-run WANDB_PROJECT=cs552-kdsd ./rcp_support/submit_train.sh
WANDB_PROJECT="${WANDB_PROJECT:-cs552-kdsd}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_NAME}}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"

if [[ "${GASPAR}" == "gaspar" || -z "${GASPAR}" ]]; then
    echo "ERROR: edit submit_train.sh and set GASPAR to your EPFL GASPAR username." >&2
    exit 1
fi

if [[ "${GROUP}" == "gXX" || -z "${GROUP}" ]]; then
    echo "ERROR: edit submit_train.sh and set GROUP to your team number (e.g. g07)." >&2
    exit 1
fi

if [[ "${TRAIN_COMMAND}" == *"<your-repo>"* || -z "${TRAIN_COMMAND}" ]]; then
    echo "ERROR: edit TRAIN_COMMAND before submitting a training job." >&2
    exit 1
fi

if [[ "${WANDB_MODE}" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: set WANDB_API_KEY before submitting, or run with WANDB_MODE=offline." >&2
    exit 1
fi

GPUS=1
NODE="${NODE:-a100-40g}"
SUFFIX="${1:-train}"
JOB_NAME="cs552-${GASPAR}-${GROUP}-${SUFFIX}-$(date +%H%M%S)"
PROJECT="course-cs-552-${GASPAR}"

IMAGE="registry.rcp.epfl.ch/course-cs-552/base-vllm:v1"

SCRATCH_PVC="course-cs-552-scratch-${GROUP}"
SHARED_RO_PVC="course-cs-552-shared-ro"
SHARED_RW_PVC="course-cs-552-shared-rw"

echo ">>> Submitting training job ${JOB_NAME}  (1 GPU)"

runai submit \
  --name "${JOB_NAME}" \
  -p "${PROJECT}" \
  --image "${IMAGE}" \
  --gpu "${GPUS}" \
  --large-shm \
  --node-pools "${NODE}" \
  --working-dir /scratch \
  --environment HF_HOME=/scratch/hf_cache \
  --environment HF_HUB_ENABLE_HF_TRANSFER=1 \
  --environment WANDB_DIR=/scratch/wandb \
  --environment WANDB_MODE="${WANDB_MODE}" \
  --environment WANDB_PROJECT="${WANDB_PROJECT}" \
  --environment WANDB_NAME="${WANDB_RUN_NAME}" \
  --environment WANDB_ENTITY="${WANDB_ENTITY}" \
  --environment WANDB_API_KEY="${WANDB_API_KEY:-}" \
  --environment TRAIN_COMMAND="${TRAIN_COMMAND}" \
  --existing-pvc "claimname=${SCRATCH_PVC},path=/scratch" \
  --existing-pvc "claimname=${SHARED_RO_PVC},path=/shared-ro" \
  --existing-pvc "claimname=${SHARED_RW_PVC},path=/shared-rw" \
  --command -- /bin/bash -lc "\
    set -euo pipefail && \
    mkdir -p /scratch/hf_cache /scratch/wandb /scratch/runs && \
    ln -sf \"\$(command -v python3)\" /usr/local/bin/python && \
    cd /scratch && \
    eval \"\${TRAIN_COMMAND}\""

cat <<EOF

>>> Training job submitted: ${JOB_NAME}

Watch it start:    runai describe job ${JOB_NAME} -p ${PROJECT}
Stream logs:       runai logs -f ${JOB_NAME} -p ${PROJECT}
List jobs:         runai list jobs -p ${PROJECT}
Stop the job:      runai delete job ${JOB_NAME} -p ${PROJECT}

This is a Run:AI training job, not an interactive Jupyter job. It exits
when TRAIN_COMMAND finishes. Make sure your training code writes
checkpoints to /scratch and can resume after preemption.
EOF
