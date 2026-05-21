#!/bin/bash
# Submit one Run:AI job that runs one Qwen3 training run per loss.
# Run this from the repo root on the laptop/cluster login machine where runai is
# configured. Defaults are for a full one-epoch UltraChat-50k sweep on one
# 40GB A100. Override env vars as needed, e.g.
#   LOSSES="ce fkl rkl jsd" WANDB_GROUP=qwen3_0p6b_sweep ./scripts/submit_qwen3_loss_sweep.sh
#   DRAFT_SIZE=1.7b LOSSES="ce" ./scripts/submit_qwen3_loss_sweep.sh

set -euo pipefail

DRAFT_SIZE="${DRAFT_SIZE:-0.6b}"  # 0.6b or 1.7b
DATA="${DATA:-ultrachat_50k}"
STEPS="${STEPS:-0}" # Set max_steps=0 to use num_train_epochs
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-32}"
LR="${LR:-1e-5}"
ALPHA="${ALPHA:-1.0}" # only used for non-CE losses; ignored when LOSSES includes only "ce".
TEMP="${TEMP:-1.0}"
SEED="${SEED:-42}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
KD_CHUNK_SIZE="${KD_CHUNK_SIZE:-128}"
COMPILE_TARGET="${COMPILE_TARGET:-false}"
TARGET_ID="${TARGET_ID:-}" # optional override, e.g. Qwen/Qwen3-14B
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_EVAL="${RUN_EVAL:-true}"
EVAL_PRETRAINED_BASELINE="${EVAL_PRETRAINED_BASELINE:-true}"
EVAL_PROMPTS_JSONL="${EVAL_PROMPTS_JSONL:-/scratch/cs552-data/processed/${DATA}/eval.jsonl}"
EVAL_PROMPTS_LIMIT="${EVAL_PROMPTS_LIMIT:-50}"
EVAL_GAMMA="${EVAL_GAMMA:-4}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-256}"
EVAL_WARMUP="${EVAL_WARMUP:-1}"
EVAL_REPEATS="${EVAL_REPEATS:-3}"

REPO_BRANCH="${REPO_BRANCH:-codex/qwen3}"

case "${DRAFT_SIZE}" in
  0.6b|0_6b)
    DRAFT_ID="Qwen/Qwen3-0.6B"
    DRAFT_TAG="0p6b"
    LOSSES="${LOSSES:-fkl rkl jsd ce}"
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
echo ">>> max_steps: ${STEPS}"
echo ">>> epochs: ${EPOCHS}"
echo ">>> per-device batch size: ${BATCH_SIZE}"
echo ">>> gradient accumulation steps: ${GRAD_ACCUM_STEPS}"
echo ">>> KD chunk size: ${KD_CHUNK_SIZE}"
echo ">>> compile_target: ${COMPILE_TARGET}"
echo ">>> target override: ${TARGET_ID:-<config default>}"
echo ">>> run eval after training: ${RUN_EVAL}"
echo ">>> eval pretrained baseline: ${EVAL_PRETRAINED_BASELINE}"
echo ">>> eval prompts: ${EVAL_PROMPTS_JSONL}"
echo ">>> eval prompts limit: ${EVAL_PROMPTS_LIMIT}"
echo ">>> eval gamma: ${EVAL_GAMMA}"
echo ">>> eval max_new_tokens: ${EVAL_MAX_NEW_TOKENS}"
echo ">>> eval warmup/repeats: ${EVAL_WARMUP}/${EVAL_REPEATS}"

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
train.num_train_epochs=${EPOCHS} \
train.per_device_train_batch_size=${BATCH_SIZE} \
train.per_device_eval_batch_size=${BATCH_SIZE} \
train.gradient_accumulation_steps=${GRAD_ACCUM_STEPS} \
train.learning_rate=${LR} \
train.compile_target=${COMPILE_TARGET} \
data.max_seq_len=${MAX_SEQ_LEN} \
loss.chunk_size=${KD_CHUNK_SIZE} \
seed=${SEED} \
run_name=\${run_name}

  echo ">>> Finished \${run_name}"
done

if [[ "${RUN_EVAL}" == "true" || "${RUN_EVAL}" == "1" ]]; then
  echo ">>> Training sweep finished; starting SD evaluation"

  eval_result_runs=()

  if [[ "${EVAL_PRETRAINED_BASELINE}" == "true" || "${EVAL_PRETRAINED_BASELINE}" == "1" ]]; then
    baseline_eval_run="qwen3_${DRAFT_TAG}_pretrain_${DATA}_seed${SEED}_eval_g${EVAL_GAMMA}_max${EVAL_MAX_NEW_TOKENS}"
    echo ">>> Evaluating pretrained draft baseline: \${baseline_eval_run}"
    python scripts/evaluate_sd.py \
model=qwen3 data=${DATA} ${TARGET_OVERRIDE} \
draft=${DRAFT_ID} \
prompts.jsonl=${EVAL_PROMPTS_JSONL} \
prompts.limit=${EVAL_PROMPTS_LIMIT} \
runtime.gamma=${EVAL_GAMMA} \
runtime.max_new_tokens=${EVAL_MAX_NEW_TOKENS} \
eval.n_warmup=${EVAL_WARMUP} \
eval.n_repeats=${EVAL_REPEATS} \
run_name=\${baseline_eval_run}
    eval_result_runs+=("\${baseline_eval_run}")
  fi

  for loss in \${LOSSES}; do
    train_run="qwen3_${DRAFT_TAG}_\${loss}_${DATA}_seed${SEED}"
    eval_run="\${train_run}_eval_g${EVAL_GAMMA}_max${EVAL_MAX_NEW_TOKENS}"

    echo ">>> Evaluating trained draft: \${eval_run}"
    python scripts/evaluate_sd.py \
model=qwen3 data=${DATA} ${TARGET_OVERRIDE} \
draft=checkpoints/\${train_run}/model \
prompts.jsonl=${EVAL_PROMPTS_JSONL} \
prompts.limit=${EVAL_PROMPTS_LIMIT} \
runtime.gamma=${EVAL_GAMMA} \
runtime.max_new_tokens=${EVAL_MAX_NEW_TOKENS} \
eval.n_warmup=${EVAL_WARMUP} \
eval.n_repeats=${EVAL_REPEATS} \
run_name=\${eval_run}
    eval_result_runs+=("\${eval_run}")
  done

  echo ">>> Final SD evaluation summary"
  python -c 'import json, sys
from pathlib import Path

rows = []
for run in sys.argv[1:]:
    path = Path("/scratch/cs552-results") / run / "eval_summary.json"
    if not path.exists():
        rows.append((run, "missing", "", "", "", "", ""))
        continue
    with path.open() as f:
        summary = json.load(f)
    rows.append((
        run,
        "%.3fx" % summary.get("speedup", float("nan")),
        "%.3f" % summary.get("acceptance_rate", float("nan")),
        "%.2f" % summary.get("avg_accepted_tokens", float("nan")),
        "%.2f" % summary.get("tokens_per_second", float("nan")),
        "%.2f" % summary.get("sd_time_s", float("nan")),
        "%.2f" % summary.get("vanilla_time_s", float("nan")),
    ))

headers = ("run", "speedup", "accept", "avg_acc", "tok/s", "sd_s", "vanilla_s")
widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]
line = "  ".join(str(h).ljust(w) for h, w in zip(headers, widths))
print(line)
print("-" * len(line))
for row in rows:
    print("  ".join(str(x).ljust(w) for x, w in zip(row, widths)))
' "\${eval_result_runs[@]}"
else
  echo ">>> RUN_EVAL=${RUN_EVAL}; skipping SD evaluation"
fi
EOF

echo ">>> Submitting one sequential Qwen3 experiment job: ${EXPERIMENT_NAME}"
REPO_BRANCH="${REPO_BRANCH}" RUN_NAME="${EXPERIMENT_NAME}" RUN_COMMAND="${run_command}" ./rcp_support/submit_train.sh "qwen3-${DRAFT_TAG}-sweep"
