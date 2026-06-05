#!/bin/bash
# Run OPD training for every selected loss and data arm, then optionally
# evaluate pretrained and OPD-trained drafts. Executed inside the RunAI pod.

set -euo pipefail

TARGET_ID="${TARGET_ID:-Qwen/Qwen3-8B}"
DRAFT_ID="${DRAFT_ID:-Qwen/Qwen3-0.6B}"
DRAFT_TAG="${DRAFT_TAG:-0p6b}"
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
EVAL_PROMPTS_LIMIT="${EVAL_PROMPTS_LIMIT:-50}"
EVAL_GAMMA="${EVAL_GAMMA:-4}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-256}"
EVAL_WARMUP="${EVAL_WARMUP:-1}"
EVAL_REPEATS="${EVAL_REPEATS:-3}"
EVAL_REPORT_TO_WANDB="${EVAL_REPORT_TO_WANDB:-true}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-opd_qwen3_8btarget_${DRAFT_TAG}_seed${SEED}}"
WANDB_GROUP="${WANDB_GROUP:-${EXPERIMENT_NAME}}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-opd_qwen3_8btarget_${DRAFT_TAG}}"

target_override=("model.target=${TARGET_ID}")
opd_prompt_override=()
if [[ -n "${OPD_MAX_PROMPT_TOKENS}" ]]; then
  opd_prompt_override=("train.opd.max_prompt_tokens=${OPD_MAX_PROMPT_TOKENS}")
fi

echo ">>> OPD Qwen3 experiment: ${EXPERIMENT_NAME}"
echo ">>> W&B group: ${WANDB_GROUP}"
echo ">>> target: ${TARGET_ID}"
echo ">>> draft: ${DRAFT_ID}"
echo ">>> data variants: ${DATA_VARIANTS}"
echo ">>> losses: ${LOSSES}"
echo ">>> max_seq_len: ${MAX_SEQ_LEN}"
echo ">>> max_steps: ${STEPS}"
echo ">>> epochs: ${EPOCHS}"
echo ">>> per-device batch size: ${BATCH_SIZE}"
echo ">>> gradient accumulation steps: ${GRAD_ACCUM_STEPS}"
echo ">>> OPD gamma/rollout: ${OPD_GAMMA}/${OPD_ROLLOUT_MAX_NEW_TOKENS}"
echo ">>> OPD mode/temp/top_p: ${OPD_MODE}/${OPD_TEMPERATURE}/${OPD_TOP_P}"
echo ">>> OPD replay examples per prompt: ${OPD_MAX_REPLAY_EXAMPLES}"
echo ">>> KD chunk size: ${KD_CHUNK_SIZE}"
echo ">>> compile_target: ${COMPILE_TARGET}"
echo ">>> run eval after training: ${RUN_EVAL}"

export PYTORCH_CUDA_ALLOC_CONF

trained_runs=()
for data in ${DATA_VARIANTS}; do
  for loss in ${LOSSES}; do
    run_name="${RUN_NAME_PREFIX}_${loss}_${data}_seed${SEED}"

    if [[ "${loss}" == "ce" ]]; then
      loss_overrides=("loss=ce")
    else
      loss_overrides=("loss=${loss}" "loss.alpha=${ALPHA}" "loss.temperature=${TEMP}")
    fi

    echo ">>> Starting OPD run ${run_name}"
    WANDB_GROUP="${WANDB_GROUP}" \
    WANDB_NAME="${run_name}" \
    WANDB_JOB_TYPE="opd-train" \
    python scripts/train.py \
      model=qwen3 train=opd "data=${data}" "${target_override[@]}" "${loss_overrides[@]}" \
      "train.draft_init=${DRAFT_ID}" \
      "train.max_steps=${STEPS}" \
      "train.num_train_epochs=${EPOCHS}" \
      "train.per_device_train_batch_size=${BATCH_SIZE}" \
      "train.per_device_eval_batch_size=${BATCH_SIZE}" \
      "train.gradient_accumulation_steps=${GRAD_ACCUM_STEPS}" \
      "train.learning_rate=${LR}" \
      "train.compile_target=${COMPILE_TARGET}" \
      "train.eval_steps=${EVAL_REPORTING_STEPS}" \
      "train.opd.gamma=${OPD_GAMMA}" \
      "train.opd.rollout_max_new_tokens=${OPD_ROLLOUT_MAX_NEW_TOKENS}" \
      "train.opd.mode=${OPD_MODE}" \
      "train.opd.temperature=${OPD_TEMPERATURE}" \
      "train.opd.top_p=${OPD_TOP_P}" \
      "train.opd.max_replay_examples=${OPD_MAX_REPLAY_EXAMPLES}" \
      "${opd_prompt_override[@]}" \
      "data.max_seq_len=${MAX_SEQ_LEN}" \
      "loss.chunk_size=${KD_CHUNK_SIZE}" \
      "seed=${SEED}" \
      "run_name=${run_name}"

    trained_runs+=("${run_name}:${data}")
    echo ">>> Finished OPD run ${run_name}"
  done
done

if [[ "${RUN_EVAL}" != "true" && "${RUN_EVAL}" != "1" ]]; then
  echo ">>> RUN_EVAL=${RUN_EVAL}; skipping SD evaluation"
  exit 0
fi

echo ">>> OPD sweep finished; starting SD evaluation"
eval_result_runs=()

for data in ${DATA_VARIANTS}; do
  eval_prompts_jsonl="${EVAL_PROMPTS_JSONL:-/scratch/cs552-data/processed/${data}/eval.jsonl}"
  if [[ "${data}" == *_target_gen ]]; then
    eval_prompts_jsonl="${EVAL_PROMPTS_JSONL:-/scratch/cs552-data/processed/ultrachat_50k/eval.jsonl}"
  fi

  if [[ "${EVAL_PRETRAINED_BASELINE}" == "true" || "${EVAL_PRETRAINED_BASELINE}" == "1" ]]; then
    baseline_eval_run="${RUN_NAME_PREFIX}_pretrain_${data}_seed${SEED}_eval_g${EVAL_GAMMA}_max${EVAL_MAX_NEW_TOKENS}"
    echo ">>> Evaluating pretrained draft baseline: ${baseline_eval_run}"
    python scripts/evaluate_sd.py \
      model=qwen3 "data=${data}" "${target_override[@]}" \
      "draft=${DRAFT_ID}" \
      "prompts.jsonl=${eval_prompts_jsonl}" \
      "prompts.limit=${EVAL_PROMPTS_LIMIT}" \
      "runtime.gamma=${EVAL_GAMMA}" \
      "runtime.max_new_tokens=${EVAL_MAX_NEW_TOKENS}" \
      "eval.n_warmup=${EVAL_WARMUP}" \
      "eval.n_repeats=${EVAL_REPEATS}" \
      "wandb.enabled=${EVAL_REPORT_TO_WANDB}" \
      "run_name=${baseline_eval_run}"
    eval_result_runs+=("${baseline_eval_run}")
  fi
done

for item in "${trained_runs[@]}"; do
  train_run="${item%%:*}"
  data="${item#*:}"
  eval_prompts_jsonl="${EVAL_PROMPTS_JSONL:-/scratch/cs552-data/processed/${data}/eval.jsonl}"
  if [[ "${data}" == *_target_gen ]]; then
    eval_prompts_jsonl="${EVAL_PROMPTS_JSONL:-/scratch/cs552-data/processed/ultrachat_50k/eval.jsonl}"
  fi
  eval_run="${train_run}_eval_g${EVAL_GAMMA}_max${EVAL_MAX_NEW_TOKENS}"

  echo ">>> Evaluating OPD-trained draft: ${eval_run}"
  python scripts/evaluate_sd.py \
    model=qwen3 "data=${data}" "${target_override[@]}" \
    "draft=checkpoints/${train_run}/model" \
    "prompts.jsonl=${eval_prompts_jsonl}" \
    "prompts.limit=${EVAL_PROMPTS_LIMIT}" \
    "runtime.gamma=${EVAL_GAMMA}" \
    "runtime.max_new_tokens=${EVAL_MAX_NEW_TOKENS}" \
    "eval.n_warmup=${EVAL_WARMUP}" \
    "eval.n_repeats=${EVAL_REPEATS}" \
    "wandb.enabled=${EVAL_REPORT_TO_WANDB}" \
    "run_name=${eval_run}"
  eval_result_runs+=("${eval_run}")
done

echo ">>> Final OPD SD evaluation summary"
python - "${eval_result_runs[@]}" <<'PY'
import json
import sys
from pathlib import Path

rows = []
for run in sys.argv[1:]:
    path = Path("/scratch/cs552-results") / run / "eval_summary.json"
    if not path.exists():
        rows.append((run, "missing", "", "", "", "", ""))
        continue
    with path.open() as f:
        summary = json.load(f)
    rows.append(
        (
            run,
            "%.3fx" % summary.get("speedup", float("nan")),
            "%.3f" % summary.get("acceptance_rate", float("nan")),
            "%.2f" % summary.get("avg_accepted_tokens", float("nan")),
            "%.2f" % summary.get("tokens_per_second", float("nan")),
            "%.2f" % summary.get("sd_time_s", float("nan")),
            "%.2f" % summary.get("vanilla_time_s", float("nan")),
        )
    )

headers = ("run", "speedup", "accept", "avg_acc", "tok/s", "sd_s", "vanilla_s")
widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]
line = "  ".join(str(h).ljust(w) for h, w in zip(headers, widths))
print(line)
print("-" * len(line))
for row in rows:
    print("  ".join(str(x).ljust(w) for x, w in zip(row, widths)))
PY
