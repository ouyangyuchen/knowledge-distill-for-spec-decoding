#!/bin/bash
# Run one Qwen2.5 target-generated loss sweep inside the RunAI job checkout.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source "${ROOT}/scripts/env.sh"
echo ">>> Python: ${KDSD_PYTHON}"

TARGET_ID="${TARGET_ID:-Qwen/Qwen2.5-3B-Instruct}"
DRAFT_ID="${DRAFT_ID:-Qwen/Qwen2.5-0.5B-Instruct}"
DATA="${DATA:-ultrachat_50k_target_gen}"
SOURCE_PROCESSED_DIR="${SOURCE_PROCESSED_DIR:-/scratch/cs552-data/processed/ultrachat_50k}"
TARGET_CACHE_DIR="${TARGET_CACHE_DIR:-/scratch/cs552-data/target_generated/ultrachat_50k_qwen25_3b}"

LOSSES="${LOSSES:-fkl rkl jsd ce}"
STEPS="${STEPS:-8000}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
LR="${LR:-2e-5}"
ALPHA="${ALPHA:-1.0}"
TEMP="${TEMP:-1.0}"
SEED="${SEED:-42}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
KD_CHUNK_SIZE="${KD_CHUNK_SIZE:-}"
COMPILE_TARGET="${COMPILE_TARGET:-false}"
EVAL_REPORTING_STEPS="${EVAL_REPORTING_STEPS:-2000}"
SAVE_STEPS="${SAVE_STEPS:-2000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-4}"
LOAD_BEST_MODEL_AT_END="${LOAD_BEST_MODEL_AT_END:-true}"
SAVE_BEST_MODEL="${SAVE_BEST_MODEL:-true}"
VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
TRAIN_REPORT_TO_WANDB="${TRAIN_REPORT_TO_WANDB:-true}"

RUN_EVAL="${RUN_EVAL:-true}"
EVAL_PRETRAINED_BASELINE="${EVAL_PRETRAINED_BASELINE:-true}"
EVAL_PROMPTS_JSONL="${EVAL_PROMPTS_JSONL:-${SOURCE_PROCESSED_DIR}/eval.jsonl}"
EVAL_PROMPTS_LIMIT="${EVAL_PROMPTS_LIMIT:-256}"
EVAL_GAMMA="${EVAL_GAMMA:-4}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-256}"
EVAL_WARMUP="${EVAL_WARMUP:-1}"
EVAL_REPEATS="${EVAL_REPEATS:-1}"
EVAL_BACKEND="${EVAL_BACKEND:-manual}"
EVAL_REPORT_TO_WANDB="${EVAL_REPORT_TO_WANDB:-true}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen25_3btarget_0p5b_targetgen50k_seed${SEED}}"
WANDB_GROUP="${WANDB_GROUP:-${EXPERIMENT_NAME}}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-qwen25_3btarget_0p5b}"

export VLLM_WORKER_MULTIPROC_METHOD
export PYTORCH_CUDA_ALLOC_CONF

is_true() {
  [[ "$1" == "true" || "$1" == "1" || "$1" == "yes" || "$1" == "y" ]]
}

meta_target_model() {
  local meta_path="$1"
  "${KDSD_PYTHON}" -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("target_model", ""))' "${meta_path}"
}

preflight_target_cache() {
  local train_json="${TARGET_CACHE_DIR}/train.jsonl"
  local val_json="${TARGET_CACHE_DIR}/val.jsonl"
  local train_meta="${TARGET_CACHE_DIR}/train.meta.json"
  local val_meta="${TARGET_CACHE_DIR}/val.meta.json"

  if [[ -f "${train_json}" && -f "${val_json}" && -f "${train_meta}" && -f "${val_meta}" ]]; then
    local train_target
    local val_target
    train_target="$(meta_target_model "${train_meta}")"
    val_target="$(meta_target_model "${val_meta}")"
    if [[ "${train_target}" == "${TARGET_ID}" && "${val_target}" == "${TARGET_ID}" ]]; then
      echo ">>> Reusing Qwen2.5 target-generated cache at ${TARGET_CACHE_DIR}"
      return 0
    fi

    echo "ERROR: target-generated cache exists at ${TARGET_CACHE_DIR}, but metadata target_model does not match." >&2
    echo "       expected: ${TARGET_ID}" >&2
    echo "       train.meta.json: ${train_target:-<missing target_model>}" >&2
    echo "       val.meta.json: ${val_target:-<missing target_model>}" >&2
    echo "       Choose a different TARGET_CACHE_DIR or regenerate this cache deliberately." >&2
    exit 1
  fi

  local found_any=false
  local path
  for path in "${train_json}" "${val_json}" "${train_meta}" "${val_meta}"; do
    if [[ -e "${path}" ]]; then
      found_any=true
    fi
  done

  if [[ "${found_any}" == "true" ]]; then
    echo "ERROR: partial target-generated cache found at ${TARGET_CACHE_DIR}." >&2
    echo "       Expected all of train.jsonl, val.jsonl, train.meta.json, val.meta.json." >&2
    for path in "${train_json}" "${val_json}" "${train_meta}" "${val_meta}"; do
      if [[ -e "${path}" ]]; then
        echo "       present: ${path}" >&2
      else
        echo "       missing: ${path}" >&2
      fi
    done
    echo "       Fix the cache or use a fresh TARGET_CACHE_DIR." >&2
    exit 1
  fi

  echo ">>> No dedicated Qwen2.5 target-generated cache found at ${TARGET_CACHE_DIR}"
  echo ">>> scripts/train.py will auto-generate train/val target responses before training."
}

echo ">>> Qwen2.5 target-generated experiment: ${EXPERIMENT_NAME}"
echo ">>> W&B group: ${WANDB_GROUP}"
echo ">>> target: ${TARGET_ID}"
echo ">>> draft: ${DRAFT_ID}"
echo ">>> data config: ${DATA}"
echo ">>> source processed dir: ${SOURCE_PROCESSED_DIR}"
echo ">>> target cache dir: ${TARGET_CACHE_DIR}"
echo ">>> losses: ${LOSSES}"
echo ">>> max_seq_len: ${MAX_SEQ_LEN}"
echo ">>> max_steps: ${STEPS}"
echo ">>> epochs: ${EPOCHS}"
echo ">>> per-device batch size: ${BATCH_SIZE}"
echo ">>> gradient accumulation steps: ${GRAD_ACCUM_STEPS}"
echo ">>> KD chunk size: ${KD_CHUNK_SIZE}"
echo ">>> compile_target: ${COMPILE_TARGET}"
echo ">>> vLLM worker multiprocessing method: ${VLLM_WORKER_MULTIPROC_METHOD}"
echo ">>> train report to W&B: ${TRAIN_REPORT_TO_WANDB}"
echo ">>> eval after training: ${RUN_EVAL}"
echo ">>> eval pretrained baseline: ${EVAL_PRETRAINED_BASELINE}"
echo ">>> eval prompts: ${EVAL_PROMPTS_JSONL}"
echo ">>> eval prompts limit: ${EVAL_PROMPTS_LIMIT}"
echo ">>> eval gamma: ${EVAL_GAMMA}"
echo ">>> eval max_new_tokens: ${EVAL_MAX_NEW_TOKENS}"
echo ">>> eval warmup/repeats: ${EVAL_WARMUP}/${EVAL_REPEATS}"
echo ">>> eval backend: ${EVAL_BACKEND}"
echo ">>> eval report to W&B: ${EVAL_REPORT_TO_WANDB}"

preflight_target_cache

common_data_overrides=(
  "data=${DATA}"
  "data.target_generated_dir=${TARGET_CACHE_DIR}"
  "data.train_path=${TARGET_CACHE_DIR}/train.jsonl"
  "data.val_path=${TARGET_CACHE_DIR}/val.jsonl"
  "data.target_generation.output_dir=${TARGET_CACHE_DIR}"
  "data.target_generation.source_processed_dir=${SOURCE_PROCESSED_DIR}"
)

common_model_overrides=(
  "model=qwen25"
  "model.target=${TARGET_ID}"
)

for loss in ${LOSSES}; do
  run_name="${RUN_NAME_PREFIX}_${loss}_${DATA}_seed${SEED}"

  if [[ "${loss}" == "ce" ]]; then
    loss_overrides=("loss=ce")
  else
    loss_overrides=("loss=${loss}" "loss.alpha=${ALPHA}" "loss.temperature=${TEMP}")
  fi

  echo ">>> Starting ${run_name}"
  WANDB_GROUP="${WANDB_GROUP}" \
  WANDB_NAME="${run_name}" \
  WANDB_JOB_TYPE="train" \
  "${KDSD_PYTHON}" scripts/train.py \
    "${common_model_overrides[@]}" "${common_data_overrides[@]}" "${loss_overrides[@]}" \
    "train.draft_init=${DRAFT_ID}" \
    "train.max_steps=${STEPS}" \
    "train.num_train_epochs=${EPOCHS}" \
    "train.per_device_train_batch_size=${BATCH_SIZE}" \
    "train.per_device_eval_batch_size=${BATCH_SIZE}" \
    "train.gradient_accumulation_steps=${GRAD_ACCUM_STEPS}" \
    "train.learning_rate=${LR}" \
    "train.compile_target=${COMPILE_TARGET}" \
    "train.eval_steps=${EVAL_REPORTING_STEPS}" \
    "train.report_to_wandb=${TRAIN_REPORT_TO_WANDB}" \
    "data.max_seq_len=${MAX_SEQ_LEN}" \
    "loss.chunk_size=${KD_CHUNK_SIZE}" \
    "seed=${SEED}" \
    "run_name=${run_name}"

  echo ">>> Finished ${run_name}"
done

if ! is_true "${RUN_EVAL}"; then
  echo ">>> RUN_EVAL=${RUN_EVAL}; skipping SD evaluation"
  exit 0
fi

echo ">>> Training sweep finished; starting manual SD evaluation"
eval_result_runs=()

if is_true "${EVAL_PRETRAINED_BASELINE}"; then
  baseline_eval_run="${RUN_NAME_PREFIX}_pretrain_${DATA}_seed${SEED}_eval_g${EVAL_GAMMA}_max${EVAL_MAX_NEW_TOKENS}"
  echo ">>> Evaluating pretrained draft baseline: ${baseline_eval_run}"
  WANDB_GROUP="${WANDB_GROUP}" \
  WANDB_JOB_TYPE="eval" \
  "${KDSD_PYTHON}" scripts/evaluate_sd.py \
    "${common_model_overrides[@]}" "${common_data_overrides[@]}" \
    "draft=${DRAFT_ID}" \
    "prompts.jsonl=${EVAL_PROMPTS_JSONL}" \
    "prompts.limit=${EVAL_PROMPTS_LIMIT}" \
    "runtime.gamma=${EVAL_GAMMA}" \
    "runtime.max_new_tokens=${EVAL_MAX_NEW_TOKENS}" \
    "eval.backend=${EVAL_BACKEND}" \
    "eval.n_warmup=${EVAL_WARMUP}" \
    "eval.n_repeats=${EVAL_REPEATS}" \
    "wandb.enabled=${EVAL_REPORT_TO_WANDB}" \
    "run_name=${baseline_eval_run}"
  eval_result_runs+=("${baseline_eval_run}")
fi

for loss in ${LOSSES}; do
  train_run="${RUN_NAME_PREFIX}_${loss}_${DATA}_seed${SEED}"
  eval_run="${train_run}_eval_g${EVAL_GAMMA}_max${EVAL_MAX_NEW_TOKENS}"

  echo ">>> Evaluating trained draft: ${eval_run}"
  WANDB_GROUP="${WANDB_GROUP}" \
  WANDB_JOB_TYPE="eval" \
  "${KDSD_PYTHON}" scripts/evaluate_sd.py \
    "${common_model_overrides[@]}" "${common_data_overrides[@]}" \
    "draft=checkpoints/${train_run}/model" \
    "prompts.jsonl=${EVAL_PROMPTS_JSONL}" \
    "prompts.limit=${EVAL_PROMPTS_LIMIT}" \
    "runtime.gamma=${EVAL_GAMMA}" \
    "runtime.max_new_tokens=${EVAL_MAX_NEW_TOKENS}" \
    "eval.backend=${EVAL_BACKEND}" \
    "eval.n_warmup=${EVAL_WARMUP}" \
    "eval.n_repeats=${EVAL_REPEATS}" \
    "wandb.enabled=${EVAL_REPORT_TO_WANDB}" \
    "run_name=${eval_run}"
  eval_result_runs+=("${eval_run}")
done

echo ">>> Final SD evaluation summary"
"${KDSD_PYTHON}" - "${eval_result_runs[@]}" <<'PY'
import json
import sys
from pathlib import Path

rows = []
for run in sys.argv[1:]:
    path = Path("/scratch/cs552-results") / run / "eval_summary.json"
    if not path.exists():
        rows.append((run, "missing", "", "", "", "", ""))
        continue
    with path.open(encoding="utf-8") as f:
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
