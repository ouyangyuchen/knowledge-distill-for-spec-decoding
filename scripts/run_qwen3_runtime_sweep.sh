#!/bin/bash
# Evaluate one fixed Qwen3 checkpoint across runtime parameters.
# This script runs inside the RunAI pod from the checked-out repo.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source "${ROOT}/scripts/env.sh"
echo ">>> Python: ${KDSD_PYTHON}"

DRAFT_SIZE="${DRAFT_SIZE:-0.6b}"  # 0.6b or 1.7b
DATA="${DATA:-ultrachat_50k_target_gen}"
BASE_DATA="${BASE_DATA:-${DATA%_target_gen}}"
SEED="${SEED:-42}"
TARGET_ID="${TARGET_ID:-Qwen/Qwen3-8B}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GAMMAS="${GAMMAS:-1 2 4 6 8}"
TEMPERATURES="${TEMPERATURES:-0.0 0.5 1.0 2.0}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-256}"
EVAL_MAX_NEW_TOKENS_VALUES="${EVAL_MAX_NEW_TOKENS_VALUES:-128 256 512}"
EVAL_WARMUP="${EVAL_WARMUP:-1}"
EVAL_REPEATS="${EVAL_REPEATS:-1}"
EVAL_BACKEND="${EVAL_BACKEND:-vllm}"
EVAL_MODE="${EVAL_MODE:-auto}"  # auto: temperature 0 => greedy, otherwise sampling
EVAL_TOP_P="${EVAL_TOP_P:-1.0}"
EVAL_RUN_VANILLA_BASELINE="${EVAL_RUN_VANILLA_BASELINE:-true}"
EVAL_REPORT_TO_WANDB="${EVAL_REPORT_TO_WANDB:-true}"
FORCE_RERUN="${FORCE_RERUN:-false}"

RESULTS_ROOT="${RESULTS_ROOT:-/scratch/cs552-results}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/scratch/cs552-checkpoints}"
HYDRA_ROOT="${HYDRA_ROOT:-outputs/qwen3-runtime-sweep}"
PRETRAINED_CHECKPOINT_ROOT="${PRETRAINED_CHECKPOINT_ROOT:-${CHECKPOINT_ROOT}/pretrained}"
EVAL_PROMPTS_JSONL="${EVAL_PROMPTS_JSONL:-/scratch/cs552-data/processed/${BASE_DATA}/eval.jsonl}"
EVAL_PROMPTS_LIMIT="${EVAL_PROMPTS_LIMIT:-256}"

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

if [[ ! -d "${DRAFT_PATH}" ]]; then
  echo "ERROR: draft model directory does not exist: ${DRAFT_PATH}" >&2
  echo "Set CHECKPOINT_RUN, CHECKPOINT_DIR, or DRAFT_PATH to the target-generated JSD checkpoint." >&2
  exit 1
fi
if [[ ! -f "${DRAFT_PATH}/config.json" ]]; then
  echo "ERROR: draft model is missing config.json: ${DRAFT_PATH}" >&2
  exit 1
fi

target_override=()
if [[ -n "${TARGET_ID}" ]]; then
  target_override=("model.target=${TARGET_ID}")
fi

export PYTORCH_CUDA_ALLOC_CONF
mkdir -p "${RESULTS_ROOT}" "${HYDRA_ROOT}" "${PRETRAINED_CHECKPOINT_ROOT}" "$(dirname "${SUMMARY_CSV}")"

echo ">>> Qwen3 runtime sweep: ${EXPERIMENT_NAME}"
echo ">>> W&B group: ${WANDB_GROUP}"
echo ">>> checkpoint run: ${CHECKPOINT_RUN}"
echo ">>> draft path: ${DRAFT_PATH}"
echo ">>> data: ${DATA}"
echo ">>> prompts: ${EVAL_PROMPTS_JSONL}"
echo ">>> prompts limit: ${EVAL_PROMPTS_LIMIT}"
echo ">>> gammas: ${GAMMAS}"
echo ">>> temperatures: ${TEMPERATURES}"
echo ">>> max_new_tokens values: ${EVAL_MAX_NEW_TOKENS_VALUES}"
echo ">>> warmup/repeats: ${EVAL_WARMUP}/${EVAL_REPEATS}"
echo ">>> backend: ${EVAL_BACKEND}"
echo ">>> mode: ${EVAL_MODE}"
echo ">>> top_p: ${EVAL_TOP_P}"
echo ">>> run vanilla baseline: ${EVAL_RUN_VANILLA_BASELINE}"
echo ">>> report to W&B: ${EVAL_REPORT_TO_WANDB}"
echo ">>> force rerun: ${FORCE_RERUN}"
echo ">>> summary csv: ${SUMMARY_CSV}"

"${KDSD_PYTHON}" - "${SUMMARY_CSV}" <<'PY'
import csv
import sys

columns = [
    "run",
    "checkpoint_run",
    "status",
    "returncode",
    "elapsed_s",
    "eval_backend",
    "runtime_mode",
    "runtime_temperature",
    "runtime_top_p",
    "gamma",
    "max_new_tokens",
    "run_vanilla_baseline",
    "speedup",
    "acceptance_rate",
    "avg_accepted_tokens",
    "tokens_per_second",
    "sd_time_s",
    "vanilla_time_s",
    "n_prompts",
    "n_warmup",
    "n_repeats",
    "vllm_num_drafts",
    "vllm_num_draft_tokens",
    "vllm_num_accepted_tokens",
    "vllm_request_batch_size",
    "draft_path",
    "prompts_jsonl",
    "summary_path",
]

with open(sys.argv[1], "w", encoding="utf-8", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=columns)
    writer.writeheader()
PY

temp_tag() {
  local value="$1"
  value="${value//-/m}"
  value="${value//./p}"
  echo "${value}"
}

mode_for_temperature() {
  local value="$1"
  if [[ "${EVAL_MODE}" != "auto" ]]; then
    echo "${EVAL_MODE}"
  elif [[ "${value}" == "0" || "${value}" == "0.0" || "${value}" == "0.00" ]]; then
    echo "greedy"
  else
    echo "sampling"
  fi
}

append_summary_row() {
  "${KDSD_PYTHON}" - \
    "${SUMMARY_CSV}" \
    "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" "${11}" "${12}" "${13}" "${14}" "${15}" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

(
    csv_path,
    run_name,
    checkpoint_run,
    status,
    returncode,
    elapsed_s,
    backend,
    mode,
    temperature,
    top_p,
    gamma,
    max_new_tokens,
    run_vanilla,
    draft_path,
    prompts_jsonl,
    summary_path,
) = sys.argv[1:]

def metric(value):
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return value
    return ""

summary = {}
path = Path(summary_path)
if path.exists():
    with path.open("r", encoding="utf-8") as fh:
        loaded = json.load(fh)
    if isinstance(loaded, dict):
        summary = loaded

engine = (summary.get("engines") or {}).get("vllm") or {}
row = {
    "run": run_name,
    "checkpoint_run": checkpoint_run,
    "status": status,
    "returncode": returncode,
    "elapsed_s": elapsed_s,
    "eval_backend": backend,
    "runtime_mode": mode,
    "runtime_temperature": temperature,
    "runtime_top_p": top_p,
    "gamma": gamma,
    "max_new_tokens": max_new_tokens,
    "run_vanilla_baseline": run_vanilla,
    "speedup": metric(summary.get("speedup")),
    "acceptance_rate": metric(summary.get("acceptance_rate")),
    "avg_accepted_tokens": metric(summary.get("avg_accepted_tokens")),
    "tokens_per_second": metric(summary.get("tokens_per_second")),
    "sd_time_s": metric(summary.get("sd_time_s")),
    "vanilla_time_s": metric(summary.get("vanilla_time_s")),
    "n_prompts": summary.get("n_prompts", ""),
    "n_warmup": summary.get("n_warmup", ""),
    "n_repeats": summary.get("n_repeats", ""),
    "vllm_num_drafts": engine.get("num_drafts", ""),
    "vllm_num_draft_tokens": engine.get("num_draft_tokens", ""),
    "vllm_num_accepted_tokens": engine.get("num_accepted_tokens", ""),
    "vllm_request_batch_size": engine.get("request_batch_size", ""),
    "draft_path": draft_path,
    "prompts_jsonl": prompts_jsonl,
    "summary_path": summary_path,
}

with open(csv_path, "a", encoding="utf-8", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(row))
    writer.writerow(row)
PY
}

failures=0

for max_new_tokens in ${EVAL_MAX_NEW_TOKENS_VALUES}; do
  for gamma in ${GAMMAS}; do
    for temperature in ${TEMPERATURES}; do
      tag="$(temp_tag "${temperature}")"
      mode="$(mode_for_temperature "${temperature}")"
      run_name="${CHECKPOINT_RUN}_runtime_g${gamma}_t${tag}_max${max_new_tokens}"
      results_dir="${RESULTS_ROOT}/${run_name}"
      summary_path="${results_dir}/eval_summary.json"
      hydra_dir="${HYDRA_ROOT}/${run_name}"

      if [[ -f "${summary_path}" && "${FORCE_RERUN}" != "true" && "${FORCE_RERUN}" != "1" ]]; then
        echo ">>> Skipping cached ${run_name}"
        append_summary_row \
          "${run_name}" "${CHECKPOINT_RUN}" "cached" "0" "0.0" \
          "${EVAL_BACKEND}" "${mode}" "${temperature}" "${EVAL_TOP_P}" "${gamma}" \
          "${max_new_tokens}" "${EVAL_RUN_VANILLA_BASELINE}" "${DRAFT_PATH}" \
          "${EVAL_PROMPTS_JSONL}" "${summary_path}"
        continue
      fi

      echo ">>> Evaluating ${run_name}"
      start_s="$(date +%s)"
      set +e
      WANDB_GROUP="${WANDB_GROUP}" \
      WANDB_JOB_TYPE="eval" \
      "${KDSD_PYTHON}" scripts/evaluate_sd.py \
        model=qwen3 "data=${DATA}" "${target_override[@]}" \
        "draft=${DRAFT_PATH}" \
        "pretrained_checkpoint_root=${PRETRAINED_CHECKPOINT_ROOT}" \
        "prompts.jsonl=${EVAL_PROMPTS_JSONL}" \
        "prompts.hf_dataset=null" \
        "prompts.limit=${EVAL_PROMPTS_LIMIT}" \
        "runtime.mode=${mode}" \
        "runtime.temperature=${temperature}" \
        "runtime.top_p=${EVAL_TOP_P}" \
        "runtime.gamma=${gamma}" \
        "runtime.max_new_tokens=${max_new_tokens}" \
        "eval.backend=${EVAL_BACKEND}" \
        "eval.n_warmup=${EVAL_WARMUP}" \
        "eval.n_repeats=${EVAL_REPEATS}" \
        "eval.run_vanilla_baseline=${EVAL_RUN_VANILLA_BASELINE}" \
        "wandb.enabled=${EVAL_REPORT_TO_WANDB}" \
        "results_dir=${results_dir}" \
        "hydra.run.dir=${hydra_dir}" \
        "run_name=${run_name}"
      rc="$?"
      set -e
      end_s="$(date +%s)"
      elapsed_s="$((end_s - start_s))"

      status="ok"
      if [[ "${rc}" != "0" || ! -f "${summary_path}" ]]; then
        status="failed"
        failures=$((failures + 1))
        echo ">>> ERROR: ${run_name} failed with return code ${rc}" >&2
      fi

      append_summary_row \
        "${run_name}" "${CHECKPOINT_RUN}" "${status}" "${rc}" "${elapsed_s}" \
        "${EVAL_BACKEND}" "${mode}" "${temperature}" "${EVAL_TOP_P}" "${gamma}" \
        "${max_new_tokens}" "${EVAL_RUN_VANILLA_BASELINE}" "${DRAFT_PATH}" \
        "${EVAL_PROMPTS_JSONL}" "${summary_path}"
    done
  done
done

echo ">>> Wrote runtime sweep CSV: ${SUMMARY_CSV}"
if [[ "${failures}" != "0" ]]; then
  echo ">>> Runtime sweep finished with ${failures} failed eval(s)." >&2
  exit 1
fi
