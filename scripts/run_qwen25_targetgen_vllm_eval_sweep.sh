#!/bin/bash
# Evaluate Qwen2.5 target-generated loss-sweep checkpoints with vLLM SD.
# This script runs inside the RunAI job checkout.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
source "${ROOT}/scripts/env.sh"
echo ">>> Python: ${KDSD_PYTHON}"

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
EVAL_REPEATS="${EVAL_REPEATS:-1}"
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

export VLLM_WORKER_MULTIPROC_METHOD
export PYTORCH_CUDA_ALLOC_CONF
mkdir -p "${RESULTS_ROOT}" "${HYDRA_ROOT}" "${PRETRAINED_CHECKPOINT_ROOT}" "$(dirname "${SUMMARY_CSV}")"

is_true() {
  [[ "$1" == "true" || "$1" == "1" || "$1" == "yes" || "$1" == "y" ]]
}

if [[ -n "${CHECKPOINT_RUNS}" ]]; then
  checkpoint_runs=(${CHECKPOINT_RUNS})
else
  checkpoint_runs=()
  for loss in ${LOSSES}; do
    checkpoint_runs+=("${RUN_NAME_PREFIX}_${loss}_${DATA}_seed${SEED}")
  done
fi

echo ">>> Qwen2.5 target-generated vLLM eval sweep: ${EXPERIMENT_NAME}"
echo ">>> W&B group: ${WANDB_GROUP}"
echo ">>> checkpoint root: ${CHECKPOINT_ROOT}"
echo ">>> checkpoints: ${checkpoint_runs[*]}"
echo ">>> target: ${TARGET_ID}"
echo ">>> data: ${DATA}"
echo ">>> prompts: ${EVAL_PROMPTS_JSONL}"
echo ">>> prompts limit: ${EVAL_PROMPTS_LIMIT}"
echo ">>> eval backend/mode/temp/top_p: ${EVAL_BACKEND}/${EVAL_MODE}/${EVAL_TEMPERATURE}/${EVAL_TOP_P}"
echo ">>> eval gamma/max_new: ${EVAL_GAMMA}/${EVAL_MAX_NEW_TOKENS}"
echo ">>> eval warmup/repeats: ${EVAL_WARMUP}/${EVAL_REPEATS}"
echo ">>> run vanilla baseline: ${EVAL_RUN_VANILLA_BASELINE}"
echo ">>> report to W&B: ${EVAL_REPORT_TO_WANDB}"
echo ">>> force rerun: ${FORCE_RERUN}"
echo ">>> vLLM worker multiprocessing method: ${VLLM_WORKER_MULTIPROC_METHOD}"
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
    "target",
    "draft",
    "gamma",
    "max_new_tokens",
    "runtime_mode",
    "runtime_temperature",
    "runtime_top_p",
    "n_prompts",
    "n_warmup",
    "n_repeats",
    "speedup",
    "acceptance_rate",
    "avg_accepted_tokens",
    "tokens_per_second",
    "sd_time_s",
    "vanilla_time_s",
    "vllm_sd_time_s",
    "vllm_vanilla_time_s",
    "vllm_speedup",
    "vllm_tokens_per_second",
    "vllm_vanilla_tokens_per_second",
    "vllm_acceptance_rate",
    "vllm_avg_accepted_tokens",
    "vllm_num_drafts",
    "vllm_num_draft_tokens",
    "vllm_num_accepted_tokens",
    "vllm_accepted_tokens_per_pos",
    "vllm_request_batch_size",
    "results_dir",
    "summary_path",
    "draft_path",
]

with open(sys.argv[1], "w", encoding="utf-8", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=columns)
    writer.writeheader()
PY

append_summary_row() {
  "${KDSD_PYTHON}" - \
    "${SUMMARY_CSV}" \
    "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" \
    "${10}" "${11}" "${12}" "${13}" "${14}" <<'PY'
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
    gamma,
    max_new_tokens,
    mode,
    temperature,
    top_p,
    results_dir,
    summary_path,
    draft_path,
    target_id,
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
    "target": summary.get("target", target_id),
    "draft": summary.get("draft", draft_path),
    "gamma": gamma,
    "max_new_tokens": max_new_tokens,
    "runtime_mode": mode,
    "runtime_temperature": temperature,
    "runtime_top_p": top_p,
    "n_prompts": summary.get("n_prompts", ""),
    "n_warmup": summary.get("n_warmup", ""),
    "n_repeats": summary.get("n_repeats", ""),
    "speedup": metric(summary.get("speedup")),
    "acceptance_rate": metric(summary.get("acceptance_rate")),
    "avg_accepted_tokens": metric(summary.get("avg_accepted_tokens")),
    "tokens_per_second": metric(summary.get("tokens_per_second")),
    "sd_time_s": metric(summary.get("sd_time_s")),
    "vanilla_time_s": metric(summary.get("vanilla_time_s")),
    "vllm_sd_time_s": metric(engine.get("sd_time_s")),
    "vllm_vanilla_time_s": metric(engine.get("vanilla_time_s")),
    "vllm_speedup": metric(engine.get("speedup")),
    "vllm_tokens_per_second": metric(engine.get("tokens_per_second")),
    "vllm_vanilla_tokens_per_second": metric(engine.get("vanilla_tokens_per_second")),
    "vllm_acceptance_rate": metric(engine.get("acceptance_rate")),
    "vllm_avg_accepted_tokens": metric(engine.get("avg_accepted_tokens")),
    "vllm_num_drafts": engine.get("num_drafts", ""),
    "vllm_num_draft_tokens": engine.get("num_draft_tokens", ""),
    "vllm_num_accepted_tokens": engine.get("num_accepted_tokens", ""),
    "vllm_accepted_tokens_per_pos": json.dumps(engine.get("accepted_tokens_per_pos", [])),
    "vllm_request_batch_size": engine.get("request_batch_size", ""),
    "results_dir": results_dir,
    "summary_path": summary_path,
    "draft_path": draft_path,
}

with open(csv_path, "a", encoding="utf-8", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(row))
    writer.writerow(row)

print(
    ">>> METRICS "
    f"run={run_name} status={status} rc={returncode} "
    f"speedup={row['speedup']} acceptance_rate={row['acceptance_rate']} "
    f"avg_accepted_tokens={row['avg_accepted_tokens']} "
    f"tokens_per_second={row['tokens_per_second']} "
    f"sd_time_s={row['sd_time_s']} vanilla_time_s={row['vanilla_time_s']} "
    f"vllm_num_drafts={row['vllm_num_drafts']} "
    f"vllm_num_draft_tokens={row['vllm_num_draft_tokens']} "
    f"vllm_num_accepted_tokens={row['vllm_num_accepted_tokens']} "
    f"summary={summary_path}"
)
PY
}

failures=0

for checkpoint_run in "${checkpoint_runs[@]}"; do
  draft_path="${CHECKPOINT_ROOT}/${checkpoint_run}/model"
  eval_run="${checkpoint_run}_vllm_eval_g${EVAL_GAMMA}_greedy_max${EVAL_MAX_NEW_TOKENS}"
  results_dir="${RESULTS_ROOT}/${eval_run}"
  summary_path="${results_dir}/eval_summary.json"

  if [[ ! -f "${draft_path}/config.json" ]]; then
    echo ">>> ERROR: missing checkpoint model config: ${draft_path}/config.json" >&2
    append_summary_row \
      "${eval_run}" "${checkpoint_run}" "missing_checkpoint" "1" "0.0" \
      "${EVAL_GAMMA}" "${EVAL_MAX_NEW_TOKENS}" "${EVAL_MODE}" "${EVAL_TEMPERATURE}" "${EVAL_TOP_P}" \
      "${results_dir}" "${summary_path}" "${draft_path}" "${TARGET_ID}"
    failures=$((failures + 1))
    continue
  fi

  if [[ -f "${summary_path}" ]] && ! is_true "${FORCE_RERUN}"; then
    echo ">>> Reusing cached eval summary: ${summary_path}"
    append_summary_row \
      "${eval_run}" "${checkpoint_run}" "cached" "0" "0.0" \
      "${EVAL_GAMMA}" "${EVAL_MAX_NEW_TOKENS}" "${EVAL_MODE}" "${EVAL_TEMPERATURE}" "${EVAL_TOP_P}" \
      "${results_dir}" "${summary_path}" "${draft_path}" "${TARGET_ID}"
    continue
  fi

  echo ">>> Evaluating ${checkpoint_run}"
  echo ">>> draft path: ${draft_path}"
  start_s="$("${KDSD_PYTHON}" -c 'import time; print(time.perf_counter())')"
  set +e
  WANDB_GROUP="${WANDB_GROUP}" \
  WANDB_JOB_TYPE="eval" \
  "${KDSD_PYTHON}" scripts/evaluate_sd.py \
    "model=qwen25" \
    "model.target=${TARGET_ID}" \
    "data=${DATA}" \
    "draft=${draft_path}" \
    "pretrained_checkpoint_root=${PRETRAINED_CHECKPOINT_ROOT}" \
    "prompts.jsonl=${EVAL_PROMPTS_JSONL}" \
    "prompts.hf_dataset=null" \
    "prompts.limit=${EVAL_PROMPTS_LIMIT}" \
    "runtime.mode=${EVAL_MODE}" \
    "runtime.temperature=${EVAL_TEMPERATURE}" \
    "runtime.top_p=${EVAL_TOP_P}" \
    "runtime.gamma=${EVAL_GAMMA}" \
    "runtime.max_new_tokens=${EVAL_MAX_NEW_TOKENS}" \
    "eval.backend=${EVAL_BACKEND}" \
    "eval.n_warmup=${EVAL_WARMUP}" \
    "eval.n_repeats=${EVAL_REPEATS}" \
    "eval.run_vanilla_baseline=${EVAL_RUN_VANILLA_BASELINE}" \
    "wandb.enabled=${EVAL_REPORT_TO_WANDB}" \
    "results_dir=${results_dir}" \
    "hydra.run.dir=${HYDRA_ROOT}/${eval_run}" \
    "run_name=${eval_run}"
  rc=$?
  set -e
  elapsed_s="$("${KDSD_PYTHON}" -c "import time; print(f'{time.perf_counter() - float(${start_s}):.1f}')")"

  status="ok"
  if [[ "${rc}" -ne 0 || ! -f "${summary_path}" ]]; then
    status="failed"
    failures=$((failures + 1))
  fi
  append_summary_row \
    "${eval_run}" "${checkpoint_run}" "${status}" "${rc}" "${elapsed_s}" \
    "${EVAL_GAMMA}" "${EVAL_MAX_NEW_TOKENS}" "${EVAL_MODE}" "${EVAL_TEMPERATURE}" "${EVAL_TOP_P}" \
    "${results_dir}" "${summary_path}" "${draft_path}" "${TARGET_ID}"
done

echo ">>> Final Qwen2.5 vLLM eval summary CSV: ${SUMMARY_CSV}"
"${KDSD_PYTHON}" - "${SUMMARY_CSV}" <<'PY'
import csv
import sys

columns = [
    "run",
    "status",
    "speedup",
    "acceptance_rate",
    "avg_accepted_tokens",
    "tokens_per_second",
    "sd_time_s",
    "vanilla_time_s",
    "vllm_num_drafts",
    "vllm_num_draft_tokens",
    "vllm_num_accepted_tokens",
]
with open(sys.argv[1], encoding="utf-8", newline="") as fh:
    rows = list(csv.DictReader(fh))
if not rows:
    raise SystemExit(0)
widths = {
    col: max(len(col), *(len(str(row.get(col, ""))) for row in rows))
    for col in columns
}
line = "  ".join(col.ljust(widths[col]) for col in columns)
print(line)
print("-" * len(line))
for row in rows:
    print("  ".join(str(row.get(col, "")).ljust(widths[col]) for col in columns))
PY

if [[ "${failures}" -gt 0 ]]; then
  echo ">>> ERROR: ${failures} eval(s) failed or had missing checkpoints" >&2
  exit 1
fi

echo ">>> Qwen2.5 target-generated vLLM eval sweep finished"
