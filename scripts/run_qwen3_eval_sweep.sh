#!/bin/bash
# Evaluate Qwen3 pretrained + checkpoint drafts for an existing loss sweep.
# This script runs inside the RunAI pod from the checked-out repo.

set -euo pipefail

DRAFT_SIZE="${DRAFT_SIZE:-0.6b}"  # 0.6b or 1.7b
DATA="${DATA:-ultrachat_50k}"
SEED="${SEED:-42}"
TARGET_ID="${TARGET_ID:-}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

EVAL_PRETRAINED_BASELINE="${EVAL_PRETRAINED_BASELINE:-true}"
EVAL_PROMPTS_JSONL="${EVAL_PROMPTS_JSONL:-/scratch/cs552-data/processed/${DATA}/eval.jsonl}"
EVAL_PROMPTS_LIMIT="${EVAL_PROMPTS_LIMIT:-50}"
EVAL_GAMMA="${EVAL_GAMMA:-4}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-256}"
EVAL_WARMUP="${EVAL_WARMUP:-1}"
EVAL_REPEATS="${EVAL_REPEATS:-3}"
EVAL_BACKEND="${EVAL_BACKEND:-manual}"
EVAL_MODE="${EVAL_MODE:-sampling}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-1.0}"
EVAL_TOP_P="${EVAL_TOP_P:-0.9}"
EVAL_REPORT_TO_WANDB="${EVAL_REPORT_TO_WANDB:-true}"
EVAL_REPORT_CACHED_TO_WANDB="${EVAL_REPORT_CACHED_TO_WANDB:-true}"
FORCE_RERUN="${FORCE_RERUN:-false}"

RESULTS_ROOT="${RESULTS_ROOT:-/scratch/cs552-results}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints}"
HYDRA_ROOT="${HYDRA_ROOT:-outputs/qwen3-eval-sweep}"
PRETRAINED_CHECKPOINT_ROOT="${PRETRAINED_CHECKPOINT_ROOT:-${CHECKPOINT_ROOT}/pretrained}"

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
    ;;
  *)
    echo "ERROR: DRAFT_SIZE must be 0.6b or 1.7b, got '${DRAFT_SIZE}'." >&2
    exit 1
    ;;
esac

EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_${DRAFT_TAG}_${DATA}_seed${SEED}}"
WANDB_GROUP="${WANDB_GROUP:-${EXPERIMENT_NAME}}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-qwen3_${DRAFT_TAG}}"

target_override=()
if [[ -n "${TARGET_ID}" ]]; then
  target_override=("model.target=${TARGET_ID}")
fi

is_true() {
  [[ "$1" == "true" || "$1" == "1" || "$1" == "yes" || "$1" == "y" ]]
}

summary_path() {
  printf '%s/%s/eval_summary.json\n' "${RESULTS_ROOT}" "$1"
}

checkpoint_done() {
  local model_dir="$1"
  [[ -f "${model_dir}/config.json" ]] && {
    compgen -G "${model_dir}/*.safetensors" >/dev/null || \
    compgen -G "${model_dir}/pytorch_model*.bin" >/dev/null || \
    [[ -f "${model_dir}/model.safetensors.index.json" ]] || \
    [[ -f "${model_dir}/pytorch_model.bin.index.json" ]]
  }
}

report_cached_eval_to_wandb() {
  local eval_run="$1"
  local summary="$2"

  if ! is_true "${EVAL_REPORT_TO_WANDB}" || ! is_true "${EVAL_REPORT_CACHED_TO_WANDB}"; then
    return
  fi

  echo ">>> Reporting cached ${eval_run} metrics to W&B"
  WANDB_GROUP="${WANDB_GROUP}" \
  WANDB_JOB_TYPE="eval" \
  python - "${eval_run}" "${summary}" <<'PY'
import json
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf

from scripts import evaluate_sd


class Log:
    @staticmethod
    def warning(*args, **kwargs):
        print("WARNING:", *args)


run_name = sys.argv[1]
summary_path = Path(sys.argv[2])
with summary_path.open("r", encoding="utf-8") as fh:
    summary = json.load(fh)

draft = summary.get("draft")
checkpoint_meta_path, checkpoint_meta = evaluate_sd._checkpoint_metadata_from_draft(draft)
cfg = OmegaConf.create(
    {
        "run_name": run_name,
        "draft": draft,
        "wandb": {
            "project": os.environ.get("WANDB_PROJECT", "cs552-kdsd"),
            "entity": os.environ.get("WANDB_ENTITY", ""),
            "dir": os.environ.get("WANDB_DIR", "wandb"),
            "mode": os.environ.get("WANDB_MODE", "online"),
            "resume": "allow",
        },
    }
)
evaluate_sd._report_eval_to_wandb(
    cfg=cfg,
    summary=summary,
    out_dir=summary_path.parent,
    checkpoint_meta=checkpoint_meta,
    checkpoint_meta_path=checkpoint_meta_path,
    log=Log(),
)
PY
}

run_eval() {
  local eval_run="$1"
  local draft="$2"
  local summary

  summary="$(summary_path "${eval_run}")"

  if ! is_true "${FORCE_RERUN}" && [[ -f "${summary}" ]]; then
    echo ">>> Skipping ${eval_run}; cached summary exists at ${summary}"
    report_cached_eval_to_wandb "${eval_run}" "${summary}"
    eval_result_runs+=("${eval_run}")
    return
  fi

  echo ">>> Evaluating ${eval_run}"
  echo ">>> draft=${draft}"
  WANDB_GROUP="${WANDB_GROUP}" \
  WANDB_JOB_TYPE="eval" \
  python scripts/evaluate_sd.py \
    model=qwen3 "data=${DATA}" "${target_override[@]}" \
    "draft=${draft}" \
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
    "wandb.enabled=${EVAL_REPORT_TO_WANDB}" \
    "results_dir=${RESULTS_ROOT}/${eval_run}" \
    "hydra.run.dir=${HYDRA_ROOT}/${eval_run}" \
    "run_name=${eval_run}"
  eval_result_runs+=("${eval_run}")
}

export PYTORCH_CUDA_ALLOC_CONF
mkdir -p "${RESULTS_ROOT}" "${HYDRA_ROOT}" "${PRETRAINED_CHECKPOINT_ROOT}"

echo ">>> Qwen3 eval sweep: ${EXPERIMENT_NAME}"
echo ">>> W&B group: ${WANDB_GROUP}"
echo ">>> run name prefix: ${RUN_NAME_PREFIX}"
echo ">>> losses/checkpoints: ${LOSSES}"
echo ">>> checkpoint root: ${CHECKPOINT_ROOT}"
echo ">>> results root: ${RESULTS_ROOT}"
echo ">>> pretrained checkpoint root: ${PRETRAINED_CHECKPOINT_ROOT}"
echo ">>> eval pretrained baseline: ${EVAL_PRETRAINED_BASELINE}"
echo ">>> eval prompts: ${EVAL_PROMPTS_JSONL}"
echo ">>> eval prompts limit: ${EVAL_PROMPTS_LIMIT}"
echo ">>> eval gamma/max_new: ${EVAL_GAMMA}/${EVAL_MAX_NEW_TOKENS}"
echo ">>> eval warmup/repeats: ${EVAL_WARMUP}/${EVAL_REPEATS}"
echo ">>> eval backend: ${EVAL_BACKEND}"
echo ">>> eval mode/temp/top_p: ${EVAL_MODE}/${EVAL_TEMPERATURE}/${EVAL_TOP_P}"
echo ">>> eval report to W&B: ${EVAL_REPORT_TO_WANDB}"
echo ">>> report cached evals to W&B: ${EVAL_REPORT_CACHED_TO_WANDB}"
echo ">>> force rerun: ${FORCE_RERUN}"

eval_result_runs=()

if is_true "${EVAL_PRETRAINED_BASELINE}"; then
  baseline_eval_run="${RUN_NAME_PREFIX}_pretrain_${DATA}_seed${SEED}_eval_g${EVAL_GAMMA}_max${EVAL_MAX_NEW_TOKENS}"
  run_eval "${baseline_eval_run}" "${DRAFT_ID}"
fi

for loss in ${LOSSES}; do
  train_run="${RUN_NAME_PREFIX}_${loss}_${DATA}_seed${SEED}"
  model_dir="${CHECKPOINT_ROOT}/${train_run}/model"
  eval_run="${train_run}_eval_g${EVAL_GAMMA}_max${EVAL_MAX_NEW_TOKENS}"

  if ! checkpoint_done "${model_dir}"; then
    echo ">>> WARNING: skipping ${train_run}; no completed checkpoint at ${model_dir}" >&2
    continue
  fi

  run_eval "${eval_run}" "${model_dir}"
done

if [[ "${#eval_result_runs[@]}" -eq 0 ]]; then
  echo "ERROR: no eval runs were produced. Check LOSSES, checkpoint root, and EVAL_PRETRAINED_BASELINE." >&2
  exit 1
fi

echo ">>> Final cached SD evaluation summary"
python - "${RESULTS_ROOT}" "${eval_result_runs[@]}" <<'PY'
import json
import math
import sys
from pathlib import Path

results_root = Path(sys.argv[1])
runs = sys.argv[2:]
headers = (
    "run",
    "speedup",
    "accept",
    "avg_acc",
    "tok/s",
    "sd_s",
    "vanilla_s",
    "target_calls",
    "draft_calls",
)
rows = []

def fmt(value, pattern):
    if isinstance(value, (int, float)) and math.isfinite(value):
        return pattern % value
    return "nan"

for run in runs:
    path = results_root / run / "eval_summary.json"
    if not path.exists():
        rows.append((run, "missing", "", "", "", "", "", "", ""))
        continue
    with path.open("r", encoding="utf-8") as fh:
        summary = json.load(fh)
    engine = summary.get("engines", {}).get("hf", {})
    rows.append(
        (
            run,
            fmt(summary.get("speedup"), "%.3fx"),
            fmt(summary.get("acceptance_rate"), "%.3f"),
            fmt(summary.get("avg_accepted_tokens"), "%.2f"),
            fmt(summary.get("tokens_per_second"), "%.2f"),
            fmt(summary.get("sd_time_s"), "%.2f"),
            fmt(summary.get("vanilla_time_s"), "%.2f"),
            str(engine.get("target_calls", "")),
            str(engine.get("draft_calls", "")),
        )
    )

widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]
line = "  ".join(str(h).ljust(w) for h, w in zip(headers, widths))
print(line)
print("-" * len(line))
for row in rows:
    print("  ".join(str(x).ljust(w) for x, w in zip(row, widths)))
PY
