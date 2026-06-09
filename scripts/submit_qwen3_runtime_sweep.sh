#!/bin/bash
# Submit one Run:AI job that evaluates one target-generated JSD Qwen3 checkpoint
# across runtime parameters. Run this from the repo root on the laptop/cluster
# login machine where runai is configured.
#
# Common overrides:
#   CHECKPOINT_RUN=qwen3_8btarget_0p6b_tgen_jsd_ultrachat_50k_target_gen_seed42 ./scripts/submit_qwen3_runtime_sweep.sh
#   GAMMAS="1 2 4 6 8" TEMPERATURES="0.0 0.5 1.0 2.0" EVAL_MAX_NEW_TOKENS_VALUES="128 256 512" ./scripts/submit_qwen3_runtime_sweep.sh
#   DRAFT_PATH=/scratch/cs552-checkpoints/my_run/model ./scripts/submit_qwen3_runtime_sweep.sh

set -euo pipefail

# RunAI / repo checkout settings. These mirror rcp_support/submit_train.sh, but
# keep this submit payload small enough for RunAI's per-env-var value limit.
GASPAR="${GASPAR:-youyang}"
GROUP="${GROUP:-g67}"
WANDB_MODE="${WANDB_MODE:-online}"
REPO_URL="${REPO_URL:-https://github.com/ouyangyuchen/knowledge-distill-for-spec-decoding.git}"
REPO_BRANCH="${REPO_BRANCH:-codex/vllm-eval}"
REPO_DIR="${REPO_DIR:-/scratch/cs552-repos/cs552-kdsd-${GASPAR}-qwen3-runtime-sweep}"
WANDB_DIR="${WANDB_DIR:-/scratch/wandb}"
HYDRA_OUTPUTS_DIR="${HYDRA_OUTPUTS_DIR:-/scratch/cs552-hydra-outputs}"
NODE="${NODE:-a100-40g}"
PROJECT="${PROJECT:-course-cs-552-${GASPAR}}"
IMAGE="${IMAGE:-registry.rcp.epfl.ch/course-cs-552/base-vllm:v1}"
GPUS="${GPUS:-1}"

DRAFT_SIZE="${DRAFT_SIZE:-0.6b}"
DATA="${DATA:-ultrachat_50k_target_gen}"
BASE_DATA="${BASE_DATA:-${DATA%_target_gen}}"
SEED="${SEED:-42}"
TARGET_ID="${TARGET_ID:-Qwen/Qwen3-8B}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
KDSD_VENV="${KDSD_VENV:-/scratch/venvs/kdsd-vllm}"

GAMMAS="${GAMMAS:-1 2 4 6 8}"
TEMPERATURES="${TEMPERATURES:-0.0 0.5 1.0 1.5 2.0}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-256}"
EVAL_MAX_NEW_TOKENS_VALUES="${EVAL_MAX_NEW_TOKENS_VALUES:-128 256 512}"
EVAL_WARMUP="${EVAL_WARMUP:-1}"
EVAL_REPEATS="${EVAL_REPEATS:-3}"
EVAL_BACKEND="${EVAL_BACKEND:-vllm}"
EVAL_MODE="${EVAL_MODE:-auto}"
EVAL_TOP_P="${EVAL_TOP_P:-0.9}"
EVAL_RUN_VANILLA_BASELINE="${EVAL_RUN_VANILLA_BASELINE:-true}"
EVAL_REPORT_TO_WANDB="${EVAL_REPORT_TO_WANDB:-true}"
FORCE_RERUN="${FORCE_RERUN:-true}"

RESULTS_ROOT="${RESULTS_ROOT:-/scratch/cs552-results}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/scratch/cs552-checkpoints}"
HYDRA_ROOT="${HYDRA_ROOT:-outputs/qwen3-runtime-sweep}"
PRETRAINED_CHECKPOINT_ROOT="${PRETRAINED_CHECKPOINT_ROOT:-${CHECKPOINT_ROOT}/pretrained}"
EVAL_PROMPTS_JSONL="${EVAL_PROMPTS_JSONL:-/scratch/cs552-data/processed/${BASE_DATA}/eval.jsonl}"
EVAL_PROMPTS_LIMIT="${EVAL_PROMPTS_LIMIT:-50}"

if [[ "${GASPAR}" == "gaspar" || -z "${GASPAR}" ]]; then
  echo "ERROR: set GASPAR to your EPFL GASPAR username." >&2
  exit 1
fi

if [[ "${GROUP}" == "gXX" || -z "${GROUP}" ]]; then
  echo "ERROR: set GROUP to your team number, e.g. g67." >&2
  exit 1
fi

if [[ "${WANDB_MODE}" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
  echo "ERROR: set WANDB_API_KEY before submitting, or run with WANDB_MODE=offline." >&2
  exit 1
fi

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

SCRATCH_PVC="course-cs-552-scratch-${GROUP}"
SHARED_RO_PVC="course-cs-552-shared-ro"
SHARED_RW_PVC="course-cs-552-shared-rw"
JOB_NAME="cs552-${GASPAR}-${GROUP}-qwen3-runtime-$(date +%H%M%S)"

read -r -d '' BOOTSTRAP_COMMAND <<'BOOTSTRAP' || true
set -euo pipefail

mkdir -p /scratch/hf_cache /scratch/cs552-results "${WANDB_DIR}" "${CHECKPOINT_ROOT}" "${RESULTS_ROOT}" "${HYDRA_OUTPUTS_DIR}" "$(dirname "${REPO_DIR}")"

if command -v python3 >/dev/null 2>&1; then
  ln -sf "$(command -v python3)" /usr/local/bin/python 2>/dev/null || true
fi

export GIT_CONFIG_COUNT="${GIT_CONFIG_COUNT:-0}"
idx="${GIT_CONFIG_COUNT}"
export "GIT_CONFIG_KEY_${idx}=safe.directory"
export "GIT_CONFIG_VALUE_${idx}=${REPO_DIR}"
export GIT_CONFIG_COUNT="$((idx + 1))"

if [[ -e "${REPO_DIR}" && ! -d "${REPO_DIR}" ]]; then
  echo "ERROR: REPO_DIR exists but is not a directory: ${REPO_DIR}" >&2
  exit 1
fi

if [[ -d "${REPO_DIR}" && ! -d "${REPO_DIR}/.git" ]]; then
  if find "${REPO_DIR}" -mindepth 1 -maxdepth 1 | read -r _; then
    echo "ERROR: REPO_DIR exists, is not a git checkout, and is not empty: ${REPO_DIR}" >&2
    exit 1
  fi
fi

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  git clone --no-checkout --origin origin "${REPO_URL}" "${REPO_DIR}"
fi

git -C "${REPO_DIR}" remote set-url origin "${REPO_URL}"
git -C "${REPO_DIR}" fetch --prune --force origin "+refs/heads/${REPO_BRANCH}:refs/remotes/origin/${REPO_BRANCH}"
git -C "${REPO_DIR}" reset --hard HEAD || true
git -C "${REPO_DIR}" clean -ffdx || true
git -C "${REPO_DIR}" checkout --detach "origin/${REPO_BRANCH}"
git -C "${REPO_DIR}" branch -f "${REPO_BRANCH}" "origin/${REPO_BRANCH}"
git -C "${REPO_DIR}" checkout "${REPO_BRANCH}"
git -C "${REPO_DIR}" reset --hard "origin/${REPO_BRANCH}"
git -C "${REPO_DIR}" clean -ffdx

cd "${REPO_DIR}"
for pair in "checkpoints:${CHECKPOINT_ROOT}" "data:/scratch/cs552-data" "outputs:${HYDRA_OUTPUTS_DIR}" "wandb:${WANDB_DIR}"; do
  name="${pair%%:*}"
  target="${pair#*:}"
  if [[ -L "${name}" ]]; then
    rm "${name}"
  elif [[ -e "${name}" ]]; then
    echo "ERROR: ${REPO_DIR}/${name} exists and is not a symlink." >&2
    exit 1
  fi
  mkdir -p "${target}"
  ln -s "${target}" "${name}"
done

echo ">>> Checked out ${REPO_BRANCH} at $(git rev-parse HEAD)"
echo ">>> Running Qwen3 runtime sweep"
bash scripts/run_qwen3_runtime_sweep.sh
BOOTSTRAP

runai_args=(
  submit
  --name "${JOB_NAME}"
  -p "${PROJECT}"
  --image "${IMAGE}"
  --gpu "${GPUS}"
  --large-shm
  --node-pools "${NODE}"
  --working-dir /scratch
  --environment-variable "HF_HOME=/scratch/hf_cache"
  --environment-variable "HF_HUB_ENABLE_HF_TRANSFER=1"
  --environment-variable "WANDB_DIR=${WANDB_DIR}"
  --environment-variable "WANDB_MODE=${WANDB_MODE}"
  --environment-variable "WANDB_PROJECT=${WANDB_PROJECT:-cs552-kdsd}"
  --environment-variable "WANDB_NAME=${EXPERIMENT_NAME}"
  --environment-variable "WANDB_ENTITY=${WANDB_ENTITY:-}"
  --environment-variable "WANDB_API_KEY=${WANDB_API_KEY:-}"
  --environment-variable "REPO_URL=${REPO_URL}"
  --environment-variable "REPO_BRANCH=${REPO_BRANCH}"
  --environment-variable "REPO_DIR=${REPO_DIR}"
  --environment-variable "CHECKPOINT_ROOT=${CHECKPOINT_ROOT}"
  --environment-variable "RESULTS_ROOT=${RESULTS_ROOT}"
  --environment-variable "HYDRA_OUTPUTS_DIR=${HYDRA_OUTPUTS_DIR}"
  --environment-variable "DRAFT_SIZE=${DRAFT_SIZE}"
  --environment-variable "DATA=${DATA}"
  --environment-variable "BASE_DATA=${BASE_DATA}"
  --environment-variable "SEED=${SEED}"
  --environment-variable "TARGET_ID=${TARGET_ID}"
  --environment-variable "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
  --environment-variable "KDSD_VENV=${KDSD_VENV}"
  --environment-variable "GAMMAS=${GAMMAS}"
  --environment-variable "TEMPERATURES=${TEMPERATURES}"
  --environment-variable "EVAL_MAX_NEW_TOKENS=${EVAL_MAX_NEW_TOKENS}"
  --environment-variable "EVAL_MAX_NEW_TOKENS_VALUES=${EVAL_MAX_NEW_TOKENS_VALUES}"
  --environment-variable "EVAL_WARMUP=${EVAL_WARMUP}"
  --environment-variable "EVAL_REPEATS=${EVAL_REPEATS}"
  --environment-variable "EVAL_BACKEND=${EVAL_BACKEND}"
  --environment-variable "EVAL_MODE=${EVAL_MODE}"
  --environment-variable "EVAL_TOP_P=${EVAL_TOP_P}"
  --environment-variable "EVAL_RUN_VANILLA_BASELINE=${EVAL_RUN_VANILLA_BASELINE}"
  --environment-variable "EVAL_REPORT_TO_WANDB=${EVAL_REPORT_TO_WANDB}"
  --environment-variable "FORCE_RERUN=${FORCE_RERUN}"
  --environment-variable "HYDRA_ROOT=${HYDRA_ROOT}"
  --environment-variable "PRETRAINED_CHECKPOINT_ROOT=${PRETRAINED_CHECKPOINT_ROOT}"
  --environment-variable "EVAL_PROMPTS_JSONL=${EVAL_PROMPTS_JSONL}"
  --environment-variable "EVAL_PROMPTS_LIMIT=${EVAL_PROMPTS_LIMIT}"
  --environment-variable "CHECKPOINT_RUN=${CHECKPOINT_RUN}"
  --environment-variable "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
  --environment-variable "DRAFT_PATH=${DRAFT_PATH}"
  --environment-variable "EXPERIMENT_NAME=${EXPERIMENT_NAME}"
  --environment-variable "WANDB_GROUP=${WANDB_GROUP}"
  --environment-variable "SUMMARY_CSV=${SUMMARY_CSV}"
  --existing-pvc "claimname=${SCRATCH_PVC},path=/scratch"
  --existing-pvc "claimname=${SHARED_RO_PVC},path=/shared-ro"
  --existing-pvc "claimname=${SHARED_RW_PVC},path=/shared-rw"
  --command -- /bin/bash -lc "${BOOTSTRAP_COMMAND}"
)

echo ">>> Submitting Qwen3 runtime sweep job: ${EXPERIMENT_NAME}"
echo ">>> Job name: ${JOB_NAME}"
echo ">>> Fixed checkpoint: ${CHECKPOINT_RUN}"
echo ">>> Draft path inside pod: ${DRAFT_PATH}"
echo ">>> Results root inside pod: ${RESULTS_ROOT}"
echo ">>> Summary CSV inside pod: ${SUMMARY_CSV}"
echo ">>> Gammas: ${GAMMAS}"
echo ">>> Temperatures: ${TEMPERATURES}"
echo ">>> Max new token lengths: ${EVAL_MAX_NEW_TOKENS_VALUES}"
echo ">>> Eval backend/mode/top_p: ${EVAL_BACKEND}/${EVAL_MODE}/${EVAL_TOP_P}"
echo ">>> Python venv inside pod: ${KDSD_VENV}"
echo ">>> W&B enabled: ${EVAL_REPORT_TO_WANDB}"
runai "${runai_args[@]}"

cat <<EOF

>>> Runtime sweep job submitted: ${JOB_NAME}

Watch it start:    runai describe job ${JOB_NAME} -p ${PROJECT}
Stream logs:       runai logs -f ${JOB_NAME} -p ${PROJECT}
Stop the job:      runai delete job ${JOB_NAME} -p ${PROJECT}
EOF
