#!/usr/bin/env bash
# scaffold.sh — create the project tree and empty placeholder files.
# Idempotent: existing files are left untouched (uses `mkdir -p` and a guarded touch).
# Does NOT write any module code; that comes in follow-up commits.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# ---- helpers -----------------------------------------------------------------
mkd() { mkdir -p "$1"; }

# create empty file only if it does not exist
mkf() {
  local f="$1"
  if [[ ! -e "$f" ]]; then
    mkdir -p "$(dirname "$f")"
    : > "$f"
  fi
}

# write content only if file does not exist (preserves any user edits)
write_if_missing() {
  local f="$1"; shift
  if [[ ! -e "$f" ]]; then
    mkdir -p "$(dirname "$f")"
    cat > "$f"
  else
    # discard heredoc input
    cat > /dev/null
  fi
}

# ---- top-level files ---------------------------------------------------------
write_if_missing pyproject.toml <<'TOML'
[project]
name = "kdsd"
version = "0.0.0"
description = "Knowledge distillation for speculative decoding (CS-552 MNLP project)."
readme = "README.md"
requires-python = ">=3.11,<3.13"
dependencies = [
  "torch>=2.4",
  "transformers>=4.45",
  "accelerate>=0.34",
  "datasets>=2.20",
  "peft>=0.12",
  "hydra-core>=1.3",
  "omegaconf>=2.3",
  "wandb>=0.17",
  "jsonlines>=4.0",
  "rich>=13.7",
  "numpy>=1.26",
  "tqdm>=4.66",
  "safetensors>=0.4",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.0",
  "ruff>=0.6",
  "pre-commit>=3.7",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/kdsd"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.pytest.ini_options]
testpaths = ["tests"]
TOML

write_if_missing .python-version <<'PY'
3.11
PY

write_if_missing .gitignore <<'GIT'
# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
.eggs/

# uv
.uv/

# Project artefacts (never commit)
data/
checkpoints/
results/
wandb/
outputs/
multirun/
report/

# HF / model caches
.cache/
*.safetensors
*.bin

# Editor / OS
.DS_Store
.idea/
.vscode/
*.swp
GIT

write_if_missing README.md <<'MD'
# CS-552 MNLP Project

Knowledge distillation for speculative decoding.

See [CLAUDE.md](CLAUDE.md) for the full design, module contracts, Hydra config
layout, pipeline, and SLURM workflow.

## Quickstart

```bash
uv sync
uv run pytest -q          # local unit tests only
```

All training / evaluation runs go through SLURM — see CLAUDE.md ¶ Pipeline.
MD

# ---- configs (empty placeholders) -------------------------------------------
mkf configs/config.yaml
mkf configs/model/qwen25.yaml

for f in ultrachat_10k ultrachat_25k ultrachat_50k ultrachat_50k_target_gen alpaca_50k eval_holdout; do
  mkf "configs/data/${f}.yaml"
done

for f in ce fkl rkl jsd; do
  mkf "configs/loss/${f}.yaml"
done

mkf configs/train/default.yaml
mkf configs/eval/default.yaml
mkf configs/eval/runtime_sweep.yaml
mkf configs/runtime/default.yaml
mkf configs/runtime/sweep.yaml
mkf configs/benchmark/default.yaml
mkf configs/benchmark/full.yaml

# ---- src/kdsd (package) -----------------------------------------------------
mkf src/kdsd/__init__.py

mkf src/kdsd/models/__init__.py
mkf src/kdsd/models/loader.py
mkf src/kdsd/models/kd_pair.py

mkf src/kdsd/data/__init__.py
for f in download process target_generate logit_cache dataset; do
  mkf "src/kdsd/data/${f}.py"
done

mkf src/kdsd/losses/__init__.py
for f in ce fkl rkl jsd combined; do
  mkf "src/kdsd/losses/${f}.py"
done

mkf src/kdsd/sd/__init__.py
for f in hf_assisted custom_loop instrument; do
  mkf "src/kdsd/sd/${f}.py"
done

mkf src/kdsd/eval/__init__.py
mkf src/kdsd/eval/runner.py
mkf src/kdsd/eval/metrics.py
mkf src/kdsd/eval/benchmarks/__init__.py
for f in base judge_gpt4 mt_bench registry; do
  mkf "src/kdsd/eval/benchmarks/${f}.py"
done

mkf src/kdsd/train/__init__.py
mkf src/kdsd/train/trainer.py
mkf src/kdsd/train/callbacks.py

mkf src/kdsd/utils/__init__.py
for f in io timing logging; do
  mkf "src/kdsd/utils/${f}.py"
done

# ---- scripts ----------------------------------------------------------------
for f in prepare_data generate_target_responses cache_target_logits train evaluate_sd runtime_sweep aggregate_results; do
  mkf "scripts/${f}.py"
done

for f in train.slurm eval.slurm target_gen.slurm runtime_sweep.slurm submit_array.sh; do
  mkf "scripts/slurm/${f}"
done

# ---- tests ------------------------------------------------------------------
mkf tests/__init__.py
for f in test_losses test_data test_eval_schema test_config_compose; do
  mkf "tests/${f}.py"
done

# ---- notebooks (empty dir, kept by .gitkeep) --------------------------------
mkd notebooks
mkf notebooks/.gitkeep

# ---- gitignored artefact dirs (kept around with .gitkeep so layout is visible) ----
# These will be ignored by git but the directories exist locally.
for d in data checkpoints results report; do
  mkd "$d"
done

echo "Scaffold complete. Tree:"
if command -v tree >/dev/null 2>&1; then
  tree -a -I '.git|__pycache__|.venv' -L 4
else
  find . -path ./.git -prune -o -print | sed -e "s|^\./||" | sort
fi
