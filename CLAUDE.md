# Project Reference

This file is the living design + module-API reference for this repository. New
contributors and assistants should read it before adding code. It is intentionally
prescriptive: the pipeline shape and inter-module contracts must stay stable so that
ablations across **KD objective**, **training data**, and **runtime decoding** can be
combined without bespoke glue.

The project distills a small draft model so that `Qwen/Qwen2.5-0.5B-Instruct` can
serve as a high-acceptance speculative-decoding (SD) draft for
`Qwen/Qwen2.5-3B-Instruct`. The two models share a tokenizer (verified) and are
FP16/BF16 friendly.

---

## Stack & Environment

- **Python 3.11**, `torch>=2.4`, `transformers>=4.45` (Qwen2.5 support).
- **`uv`** is the only dependency manager. `pyproject.toml` + committed `uv.lock` are
  the single source of truth — do not introduce `requirements.txt`.
- Core deps: `torch`, `transformers>=4.45`, `accelerate`, `datasets`, `peft`
  (optional LoRA path), `hydra-core`, `omegaconf`, `wandb` (optional, gated by env
  var), `jsonlines`, `rich`, `numpy`, `tqdm`, `safetensors`.
  Dev deps: `pytest`, `ruff`, `pre-commit`.
- HF auth: set `HF_TOKEN` in the environment (or `huggingface-cli login`). Models
  cache under `~/.cache/huggingface` by default; override with `HF_HOME`.
- GPU target: 1× 40GB A100 on the EPFL RCP RunAI cluster (course cap),
  `bf16` preferred. The course image (`registry.rcp.epfl.ch/course-cs-552/base-vllm:v1`)
  ships torch 2.8 + cu128, transformers 4.57, vLLM 0.11, and JupyterLab 4.5.
  See `rcp_support/README.md` for cluster setup, submission, and storage layout.

Bootstrap:

```bash
uv sync
uv run python -c "import torch; print(torch.cuda.is_available())"
```

**Local dev policy.** Mac/Windows hosts are for code editing and **unit tests only**
(`uv run pytest -q`). No model inference, training, or evaluation runs locally — all
of those go through RunAI on the EPFL RCP cluster. See `rcp_support/README.md` for
cluster setup, and `notebooks/run_eval_pipeline.ipynb` for the interactive eval
driver.

---

## Directory Structure

The internal package is named `kdsd` ("knowledge distillation for speculative
decoding") — descriptive and neutral.

```
cs552-mnlp-project/
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
├── README.md
├── CLAUDE.md
├── configs/                        # Hydra configs (composable)
│   ├── config.yaml
│   ├── model/
│   │   └── qwen25.yaml
│   ├── data/
│   │   ├── ultrachat_10k.yaml
│   │   ├── ultrachat_25k.yaml
│   │   ├── ultrachat_50k.yaml
│   │   ├── ultrachat_50k_target_gen.yaml
│   │   ├── alpaca_50k.yaml
│   │   └── eval_holdout.yaml
│   ├── loss/
│   │   ├── ce.yaml
│   │   ├── fkl.yaml
│   │   ├── rkl.yaml
│   │   └── jsd.yaml
│   ├── train/
│   │   └── default.yaml
│   ├── eval/
│   │   ├── default.yaml
│   │   └── runtime_sweep.yaml
│   ├── runtime/
│   │   ├── default.yaml
│   │   └── sweep.yaml
│   ├── benchmark/
│   │   ├── default.yaml
│   │   └── full.yaml
│   └── speedup/
│       ├── hf.yaml                # HF custom impl is sole timing source
│       └── vllm.yaml              # default — vLLM speedup pass (subprocess-isolated)
├── src/kdsd/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── kd_pair.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py
│   │   ├── process.py
│   │   ├── target_generate.py
│   │   ├── logit_cache.py
│   │   └── dataset.py
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── ce.py
│   │   ├── fkl.py
│   │   ├── rkl.py
│   │   ├── jsd.py
│   │   └── combined.py
│   ├── sd/
│   │   ├── __init__.py
│   │   ├── hf_assisted.py
│   │   ├── custom_loop.py
│   │   ├── instrument.py
│   │   └── vllm_runner.py         # subprocess-isolated vLLM speedup pass
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── runner.py
│   │   ├── metrics.py
│   │   └── benchmarks/
│   │       ├── __init__.py
│   │       ├── base.py
│   │       ├── judge_gpt4.py
│   │       ├── mt_bench.py
│   │       └── registry.py
│   ├── train/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   └── callbacks.py
│   └── utils/
│       ├── __init__.py
│       ├── io.py
│       ├── timing.py
│       └── logging.py
├── scripts/
│   ├── prepare_data.py
│   ├── generate_target_responses.py
│   ├── cache_target_logits.py
│   ├── train.py
│   ├── evaluate_sd.py            # one phase per invocation (engine=hf|vllm)
│   ├── run_eval_pipeline.py      # drives both phases as separate processes
│   ├── hf_sd_speedup.py          # standalone HF SD speedup probe
│   ├── vllm_sd_speedup.py        # standalone vLLM SD speedup probe
│   ├── runtime_sweep.py
│   └── aggregate_results.py
├── tests/
│   ├── __init__.py
│   ├── test_losses.py
│   ├── test_data.py
│   ├── test_eval_schema.py
│   └── test_config_compose.py
├── notebooks/                    # RunAI deliverable layout
│   ├── submit.sh                 # team-wide Jupyter launcher (copy of rcp_support/submit.sh)
│   ├── run_eval_pipeline.ipynb   # interactive driver for scripts/evaluate_sd.py
│   └── <first>_<last>_<sciper>.ipynb  # one per teammate (deliverable)
├── rcp_support/                  # upstream RunAI / RCP starter (do not modify)
│   ├── README.md                 # canonical cluster guide
│   ├── submit.sh                 # interactive Jupyter starter
│   ├── submit_train.sh           # non-interactive training-job starter
│   ├── Dockerfile                # optional custom-image template
│   └── build.sh                  # optional Harbor push helper
└── (gitignored) data/, checkpoints/, results/, wandb/
```

---

## Module Contracts

These are the only interfaces other modules may depend on. Internal helpers in each
subpackage stay private.

### 1. Checkpoint contract

Every training run writes:

```
checkpoints/<run_name>/
  config.yaml   # the resolved Hydra config (OmegaConf.save)
  model/        # safetensors + tokenizer (HF .save_pretrained format)
  meta.json     # {git_sha, train_loss_final, steps, dataset_id, draft_init}
```

Consequence: `evaluate_sd.py draft=checkpoints/<run_name>/model` works for **every**
checkpoint, with no special-casing per training recipe.

### 2. Eval contract — `scripts/evaluate_sd.py`

CLI for SD evaluation. Runs **one phase per invocation** (selected via the
top-level `engine` Hydra field) so vLLM never shares a CUDA context with HF
— vLLM cannot release one cleanly, and the second engine load OOMs in the
same process. `scripts/run_eval_pipeline.py` is the orchestrator that drives
both phases in sequence with the same `run_name`. Always writes:

```
results/<run_name>/
  eval_summary.json   # the schema below — used by aggregate_results.py
  generations.jsonl   # per-prompt: prompt, generation, accepted_lens[], times{}
  timing.json         # raw per-prompt timings, n_warmup, n_repeats
  config.yaml         # resolved config snapshot
```

`eval_summary.json` schema (frozen top-level keys, plus an optional `engines`
block). `quality_score` is a **dict** so a single config can be scored against
multiple benchmarks in one run:

```json
{
  "model": "kd_jsd_50k_target_gen",
  "target": "Qwen/Qwen2.5-3B-Instruct",
  "draft":  "checkpoints/kd_jsd_50k_target_gen/model",
  "acceptance_rate": 0.52,
  "avg_accepted_tokens": 2.1,
  "vanilla_time_s": 123.4,
  "sd_time_s": 87.2,
  "speedup": 1.41,
  "tokens_per_second": 18.6,
  "quality_score": {
    "gpt4_judge_vs_target": 7.1,
    "mt_bench": 6.8,
    "exact_match_vs_target": 0.42
  },
  "decoding": {"mode": "greedy", "max_new_tokens": 256, "num_assistant_tokens": 4},
  "n_prompts": 200, "n_warmup": 2, "n_repeats": 3,
  "engines": {
    "hf":   {"sd_time_s": 95.0, "vanilla_time_s": null, "speedup": 1.0,
             "tokens_per_second": 16.9, "acceptance_rate": 0.52,
             "avg_accepted_tokens": 2.1, "n_outer_steps": 1240,
             "target_calls": 1240, "draft_calls": 4960,
             "draft_forward_s": 12.1, "target_forward_s": 71.4,
             "batched": false},
    "vllm": {"sd_time_s": 87.2, "vanilla_time_s": 123.4, "speedup": 1.41,
             "tokens_per_second": 18.6, "vanilla_tokens_per_second": 13.2,
             "vanilla_tokens": 1632, "sd_tokens": 1622,
             "repeats": 3, "n_warmup": 1, "batched": true,
             "ok": true, "vanilla_ok": true, "sd_ok": true, "error": null,
             "spec_stats": {"draft_acceptance_rate": 0.55,
                            "system_efficiency": 0.71}}
  }
}
```

`"quality_score": {}` is valid (skip benchmarks). Validation lives in
`tests/test_eval_schema.py` and `src/kdsd/utils/io.py`. `aggregate_results.py` reads
`quality_score.<key>`, so adding a benchmark never requires changing the aggregator.
`engines` is **optional** (not in `REQUIRED_SUMMARY_KEYS`) so older summary files
still validate; when present, each engine's sub-block must be a dict.

**Two phases** (selected via the top-level `engine` field; the `speedup/`
config group still selects which vLLM kwargs apply):

- `engine=hf` — load HF target (and optional draft), run the instrumented
  loop in `src/kdsd/sd/instrument.py`, write the full eval_summary +
  generations + timing artefacts. This phase is the sole source of trusted
  per-step metrics: `acceptance_rate`, `avg_accepted_tokens`, per-prompt
  `accepted_lens`. The HF vanilla baseline is **skipped** when the resolved
  speedup config is `vllm` (vanilla timing will come from the vLLM phase
  instead); otherwise it's measured in the same job so `speedup` is anchored
  to fresh `vanilla_time_s`. The HF loop has unavoidable D→H syncs
  (accept-mask transfer, EOS scan, rejection-resample `s>0` check) so its
  `speedup` is conservative.
- `engine=vllm` — skip HF model loading entirely. Re-tokenize the prompts in
  the parent (CPU-only) and run vanilla + SD passes through vLLM in spawn
  subprocesses, then read the prior `eval_summary.json`, merge a `vllm`
  block under `engines.vllm`, and overwrite the top-level `sd_time_s` /
  `vanilla_time_s` / `speedup` / `tokens_per_second` when the SD pass
  succeeded. Errors out if no prior summary exists for `run_name`. Both
  engines see byte-identical model input: prompts are chat-templated and
  tokenized with `add_special_tokens=False` and pushed to vLLM via the
  `TokensPrompt` dict form. `skip_tokenizer_init` is **False** on the LLM
  (vLLM's spec-decode worker requires the tokenizer for stop-token logic,
  and EOS detection on output requires it too); the input is still bypassed
  because `TokensPrompt` skips the tokenizer for encoding.

Run both phases together:

```bash
uv run python scripts/run_eval_pipeline.py \
    run_name=spec_smoke draft=Qwen/Qwen2.5-0.5B-Instruct \
    prompts.jsonl=data/processed/eval.jsonl prompts.limit=20
```

Or run a single phase directly:

```bash
uv run python scripts/evaluate_sd.py engine=hf  run_name=spec_smoke ...
uv run python scripts/evaluate_sd.py engine=vllm run_name=spec_smoke ...
```

`engines.{hf,vllm}` preserves both engines' raw numbers regardless of which
overwrites the top-level fields.

### 3. Loss contract — `src/kdsd/losses/combined.py`

```python
def kd_loss(
    student_logits: Tensor,         # [B, T, V]
    teacher_logits: Tensor | None,  # None when using cached top-k path
    teacher_topk_ids: Tensor | None,
    teacher_topk_logp: Tensor | None,
    labels: Tensor,                 # -100 on prompt tokens
    *, kind: Literal["fkl","rkl","jsd","ce"],
    temperature: float = 1.0,
    alpha: float = 0.5,             # weight on KD term; (1-alpha) * CE
) -> dict[str, Tensor]:             # {"loss", "ce", "kd"}
```

Other modules import only `kd_loss`.

### 4. SD instrumentation contract — `src/kdsd/sd/instrument.py`

```python
@dataclass
class SDStats:
    accepted_lens: list[int]    # per generation step
    target_calls: int
    draft_calls: int
    draft_forward_s: float
    target_forward_s: float

def speculative_generate(
    target, draft, input_ids, *, gamma: int, max_new: int, ...
) -> tuple[Tensor, SDStats]
```

This is the source of truth for acceptance-rate / runtime-profile metrics. The
GPU-only forward times use CUDA events so the caller's wall-clock isn't
contaminated by per-forward sync overhead.

### 4b. SD speedup contract — `src/kdsd/sd/vllm_runner.py`

```python
@dataclass
class VllmEngineResult:
    ok: bool
    elapsed_s: float
    tokens: int
    spec_stats: VllmSpecStats | None    # only set on the SD engine result
    error: str | None

@dataclass
class VllmSpeedupResult:
    vanilla: VllmEngineResult
    sd: VllmEngineResult
    repeats: int
    n_warmup: int
    # properties: vanilla_tokens_per_second, sd_tokens_per_second, speedup

def run_vllm_speedup(
    *, prompt_token_ids: list[list[int]],
    target_id: str, draft_id: str | None,
    gamma: int, max_new_tokens: int,
    mode: str, temperature: float, top_p: float, seed: int,
    vllm_cfg: dict,                    # resolved configs/speedup/vllm.yaml
) -> VllmSpeedupResult
```

The runner spawns one subprocess per engine (vanilla, SD) — vLLM can't free
CUDA state in-process, so a single Python process can host at most one engine
at a time. `prompt_token_ids` must be tokenized in the parent with the same
chat template + `add_special_tokens=False` settings as
`runner._generate_one`. The runner also clamps the requested
`gpu_memory_utilization` to actually-free GPU memory at subprocess start so
the same config works on shared and dedicated GPUs.
`scripts/vllm_sd_speedup.py` is a thin CLI over the same module for
standalone use; `scripts/evaluate_sd.py engine=vllm` reuses
`run_vllm_pass` + `merge_vllm_into_summary` from `kdsd.eval.runner`.

### 5. Data contract — processed split format

Every processed/target-generated split is a `.jsonl` with one record per row:

```json
{"id": "...", "prompt_text": "...", "response_text": "...", "source": "ultrachat|target"}
```

KD training re-tokenizes and masks at load time; this keeps text-level data
inspectable and decouples the on-disk format from the tokenizer in use.

### 6. Benchmark contract — `src/kdsd/eval/benchmarks/base.py`

```python
class Benchmark(ABC):
    name: str   # key in quality_score dict
    @abstractmethod
    def score(
        self, generations: list[dict], target_generations: list[dict] | None
    ) -> float: ...
```

`registry.py` maps name → class. The `benchmark/*.yaml` Hydra group lists which
benchmarks to run; missing API keys (e.g. for the GPT-4 judge) skip that benchmark
with a warning rather than failing the eval.

---

## Hydra Config Layout

`configs/config.yaml` is the top-level defaults list reused by every script:

```yaml
defaults:
  - model: qwen25
  - data: ultrachat_50k
  - loss: fkl
  - train: default
  - eval: default
  - runtime: default
  - benchmark: default
  - speedup: vllm
  - _self_

run_name: ${loss}_${data}_seed${seed}
seed: 42
engine: hf                  # which eval phase to run: hf | vllm
output_dir: checkpoints/${run_name}
results_dir: results/${run_name}
hf_cache: ${oc.env:HF_HOME,~/.cache/huggingface}
```

Every ablation is one CLI flip. All commands run **inside the RunAI pod**
— either from a Jupyter terminal, `runai bash <job>`, or as the
`TRAIN_COMMAND` of `rcp_support/submit_train.sh` for unattended runs. From
your laptop, launch the pod with `notebooks/submit.sh` (see `rcp_support/`).

| Ablation              | Command                                                                 |
|-----------------------|-------------------------------------------------------------------------|
| KD-objective sweep    | `python scripts/train.py -m loss=fkl,rkl,jsd`                            |
| Data-scale sweep      | `python scripts/train.py -m data=ultrachat_10k,ultrachat_25k,ultrachat_50k` |
| Response source       | `python scripts/train.py data=ultrachat_50k_target_gen`                  |
| Runtime sweep         | `python scripts/runtime_sweep.py runtime=sweep draft=checkpoints/<best>/model` |
| Multi-benchmark eval  | `python scripts/evaluate_sd.py benchmark=full draft=...`                 |
| HF-only timing (no vLLM) | `python scripts/evaluate_sd.py engine=hf speedup=hf draft=...`        |
| Two-phase eval (HF + vLLM) | `python scripts/run_eval_pipeline.py draft=...` (or `notebooks/run_eval_pipeline.ipynb`) |

---

## Pipeline — End-to-End

### Step 0: Bootstrap

```bash
uv sync
uv run huggingface-cli login   # store HF_TOKEN
```

### Step 1: Data

```bash
uv run python scripts/prepare_data.py data=ultrachat_50k
uv run python scripts/generate_target_responses.py data=ultrachat_50k    # optional arm
uv run python scripts/cache_target_logits.py data=ultrachat_50k topk=64
```

Writes:

- `data/processed/ultrachat_50k/{train,val,eval}.jsonl`
- `data/target_generated/ultrachat_50k/{train,val}.jsonl`
- `data/logit_cache/ultrachat_50k_topk64/{shard_*.npz, meta.json}`

### Step 2: Train

```bash
uv run python scripts/train.py loss=jsd data=ultrachat_50k_target_gen \
  train.steps=4000 train.alpha=0.5 train.temperature=1.0 \
  run_name=kd_jsd_50k_targetgen_a0.5
```

Writes `checkpoints/kd_jsd_50k_targetgen_a0.5/{model,config.yaml,meta.json}`.

### Step 3: Evaluate

The eval is split into two phases that must run in **separate processes**
(vLLM cannot share a CUDA context with HF). On RunAI, the **primary flow is
the notebook** `notebooks/run_eval_pipeline.ipynb`, which mirrors the
script's subprocess split and reads the merged `eval_summary.json` inline.

Headless / unattended use (e.g. `runai bash` or
`rcp_support/submit_train.sh`) keeps the equivalent CLI:

```bash
uv run python scripts/run_eval_pipeline.py \
  draft=checkpoints/kd_jsd_50k_targetgen_a0.5/model \
  eval=default benchmark=default
```

Either path runs `evaluate_sd.py engine=hf` (instrumented HF loop →
produces `acceptance_rate`, `accepted_lens`, generations, an HF-side
speedup) followed by `evaluate_sd.py engine=vllm` (vLLM vanilla + SD passes
in spawn subprocesses → overwrites top-level `sd_time_s` /
`vanilla_time_s` / `speedup` / `tokens_per_second` in the same
`eval_summary.json` and adds an `engines.vllm` block). End artefacts:
`results/kd_jsd_50k_targetgen_a0.5/{eval_summary.json,generations.jsonl,timing.json}`.

If vLLM is not viable (mismatched draft/target vocab without `ngram=true`,
no CUDA, debugging the custom loop), run only the HF phase and tell it to
keep HF as the timing source:

```bash
uv run python scripts/evaluate_sd.py engine=hf speedup=hf \
  draft=checkpoints/kd_jsd_50k_targetgen_a0.5/model
```

Top-level timing fields then come solely from `src/kdsd/sd/instrument.py`,
including a fresh vanilla baseline measured in the same job.

Skip individual phases via the orchestrator: `--skip-hf` (uses an existing
`eval_summary.json` and only re-runs vLLM) or `--skip-vllm` (HF metrics
only). The notebook supports the same by simply not executing one of the
two phase cells.

### Step 4: Runtime sweep

```bash
uv run python scripts/runtime_sweep.py \
  draft=checkpoints/<best>/model runtime=sweep
```

Iterates `gamma ∈ {1,2,4,6,8}` × `max_new ∈ {128,256}`, writing one
`results/<best>__gamma{γ}_max{n}/eval_summary.json` per cell.

### Step 5: Aggregate

```bash
uv run python scripts/aggregate_results.py results/ -o report/attribution_table.md
```

Joins every `eval_summary.json` into a staged attribution table — vanilla →
pretrained-draft SD → KD-adapted → data-adapted → runtime-tuned — with one column
per `quality_score.<benchmark>`.

---

## Cluster: RunAI / RCP

The project runs on the EPFL RCP RunAI cluster. `rcp_support/README.md` is
the canonical guide (one-time setup, port-forwarding, GPU etiquette,
storage). Two submission paths:

**Interactive Jupyter (primary, also the deliverable launcher).** From the
laptop:

```bash
./notebooks/submit.sh           # default suffix "lab"
./notebooks/submit.sh exp1      # custom job-name suffix
runai port-forward <job-name> --port 8888:8888
# open http://localhost:8888  (token: cs552)
```

This is a copy of `rcp_support/submit.sh` placed at the path the rcp_support
README mandates for the deliverable. It mounts `/scratch` (group PVC),
`/shared-ro`, `/shared-rw`, sets `HF_HOME=/scratch/hf_cache` and
`HF_HUB_ENABLE_HF_TRANSFER=1`, and starts JupyterLab in `/scratch`. Edit
`GASPAR` for your own runs and `GROUP` for your team — the
submitted file may keep `GASPAR="gaspar"` (TAs replace), but `GROUP` must be
correct because it selects your team's scratch PVC.

**Non-interactive training job.** For long runs that should execute a command
and exit, use `rcp_support/submit_train.sh` (do **not** submit it as a
deliverable). Set `TRAIN_COMMAND`, e.g. for an unattended eval:

```bash
TRAIN_COMMAND='cd /scratch/<repo> && uv run python scripts/run_eval_pipeline.py \
  run_name=kd_jsd_50k draft=checkpoints/kd_jsd_50k/model'
./rcp_support/submit_train.sh
```

Training jobs are lower priority than interactive jobs and can be preempted —
write checkpoints to `/scratch` and resume from them.

**Storage contract.** Code lives in the git repo (cloned under `/scratch/`
inside the pod). HF cache and wandb logs live under `/scratch/hf_cache`
and `/scratch/wandb`. Deliverable notebooks must run from a clean clone of
the repo plus the course/group PVCs — no dependence on personal home or
ad-hoc files in `/scratch`. Anything in `/scratch` is wiped end of July 2026.

---

## File Build Order

When implementing the prototype, build in this order so downstream modules always
have a stable contract to call:

1. `pyproject.toml`, `.python-version`, `.gitignore`, `README.md`, `CLAUDE.md`.
2. `src/kdsd/utils/{io,timing,logging}.py` — needed by everything.
3. `src/kdsd/models/{loader,kd_pair}.py` — load Qwen target/draft pair.
4. `src/kdsd/sd/instrument.py` — custom speculative-decoding loop with rejection
   sampling and KV-cache (`DynamicCache`) management. We implement this
   directly rather than going through HF's `model.generate(assistant_model=draft)`
   path because the frozen `eval_summary.json` schema requires per-step
   `accepted_lens`, which HF's assisted decoding does not surface cleanly.
   `hf_assisted.py` / `custom_loop.py` remain as optional alternative entrypoints.
5. `src/kdsd/eval/{runner,metrics}.py` + `src/kdsd/eval/benchmarks/{base,registry}.py`
   - `src/kdsd/sd/vllm_runner.py` (optional vLLM speedup pass)
   - `scripts/evaluate_sd.py` — the contract everyone depends on; build first.
6. `src/kdsd/data/{download,process,target_generate,logit_cache,dataset}.py`
   - `scripts/prepare_data.py`, `scripts/generate_target_responses.py`,
   `scripts/cache_target_logits.py`.
7. `src/kdsd/losses/{ce,fkl,rkl,jsd,combined}.py` + `tests/test_losses.py`.
8. `src/kdsd/train/{trainer,callbacks}.py` + `scripts/train.py`.
9. `src/kdsd/eval/benchmarks/{judge_gpt4,mt_bench}.py`.
10. `scripts/runtime_sweep.py`, `scripts/aggregate_results.py`.
11. `configs/**` — fill once `src/` is stable, then add ablation YAMLs cheaply.
12. `notebooks/{submit.sh, run_eval_pipeline.ipynb}` last, after the Python
    entrypoints work. `rcp_support/` is provided upstream — do not modify;
    `notebooks/submit.sh` is a verbatim copy with `GROUP` filled in.

External libraries leaned on rather than reimplemented:

- HF `DynamicCache` for KV-cache management inside our custom SD loop.
  (HF's `model.generate(assistant_model=draft, num_assistant_tokens=γ)` is a
  fine fallback path, but we use the custom loop to expose `accepted_lens`.)
- HF `Trainer` with a `compute_loss` override for KD.
- `datasets.load_dataset` for UltraChat / Alpaca download + caching.
- `accelerate` for mixed-precision and (later) multi-GPU.

---

## Verification

**Local (Mac/Windows) — unit tests only.** No model inference is exercised locally.

```bash
uv sync
uv run pytest -q
```

Required passing tests:

- `tests/test_losses.py` — FKL ≠ RKL on asymmetric distributions; JSD symmetric;
  CE-only matches masked NLL.
- `tests/test_data.py` — prompt masking sets `labels = -100` on prompt tokens and
  leaves response tokens intact.
- `tests/test_eval_schema.py` — fixture `eval_summary.json` with the dict-form
  `quality_score` validates; missing required field is rejected.
- `tests/test_config_compose.py` — every advertised Hydra combination
  (every `loss` × every `data`) resolves without error.

**Cluster — end-to-end smoke (one short job, ~10 min on A100).**

From the laptop, launch the interactive pod:

```bash
./notebooks/submit.sh smoke
runai port-forward <job-name> --port 8888:8888
```

Inside Jupyter (token `cs552`), in a Jupyter terminal under
`/scratch/<repo>`:

```bash
uv run python scripts/train.py run_name=smoke loss=fkl data=ultrachat_10k
```

Then open `notebooks/run_eval_pipeline.ipynb`, set
`RUN_NAME=smoke`, `DRAFT=checkpoints/smoke/model`, `Run All`. Delete the
job afterwards: `runai delete job <job-name>`.

Smoke acceptance:

- Training writes a loadable checkpoint and `meta.json`.
- Eval writes a schema-valid `eval_summary.json` for both `draft=null` (vanilla) and
  the trained draft, with finite, positive `speedup` and a populated
  `quality_score` dict.
- `aggregate_results.py` over a 2-row `results/` produces a markdown table.

Once cluster smoke is green, the documented ablation commands above are ready.

---

## Conventions

- Result identifiers (`run_name`) flow from Hydra and are reused as both the
  `checkpoints/<run_name>/` and `results/<run_name>/` directory names. Pick names
  that round-trip with `aggregate_results.py`.
- Never commit `data/`, `checkpoints/`, `results/`, or `wandb/`.
- Schema changes to `eval_summary.json` are breaking — bump a `schema_version` field
  and update `tests/test_eval_schema.py` + `aggregate_results.py` in the same PR.
- Hydra overrides go on the CLI, not in code. If you find yourself hard-coding a
  hyperparameter inside a script, it belongs in a config group.
