"""SD evaluation entrypoint — see README.md §"Eval contract".

Runs ONE phase of the eval pipeline per invocation. Phases are kept in
separate processes (selected via the top-level `engine` Hydra field) because
vLLM cannot release its CUDA context, so it must never share one with HF.

    engine=hf    Load HF target (+ optional draft), run the instrumented loop,
                 write results/<run_name>/eval_summary.json plus generations
                 and timing artefacts.

    engine=vllm  Skip HF model loading. Re-tokenize prompts with the target's
                 tokenizer and run vanilla+SD passes through vLLM in spawn
                 subprocesses. Read results/<run_name>/eval_summary.json,
                 merge in the vLLM block (overwrites top-level
                 sd_time_s/vanilla_time_s/speedup/tps when SD ran), write back.
                 Errors out if no prior eval_summary.json exists for this
                 run_name — run engine=hf first, or use
                 scripts/run_eval_pipeline.py to drive both phases.

Override anything from the CLI, e.g.:

    uv run python scripts/evaluate_sd.py engine=hf \\
        draft=Qwen/Qwen2.5-0.5B-Instruct \\
        prompts.jsonl=data/processed/eval.jsonl \\
        prompts.limit=20 \\
        runtime.gamma=4 runtime.max_new_tokens=128 \\
        run_name=spec_smoke

The HF cache directory comes from `cfg.hf_cache` (defaults to /scratch/hf_cache
on the RunAI pod). It is exported into HF_HOME before transformers is
imported, since huggingface_hub reads HF_HOME at import time.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Make `src/` importable when running this file directly.
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Heavy imports (transformers, kdsd.*) are deferred into `main()` so that
# HF_HOME can be set from the resolved config first.
import hydra  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # ---- 1. Set HF_HOME BEFORE importing transformers/huggingface_hub.
    # huggingface_hub freezes its cache paths at import time, so a later
    # os.environ assignment is silently ignored.
    if cfg.get("hf_cache"):
        hf_home = os.path.expanduser(str(cfg.hf_cache))
        Path(hf_home).mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = hf_home
        os.environ["HF_HUB_CACHE"] = str(Path(hf_home) / "hub")
        os.environ["HF_DATASETS_CACHE"] = str(Path(hf_home) / "datasets")

    engine = str(cfg.get("engine", "hf")).lower()
    if engine not in ("hf", "vllm"):
        raise ValueError(f"engine must be 'hf' or 'vllm', got {engine!r}")

    out_dir = Path(cfg.results_dir)
    if not out_dir.is_absolute():
        out_dir = _ROOT / out_dir

    if engine == "hf":
        _run_hf_phase(cfg, out_dir)
    else:
        _run_vllm_phase(cfg, out_dir)


def _run_hf_phase(cfg: DictConfig, out_dir: Path) -> None:
    """Load HF target (+ optional draft), run instrumented eval, write artefacts."""
    import random
    import numpy as np
    import torch

    from kdsd.eval.runner import run_hf_eval
    from kdsd.models.loader import load_pair
    from kdsd.utils.io import read_jsonl, validate_eval_summary, write_json, write_jsonl
    from kdsd.utils.logging import get_logger

    LOG = get_logger("kdsd.evaluate_sd")
    LOG.info("[engine=hf] resolved config:\n%s", OmegaConf.to_yaml(cfg))
    LOG.info("HF_HOME=%s", os.environ.get("HF_HOME"))

    seed = int(cfg.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    pair = load_pair(
        target_id=str(cfg.model.target),
        draft_spec=(None if cfg.draft in (None, "", "null") else str(cfg.draft)),
        dtype=str(cfg.model.dtype),
        device=str(cfg.model.device),
        attn_impl=str(cfg.model.attn_impl),
        trust_remote_code=bool(cfg.model.trust_remote_code),
        draft_default=str(cfg.model.get("draft_default") or "") or None,
    )
    LOG.info(
        "Loaded target=%s draft=%s on %s (dtype=%s)",
        pair.target_id, pair.draft_id, pair.device, pair.dtype,
    )

    prompts = _load_prompts(cfg, read_jsonl, LOG)
    LOG.info("Loaded %d prompts", len(prompts))

    # Skip the HF vanilla baseline whenever a vLLM phase is going to be appended
    # (the orchestrator drives that). Default speedup config is `vllm`, so the
    # presence of a vLLM-style speedup config is the signal.
    speedup_engine = str(cfg.speedup.get("engine", "hf")).lower()
    skip_vanilla_baseline = (speedup_engine == "vllm")

    summary, rows = run_hf_eval(
        target=pair.target,
        draft=pair.draft,
        tokenizer=pair.tokenizer,
        prompts=prompts,
        runtime=OmegaConf.to_container(cfg.runtime, resolve=True),  # type: ignore[arg-type]
        eval_cfg=OmegaConf.to_container(cfg.eval, resolve=True),    # type: ignore[arg-type]
        device=pair.device,
        target_id=pair.target_id,
        draft_id=pair.draft_id,
        run_name=str(cfg.run_name),
        benchmarks=list(cfg.benchmark.get("benchmarks") or []),
        skip_vanilla_baseline=skip_vanilla_baseline,
    )

    validate_eval_summary(summary)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "eval_summary.json", summary)
    if cfg.eval.get("write_generations", True):
        write_jsonl(out_dir / "generations.jsonl", rows)
    _write_timing_json(out_dir, summary, write_json)
    OmegaConf.save(cfg, out_dir / "config.yaml")

    LOG.info("Wrote eval results to %s", out_dir)
    LOG.info(
        "[engine=hf] acceptance_rate=%.3f avg_accepted_tokens=%.2f "
        "speedup=%.2fx tokens/s=%.1f",
        summary["acceptance_rate"], summary["avg_accepted_tokens"],
        summary["speedup"], summary["tokens_per_second"],
    )


def _run_vllm_phase(cfg: DictConfig, out_dir: Path) -> None:
    """Run vLLM vanilla+SD passes; merge into the existing eval_summary.json."""
    from transformers import AutoTokenizer

    from kdsd.eval.runner import merge_vllm_into_summary, run_vllm_pass
    from kdsd.utils.io import read_jsonl, validate_eval_summary, write_json
    from kdsd.utils.logging import get_logger

    LOG = get_logger("kdsd.evaluate_sd")
    LOG.info("[engine=vllm] resolved config:\n%s", OmegaConf.to_yaml(cfg))

    summary_path = out_dir / "eval_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"engine=vllm requires an existing {summary_path}. "
            "Run engine=hf first, or use scripts/run_eval_pipeline.py."
        )
    with summary_path.open("r", encoding="utf-8") as fh:
        summary = json.load(fh)

    # Tokenizer-only — no GPU footprint in the parent.
    tokenizer = AutoTokenizer.from_pretrained(
        str(cfg.model.target),
        trust_remote_code=bool(cfg.model.trust_remote_code),
    )

    prompts = _load_prompts(cfg, read_jsonl, LOG)
    LOG.info("Loaded %d prompts", len(prompts))

    draft_id = None if cfg.draft in (None, "", "null") else str(cfg.draft)
    vllm_result = run_vllm_pass(
        tokenizer=tokenizer,
        prompts=prompts,
        runtime=OmegaConf.to_container(cfg.runtime, resolve=True),  # type: ignore[arg-type]
        speedup_cfg=OmegaConf.to_container(cfg.speedup, resolve=True),  # type: ignore[arg-type]
        target_id=str(cfg.model.target),
        draft_id=draft_id,
        seed=int(cfg.seed),
    )

    summary = merge_vllm_into_summary(summary, vllm_result)
    validate_eval_summary(summary)
    write_json(summary_path, summary)
    _write_timing_json(out_dir, summary, write_json)

    LOG.info("Updated %s with vLLM block", summary_path)
    LOG.info(
        "[engine=vllm] vanilla_ok=%s sd_ok=%s speedup=%.2fx tokens/s=%.1f",
        vllm_result.vanilla.ok, vllm_result.sd.ok,
        summary["speedup"], summary["tokens_per_second"],
    )


def _write_timing_json(out_dir: Path, summary: dict, write_json) -> None:
    """Project the timing fields out of `summary` into the standalone
    timing.json artefact (kept for back-compat with downstream scripts)."""
    write_json(out_dir / "timing.json", {
        "sd_time_s": summary["sd_time_s"],
        "vanilla_time_s": summary["vanilla_time_s"],
        "tokens_per_second": summary["tokens_per_second"],
        "n_warmup": summary["n_warmup"],
        "n_repeats": summary["n_repeats"],
    })


def _load_prompts(cfg: DictConfig, read_jsonl, LOG):
    """Load prompts from a jsonl path, an HF dataset spec, or a built-in fallback."""
    # PromptRecord lives in kdsd.eval.runner; import here (post-HF_HOME setup).
    from kdsd.eval.runner import PromptRecord

    p = cfg.prompts
    records: list[PromptRecord] = []
    if p.get("jsonl"):
        path = Path(p.jsonl)
        if not path.is_absolute():
            path = _ROOT / path
        for i, row in enumerate(read_jsonl(path)):
            records.append(
                PromptRecord(
                    id=str(row.get("id", i)),
                    prompt_text=row["prompt_text"],
                    response_text=row.get("response_text"),
                    source=row.get("source"),
                )
            )
    elif p.get("hf_dataset"):
        from datasets import load_dataset
        spec = p.hf_dataset
        ds = load_dataset(spec["name"], split=spec.get("split", "train"))
        field = spec.get("prompt_field", "prompt")
        for i, row in enumerate(ds):
            records.append(
                PromptRecord(
                    id=str(row.get("id", i)),
                    prompt_text=row[field],
                    source=spec["name"],
                )
            )
    else:
        LOG.warning("No prompt source set; using a tiny built-in smoke list")
        for i, t in enumerate([
            "Explain the theory of relativity in simple terms.",
            "Write a Python function to compute the Fibonacci sequence.",
            "What are the main differences between TCP and UDP?",
            "Summarise the plot of Romeo and Juliet in one paragraph.",
        ]):
            records.append(PromptRecord(id=f"smoke-{i}", prompt_text=t, source="builtin"))

    limit = p.get("limit")
    if limit is not None and limit > 0:
        records = records[: int(limit)]
    return records


if __name__ == "__main__":
    main()
