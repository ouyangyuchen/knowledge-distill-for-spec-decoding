"""Drives the SD eval over a list of prompts and assembles eval_summary.json.

The HF and vLLM measurement passes are split into independent functions so a
caller can run each in its own process — vLLM cannot share a CUDA context with
HF without OOM. The orchestrator (scripts/run_eval_pipeline.py) drives this:

- `run_hf_eval(...)`: instrumented HF loop. Sole source for per-step metrics
  (acceptance_rate, accepted_lens, ...). Also produces an HF-side wall-clock
  speedup, which is conservative because of D→H syncs in the loop.
- `run_vllm_pass(...)` + `merge_vllm_into_summary(...)`: optional second pass
  that runs vanilla and SD through vLLM in spawn subprocesses; the merge helper
  promotes those numbers into the top-level summary fields and records the raw
  per-engine block under `engines.vllm`.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

from transformers import PreTrainedTokenizerBase

from kdsd.eval.benchmarks import registry as bench_registry
from kdsd.eval.metrics import aggregate_sd_stats
from kdsd.sd.instrument import SDStats, speculative_generate, vanilla_generate
from kdsd.utils.logging import get_logger
from kdsd.utils.timing import cuda_sync

LOG = get_logger("kdsd.eval")


@dataclass
class PromptRecord:
    id: str
    prompt_text: str
    response_text: Optional[str] = None  # reference (target_generated) text, if known
    source: Optional[str] = None


def _format_chat(tokenizer: PreTrainedTokenizerBase, prompt: str) -> str:
    """Use the model's chat template if available; fall back to raw prompt."""
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def _generate_one(
    *,
    target,
    draft,
    tokenizer: PreTrainedTokenizerBase,
    prompt_text_formatted: str,
    runtime: dict,
    device: str,
) -> tuple[str, SDStats, float]:
    enc = tokenizer(prompt_text_formatted, return_tensors="pt", add_special_tokens=False)
    input_ids = enc.input_ids.to(device)
    eos_id = tokenizer.eos_token_id

    cuda_sync()
    t0 = time.perf_counter()
    if draft is not None:
        ids, stats = speculative_generate(
            target,
            draft,
            input_ids,
            gamma=int(runtime["gamma"]),
            max_new_tokens=int(runtime["max_new_tokens"]),
            mode=str(runtime["mode"]),
            temperature=float(runtime["temperature"]),
            top_p=float(runtime["top_p"]),
            eos_token_id=eos_id,
        )
    else:
        ids, stats = vanilla_generate(
            target,
            input_ids,
            max_new_tokens=int(runtime["max_new_tokens"]),
            mode=str(runtime["mode"]),
            temperature=float(runtime["temperature"]),
            top_p=float(runtime["top_p"]),
            eos_token_id=eos_id,
        )
    cuda_sync()
    elapsed = time.perf_counter() - t0
    new_ids = ids[0, input_ids.shape[1]:]
    text_out = tokenizer.decode(new_ids, skip_special_tokens=True)
    return text_out, stats, elapsed


def run_hf_eval(
    *,
    target,
    draft,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[PromptRecord],
    runtime: dict,
    eval_cfg: dict,
    device: str,
    target_id: str,
    draft_id: Optional[str],
    run_name: str,
    benchmarks: list[str],
    skip_vanilla_baseline: bool = False,
) -> tuple[dict, list[dict]]:
    """Run the HF instrumented eval and return (eval_summary, generations rows).

    `skip_vanilla_baseline=True` skips the HF vanilla pass — set this when the
    caller plans to source vanilla timing from a later vLLM pass.
    """
    n_warmup = int(eval_cfg.get("n_warmup", 1))
    n_repeats = int(eval_cfg.get("n_repeats", 1))
    run_vanilla_baseline = bool(eval_cfg.get("run_vanilla_baseline", True))
    do_vanilla_hf = (
        (draft is not None) and run_vanilla_baseline and (not skip_vanilla_baseline)
    )
    if skip_vanilla_baseline and run_vanilla_baseline:
        LOG.info(
            "skip_vanilla_baseline=True → skipping HF vanilla baseline; "
            "vanilla_time_s will come from a later engine pass."
        )

    # Apply chat template once per prompt; the vLLM phase (separate process)
    # re-derives the same strings via _format_chat, so model input stays
    # identical across engines.
    formatted_prompts = [_format_chat(tokenizer, r.prompt_text) for r in prompts]

    # Warmup both code paths so the first measured prompt isn't paying for
    # cuBLAS / autotuner / kernel JIT on either side.
    if formatted_prompts and n_warmup > 0:
        warmup_text = formatted_prompts[0]
        for _ in range(n_warmup):
            _generate_one(
                target=target, draft=draft, tokenizer=tokenizer,
                prompt_text_formatted=warmup_text, runtime=runtime, device=device,
            )
            if do_vanilla_hf:
                _generate_one(
                    target=target, draft=None, tokenizer=tokenizer,
                    prompt_text_formatted=warmup_text, runtime=runtime, device=device,
                )

    # Interleave SD and vanilla per-prompt: running both in alternation keeps
    # GPU state (allocator fragmentation, clock state, thermal headroom)
    # roughly equivalent across the two measurements, instead of having one
    # block of SD followed by a separate block of vanilla.
    sd_stats: list[SDStats] = []
    sd_total_s = 0.0
    sd_total_tokens = 0
    v_total_s = 0.0
    v_total_tokens = 0
    rows: list[dict] = []

    for rec, formatted in zip(prompts, formatted_prompts):
        sd_runs: list[tuple[str, SDStats, float]] = []
        v_runs: list[tuple[str, SDStats, float]] = []
        for _ in range(max(1, n_repeats)):
            sd_runs.append(_generate_one(
                target=target, draft=draft, tokenizer=tokenizer,
                prompt_text_formatted=formatted, runtime=runtime, device=device,
            ))
            if do_vanilla_hf:
                v_runs.append(_generate_one(
                    target=target, draft=None, tokenizer=tokenizer,
                    prompt_text_formatted=formatted, runtime=runtime, device=device,
                ))

        # Feed every repeat's SDStats into aggregation so acceptance_rate /
        # avg_accepted_tokens are computed over all runs (not just the last).
        for _, s, _ in sd_runs:
            sd_stats.append(s)
        sd_times = [t for _, _, t in sd_runs]
        sd_tokens = [s.total_new_tokens for _, s, _ in sd_runs]
        sd_total_s += sum(sd_times) / len(sd_times)
        sd_total_tokens += int(round(sum(sd_tokens) / len(sd_tokens)))

        if v_runs:
            v_times = [t for _, _, t in v_runs]
            v_tokens = [s.total_new_tokens for _, s, _ in v_runs]
            v_total_s += sum(v_times) / len(v_times)
            v_total_tokens += int(round(sum(v_tokens) / len(v_tokens)))

        # Row keeps the last SD repeat's text + accepted_lens as the
        # representative generation; per-prompt times are reported as the
        # mean across repeats.
        last_text, last_stats, _ = sd_runs[-1]
        rows.append({
            "id": rec.id,
            "prompt": rec.prompt_text,
            "generation": last_text,
            "accepted_lens": last_stats.accepted_lens,
            "times": {
                "sd_s_avg": float(sum(sd_times) / len(sd_times)),
                "vanilla_s_avg": (
                    float(sum(v_times) / len(v_times)) if v_runs else None
                ),
            },
            "n_new_tokens": last_stats.total_new_tokens,
            "n_repeats": n_repeats,
            "finished_eos": last_stats.finished_eos,
        })

    # ---- HF-side timing and rate aggregation. Always computed; used either
    # as the top-level numbers (engine=hf) or surfaced under engines.hf
    # (engine=vllm).
    hf_vanilla_time_s: float
    hf_vanilla_tokens: int
    if draft is None:
        # No draft → the "SD" pass *is* the vanilla pass; reuse its numbers
        # so speedup is naturally 1.0.
        hf_vanilla_time_s = sd_total_s
        hf_vanilla_tokens = sd_total_tokens
    elif do_vanilla_hf:
        hf_vanilla_time_s = v_total_s
        hf_vanilla_tokens = v_total_tokens
    else:
        # Vanilla baseline skipped — caller intends to source vanilla timing
        # from a later vLLM pass via merge_vllm_into_summary().
        hf_vanilla_time_s = float("nan")
        hf_vanilla_tokens = 0

    hf_sd_tps = (sd_total_tokens / sd_total_s) if sd_total_s > 0 else 0.0
    hf_vanilla_tps = (
        (hf_vanilla_tokens / hf_vanilla_time_s)
        if (hf_vanilla_time_s and hf_vanilla_time_s > 0 and hf_vanilla_tokens > 0)
        else float("nan")
    )
    hf_speedup = (hf_sd_tps / hf_vanilla_tps) if (hf_vanilla_tps and hf_vanilla_tps > 0) else 1.0

    agg = aggregate_sd_stats(
        sd_stats, gamma=int(runtime["gamma"]) if draft is not None else None
    )

    # ---- Benchmark scoring (uses only `rows`; safe after model dealloc).
    quality_score: dict[str, float] = {}
    for name in benchmarks:
        try:
            cls = bench_registry.get(name)
        except KeyError as e:
            LOG.warning("benchmark %s not registered: %s — skipping", name, e)
            continue
        try:
            quality_score[name] = float(cls().score(rows, None))
        except Exception as e:  # missing API keys etc. → skip
            LOG.warning("benchmark %s failed: %s — skipping", name, e)

    # ---- Top-level numbers: HF-sourced. A later vLLM pass (driven by the
    # orchestrator) may overwrite these via merge_vllm_into_summary().
    sd_time_s = float(sd_total_s)
    vanilla_time_s = float(hf_vanilla_time_s)
    tokens_per_second = float(hf_sd_tps)
    speedup = float(hf_speedup)

    summary: dict = {
        "model": run_name,
        "target": target_id,
        "draft": draft_id if draft_id is not None else None,
        "acceptance_rate": float(agg["acceptance_rate"]),
        "avg_accepted_tokens": float(agg["avg_accepted_tokens"]),
        "vanilla_time_s": vanilla_time_s,
        "sd_time_s": sd_time_s,
        "speedup": speedup,
        "tokens_per_second": tokens_per_second,
        "quality_score": quality_score,
        "decoding": {
            "mode": runtime["mode"],
            "max_new_tokens": int(runtime["max_new_tokens"]),
            "num_assistant_tokens": int(runtime["gamma"]),
            "temperature": float(runtime["temperature"]),
            "top_p": float(runtime["top_p"]),
        },
        "n_prompts": int(len(prompts)),
        "n_warmup": n_warmup,
        "n_repeats": n_repeats,
    }

    # `engines.hf` mirrors the HF-side numbers verbatim so they survive even
    # after merge_vllm_into_summary() overwrites the top-level fields. The
    # vLLM block is added later by that helper. `batched=False` flags that HF
    # numbers are per-request latency, not continuous-batching throughput.
    engines: dict = {
        "hf": {
            "sd_time_s": float(sd_total_s),
            "vanilla_time_s": float(hf_vanilla_time_s) if not math.isnan(hf_vanilla_time_s) else None,
            "tokens_per_second": float(hf_sd_tps),
            "speedup": float(hf_speedup),
            "acceptance_rate": float(agg["acceptance_rate"]),
            "avg_accepted_tokens": float(agg["avg_accepted_tokens"]),
            "n_outer_steps": int(agg["n_outer_steps"]),
            "target_calls": int(agg["target_calls"]),
            "draft_calls": int(agg["draft_calls"]),
            "draft_forward_s": float(agg["draft_forward_s"]),
            "target_forward_s": float(agg["target_forward_s"]),
            "batched": False,
        },
    }
    summary["engines"] = engines
    return summary, rows


# ── vLLM speedup pass (separate process) ──────────────────────────────────────


def run_vllm_pass(
    *,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[PromptRecord],
    runtime: dict,
    speedup_cfg: dict,
    target_id: str,
    draft_id: Optional[str],
    seed: int = 42,
):
    """Run vanilla+SD passes through vLLM. Returns a `VllmSpeedupResult`.

    Imports vLLM lazily so callers that never reach the vLLM phase don't pay the
    import cost. The returned object can be merged into an existing
    eval_summary dict via `merge_vllm_into_summary`.
    """
    formatted_prompts = [_format_chat(tokenizer, r.prompt_text) for r in prompts]
    # Tokenize with the SAME settings as `_generate_one` (add_special_tokens=False)
    # so vLLM (skip_tokenizer_init=True) sees byte-identical input to the HF loop.
    prompt_token_ids = [
        list(tokenizer(t, add_special_tokens=False)["input_ids"])
        for t in formatted_prompts
    ]

    from kdsd.sd.vllm_runner import run_vllm_speedup

    return run_vllm_speedup(
        prompt_token_ids=prompt_token_ids,
        target_id=target_id,
        draft_id=draft_id,
        gamma=int(runtime["gamma"]),
        max_new_tokens=int(runtime["max_new_tokens"]),
        mode=str(runtime["mode"]),
        temperature=float(runtime["temperature"]),
        top_p=float(runtime["top_p"]),
        seed=int(seed),
        vllm_cfg=speedup_cfg,
    )


def merge_vllm_into_summary(summary: dict, vllm_result) -> dict:
    """Add vLLM block under `engines.vllm` and (when SD ran) promote vLLM's
    timings to the top-level fields.
    """
    v_block: dict = {
        "ok": bool(vllm_result.sd.ok and vllm_result.vanilla.ok),
        "vanilla_ok": bool(vllm_result.vanilla.ok),
        "sd_ok": bool(vllm_result.sd.ok),
        "vanilla_time_s": float(vllm_result.vanilla.elapsed_s),
        "sd_time_s": float(vllm_result.sd.elapsed_s),
        "vanilla_tokens": int(vllm_result.vanilla.tokens),
        "sd_tokens": int(vllm_result.sd.tokens),
        "tokens_per_second": float(vllm_result.sd_tokens_per_second),
        "vanilla_tokens_per_second": float(vllm_result.vanilla_tokens_per_second),
        "speedup": float(vllm_result.speedup),
        "repeats": int(vllm_result.repeats),
        "n_warmup": int(vllm_result.n_warmup),
        "batched": True,
        "error": (vllm_result.sd.error or vllm_result.vanilla.error),
    }
    if vllm_result.sd.spec_stats is not None:
        ss = vllm_result.sd.spec_stats
        v_block["spec_stats"] = {
            "draft_acceptance_rate": ss.draft_acceptance_rate,
            "system_efficiency": ss.system_efficiency,
            "num_accepted_tokens": ss.num_accepted_tokens,
            "num_draft_tokens": ss.num_draft_tokens,
            "num_emitted_tokens": ss.num_emitted_tokens,
        }
    summary.setdefault("engines", {})["vllm"] = v_block

    if vllm_result.sd.ok and vllm_result.sd.elapsed_s > 0:
        summary["sd_time_s"] = float(vllm_result.sd.elapsed_s)
        if vllm_result.vanilla.ok and vllm_result.vanilla.elapsed_s > 0:
            summary["vanilla_time_s"] = float(vllm_result.vanilla.elapsed_s)
        else:
            summary["vanilla_time_s"] = float("nan")
        summary["tokens_per_second"] = float(vllm_result.sd_tokens_per_second)
        summary["speedup"] = (
            float(vllm_result.speedup) if vllm_result.speedup > 0 else 1.0
        )
    return summary
