"""vLLM speedup engine: vanilla + SD passes in spawn subprocesses.

vLLM cannot free CUDA graphs / NCCL state / KV cache in-process — `del llm`
silently leaks and causes OOM on the second engine load. So each pass runs in
its own `mp.get_context("spawn")` subprocess; the parent never imports vllm.

Public API:
    `VllmEngineResult`, `VllmSpeedupResult`, `run_vllm_speedup`.

Prompt parity contract: the parent pre-tokenizes prompts with the *same*
tokenizer + chat template + add_special_tokens=False settings used by the HF
custom impl in `kdsd.eval.runner._generate_one`. We pass these token IDs to
vLLM via the TokensPrompt dict form (`{"prompt_token_ids": ids}`) so vLLM
bypasses its own tokenization for input. This guarantees byte-identical model
input across both engines. The tokenizer is still loaded inside the engine
(vLLM's spec-decode worker asserts on a None tokenizer_group, and EOS
detection on output requires it).

Spawn subprocesses inherit the parent's env, including HF_HOME (set by
scripts/evaluate_sd.py before any heavy imports). No extra env plumbing.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import re
import time
import traceback
from dataclasses import asdict, dataclass
from typing import Callable, Optional


# ── Result types ──────────────────────────────────────────────────────────────


@dataclass
class VllmSpecStats:
    draft_acceptance_rate: Optional[float] = None
    system_efficiency: Optional[float] = None
    num_accepted_tokens: Optional[int] = None
    num_draft_tokens: Optional[int] = None
    num_emitted_tokens: Optional[int] = None


@dataclass
class VllmEngineResult:
    ok: bool
    elapsed_s: float = 0.0
    tokens: int = 0
    spec_stats: Optional[VllmSpecStats] = None  # only populated on the SD engine
    error: Optional[str] = None


@dataclass
class VllmSpeedupResult:
    vanilla: VllmEngineResult
    sd: VllmEngineResult
    repeats: int = 0
    n_warmup: int = 0

    @property
    def vanilla_tokens_per_second(self) -> float:
        return self.vanilla.tokens / self.vanilla.elapsed_s if self.vanilla.elapsed_s > 0 else 0.0

    @property
    def sd_tokens_per_second(self) -> float:
        return self.sd.tokens / self.sd.elapsed_s if self.sd.elapsed_s > 0 else 0.0

    @property
    def speedup(self) -> float:
        v_tps = self.vanilla_tokens_per_second
        s_tps = self.sd_tokens_per_second
        return s_tps / v_tps if v_tps > 0 else 0.0


# ── Spec-decode log scraper (runs in subprocess) ──────────────────────────────


class _SpecLogHandler(logging.Handler):
    """Intercepts vLLM log lines that contain spec-decode metrics."""

    _RE = re.compile(
        r"[Dd]raft acceptance rate[:\s=]+([0-9.]+)"
        r".*?[Ss]ystem efficiency[:\s=]+([0-9.]+)"
        r"(?:.*?[Nn]umber of accepted tokens[:\s=]+(\d+))?"
        r"(?:.*?[Nn]umber of draft tokens[:\s=]+(\d+))?"
        r"(?:.*?[Nn]umber of emitted tokens[:\s=]+(\d+))?",
        re.DOTALL | re.IGNORECASE,
    )

    def __init__(self) -> None:
        super().__init__()
        self.records: list[VllmSpecStats] = []

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if "acceptance" not in msg.lower():
            return
        m = self._RE.search(msg)
        if m:
            self.records.append(
                VllmSpecStats(
                    draft_acceptance_rate=float(m.group(1)),
                    system_efficiency=float(m.group(2)),
                    num_accepted_tokens=int(m.group(3)) if m.group(3) else None,
                    num_draft_tokens=int(m.group(4)) if m.group(4) else None,
                    num_emitted_tokens=int(m.group(5)) if m.group(5) else None,
                )
            )

    def aggregate(self) -> VllmSpecStats:
        if not self.records:
            return VllmSpecStats()
        acc = [r.draft_acceptance_rate for r in self.records if r.draft_acceptance_rate is not None]
        eff = [r.system_efficiency for r in self.records if r.system_efficiency is not None]
        tot_acc = sum(r.num_accepted_tokens or 0 for r in self.records)
        tot_dft = sum(r.num_draft_tokens or 0 for r in self.records)
        tot_emt = sum(r.num_emitted_tokens or 0 for r in self.records)
        return VllmSpecStats(
            draft_acceptance_rate=sum(acc) / len(acc) if acc else None,
            system_efficiency=sum(eff) / len(eff) if eff else None,
            num_accepted_tokens=tot_acc or None,
            num_draft_tokens=tot_dft or None,
            num_emitted_tokens=tot_emt or None,
        )


def _try_engine_stats(llm) -> VllmSpecStats:
    """Best-effort read of spec-decode stats off the engine internals.

    vLLM doesn't always emit the metric log line we scrape; this is the fallback.
    """
    try:
        executor = llm.llm_engine.model_executor
        worker = getattr(executor, "driver_worker", None)
        if worker is None:
            return VllmSpecStats()
        sdw = getattr(worker, "spec_decode_worker", None)
        if sdw is None:
            return VllmSpecStats()
        sampler = getattr(sdw, "_spec_decode_sampler", None)
        if sampler is None:
            return VllmSpecStats()
        accepted = int(getattr(sampler, "num_accepted_tokens", 0))
        emitted = int(getattr(sampler, "num_emitted_tokens", 0))
        draft = int(getattr(sampler, "num_draft_tokens", 0))
        if draft == 0:
            return VllmSpecStats()
        return VllmSpecStats(
            draft_acceptance_rate=accepted / draft,
            num_accepted_tokens=accepted,
            num_draft_tokens=draft,
            num_emitted_tokens=emitted,
        )
    except Exception:
        return VllmSpecStats()


# ── Subprocess workers ────────────────────────────────────────────────────────


def _build_sampling(sampling_cfg: dict):
    """Map runtime-style sampling cfg to a vLLM SamplingParams (subprocess only)."""
    from vllm import SamplingParams

    mode = sampling_cfg["mode"]
    max_tokens = int(sampling_cfg["max_new_tokens"])
    if mode == "greedy" or float(sampling_cfg.get("temperature", 0.0)) <= 0:
        return SamplingParams(max_tokens=max_tokens, temperature=0.0)
    return SamplingParams(
        max_tokens=max_tokens,
        temperature=float(sampling_cfg["temperature"]),
        top_p=float(sampling_cfg["top_p"]),
        seed=int(sampling_cfg["seed"]),
    )


def _clamp_gmu(requested: float, headroom_gib: float = 1.0) -> float:
    """Cap `gpu_memory_utilization` to whatever is actually free on the GPU.

    vLLM treats `gpu_memory_utilization` as a fraction of *total* device memory
    and pre-reserves that much for weights+KV-cache. If another process (or
    leftover CUDA context from the parent) already holds memory, the requested
    fraction can exceed what's free → OOM during model load. We query free
    memory at subprocess start and clamp the fraction to `(free - headroom) /
    total`, leaving a small buffer for fragmentation.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return requested
        free_b, total_b = torch.cuda.mem_get_info()
        total_gib = total_b / (1024 ** 3)
        free_gib = free_b / (1024 ** 3)
        if total_gib <= 0:
            return requested
        usable_frac = max(0.1, (free_gib - headroom_gib) / total_gib)
        clamped = min(requested, usable_frac)
        if clamped < requested:
            print(
                f"[vllm_runner] clamping gpu_memory_utilization "
                f"{requested:.2f} → {clamped:.2f} "
                f"(free={free_gib:.1f} GiB, total={total_gib:.1f} GiB, "
                f"headroom={headroom_gib:.1f} GiB)",
                flush=True,
            )
        return clamped
    except Exception:
        return requested


def _as_token_prompts(prompt_token_ids: list[list[int]]) -> list[dict]:
    """Wrap raw token id lists in vLLM's TokensPrompt dict form.

    Bypasses vLLM's tokenizer so the model sees exactly the same input as the
    HF custom impl — which matters because vLLM's default `add_special_tokens`
    can differ from `runner._generate_one`.
    """
    return [{"prompt_token_ids": list(ids)} for ids in prompt_token_ids]


def _run_passes(llm, prompt_inputs: list[dict], sampling, repeats: int) -> tuple[float, int]:
    """Return (total_wall_s, total_new_tokens) across `repeats` batched passes."""
    total_s = 0.0
    total_tok = 0
    for _ in range(repeats):
        t0 = time.perf_counter()
        outputs = llm.generate(prompt_inputs, sampling)
        total_s += time.perf_counter() - t0
        total_tok += sum(len(o.outputs[0].token_ids) for o in outputs)
    return total_s, total_tok


# Tokenizer must be loaded inside the engine: EOS detection on output requires
# it, and vLLM's spec-decode worker asserts on a None tokenizer_group. We
# still bypass it for *input* by feeding TokensPrompt below.
_COMMON_LLM_KWARGS: dict = dict(skip_tokenizer_init=False)


def _build_vanilla_llm(args_dict: dict):
    from vllm import LLM
    return LLM(
        model=args_dict["target"],
        dtype=args_dict["dtype"],
        gpu_memory_utilization=_clamp_gmu(float(args_dict["gpu_memory_utilization"])),
        tensor_parallel_size=args_dict["tensor_parallel_size"],
        enforce_eager=args_dict["enforce_eager"],
        max_model_len=args_dict["max_model_len"],
        **_COMMON_LLM_KWARGS,
    )


def _build_sd_llm(args_dict: dict):
    from vllm import LLM

    # SD loads two models (target + draft) into the same budget, so reserve
    # extra headroom on top of the vanilla case.
    gmu = _clamp_gmu(
        float(args_dict["gpu_memory_utilization"]),
        headroom_gib=2.0 if not args_dict["ngram"] else 1.0,
    )

    # vLLM 0.11 collapsed the old spec-decode kwargs (`speculative_model`,
    # `num_speculative_tokens`, `ngram_prompt_lookup_*`) into a single
    # `speculative_config` dict.
    if args_dict["ngram"]:
        speculative_config: dict = {
            "method": "ngram",
            "num_speculative_tokens": int(args_dict["gamma"]),
            "prompt_lookup_max": int(args_dict["ngram_prompt_lookup_max"]),
            "prompt_lookup_min": int(args_dict["ngram_prompt_lookup_min"]),
        }
    else:
        speculative_config = {
            "model": args_dict["draft"],
            "num_speculative_tokens": int(args_dict["gamma"]),
        }

    kwargs: dict = dict(
        model=args_dict["target"],
        speculative_config=speculative_config,
        dtype=args_dict["dtype"],
        gpu_memory_utilization=gmu,
        tensor_parallel_size=args_dict["tensor_parallel_size"],
        enforce_eager=args_dict["enforce_eager"],
        max_model_len=args_dict["max_model_len"],
        **_COMMON_LLM_KWARGS,
    )

    try:
        return LLM(**kwargs)
    except (AssertionError, ValueError) as e:
        # vLLM raises when target/draft lm_head vocabs disagree (the rejection
        # sampler operates on logits). Surface a readable error pointing at the
        # actionable fix.
        tb = traceback.format_exc()
        if "_vocab_size" in tb or "spec_decode" in tb or "vocab_size" in tb:
            raise RuntimeError(
                "vLLM rejected this draft+target pair: lm_head vocab sizes differ.\n"
                f"  target = {args_dict['target']}\n"
                f"  draft  = {args_dict['draft']}\n"
                "Pick a draft+target with matching vocab (for Qwen2.5: stay\n"
                "within {3B,1.5B,0.5B} → 151936, or within {7B,14B+} → 152064),\n"
                "or set speedup.ngram=true for prompt-lookup SD."
            ) from e
        raise


def _attach_spec_log_capture() -> _SpecLogHandler:
    """Subscribe to vLLM logger names that emit spec-decode metric lines."""
    handler = _SpecLogHandler()
    handler.setLevel(logging.DEBUG)
    for name in (
        "",
        "vllm",
        "vllm.spec_decode.spec_decode_worker",
        "vllm.worker.spec_decode.spec_decode_worker",
        "vllm.engine.metrics",
    ):
        logging.getLogger(name).addHandler(handler)
    return handler


def _worker(
    args_dict: dict,
    queue: "mp.Queue",
    *,
    build_llm: Callable[[dict], object],
    capture_spec_stats: bool,
) -> None:
    """Generic vLLM subprocess body. `build_llm` constructs the engine; when
    `capture_spec_stats=True` we also scrape spec-decode metric log lines and
    fall back to engine internals for the same numbers.
    """
    try:
        # Log capture must be attached BEFORE creating LLM so early lines aren't lost.
        log_handler = _attach_spec_log_capture() if capture_spec_stats else None

        llm = build_llm(args_dict)

        sampling = _build_sampling(args_dict["sampling"])
        warmup = _build_sampling({"mode": "greedy", "max_new_tokens": 16})
        prompt_inputs = _as_token_prompts(args_dict["prompt_token_ids"])
        warmup_inputs = prompt_inputs[: max(1, min(2, len(prompt_inputs)))]

        for _ in range(args_dict["n_warmup"]):
            llm.generate(warmup_inputs, warmup)
        if log_handler is not None:
            log_handler.records.clear()  # discard warmup stats

        s, tok = _run_passes(llm, prompt_inputs, sampling, args_dict["repeats"])

        result: dict = {"ok": True, "elapsed_s": s, "tokens": tok}
        if log_handler is not None:
            spec_stats = log_handler.aggregate()
            if spec_stats.draft_acceptance_rate is None:
                spec_stats = _try_engine_stats(llm)
            result["spec_stats"] = asdict(spec_stats)
        queue.put(result)
    except Exception as e:
        queue.put({"ok": False, "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()})


def _vanilla_worker(args_dict: dict, queue: "mp.Queue") -> None:
    _worker(args_dict, queue, build_llm=_build_vanilla_llm, capture_spec_stats=False)


def _sd_worker(args_dict: dict, queue: "mp.Queue") -> None:
    _worker(args_dict, queue, build_llm=_build_sd_llm, capture_spec_stats=True)


def _run_in_subprocess(target_fn, args_dict: dict, label: str) -> dict:
    ctx = mp.get_context("spawn")
    queue: "mp.Queue" = ctx.Queue()
    proc = ctx.Process(target=target_fn, args=(args_dict, queue))
    proc.start()
    proc.join()
    if proc.exitcode != 0 and queue.empty():
        return {
            "ok": False,
            "error": f"{label} subprocess exited with code {proc.exitcode} (no result)",
            "tb": "",
        }
    if queue.empty():
        return {"ok": False, "error": f"{label} subprocess produced no result", "tb": ""}
    return queue.get()


# ── Public entrypoint ─────────────────────────────────────────────────────────


def run_vllm_speedup(
    *,
    prompt_token_ids: list[list[int]],
    target_id: str,
    draft_id: Optional[str],
    gamma: int,
    max_new_tokens: int,
    mode: str,
    temperature: float,
    top_p: float,
    seed: int,
    vllm_cfg: dict,
) -> VllmSpeedupResult:
    """Run vanilla and SD passes through vLLM (each in its own spawn subprocess).

    `prompt_token_ids` must be tokenized in the parent with the *same*
    tokenizer/template/special-tokens settings as the HF custom impl, so both
    engines see identical model input.

    `vllm_cfg` is the resolved `configs/speedup/vllm.yaml` dict (engine,
    gpu_memory_utilization, max_model_len, tensor_parallel_size, enforce_eager,
    dtype, ngram*, n_warmup, repeats).

    Failures in either subprocess are returned in the corresponding
    `VllmEngineResult.ok=False` rather than raising — the caller may still want
    to keep the HF results.
    """
    ngram = bool(vllm_cfg.get("ngram", False))
    if not ngram and not draft_id:
        return VllmSpeedupResult(
            vanilla=VllmEngineResult(ok=False, error="no draft_id provided and ngram=false"),
            sd=VllmEngineResult(ok=False, error="no draft_id provided and ngram=false"),
        )

    sampling = {
        "mode": mode,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "seed": seed,
    }
    common = dict(
        target=target_id,
        draft=draft_id,
        ngram=ngram,
        ngram_prompt_lookup_max=int(vllm_cfg.get("ngram_prompt_lookup_max", 4)),
        ngram_prompt_lookup_min=int(vllm_cfg.get("ngram_prompt_lookup_min", 2)),
        gamma=int(gamma),
        sampling=sampling,
        n_warmup=int(vllm_cfg.get("n_warmup", 1)),
        repeats=int(vllm_cfg.get("repeats", 3)),
        dtype=str(vllm_cfg.get("dtype", "bfloat16")),
        gpu_memory_utilization=float(vllm_cfg.get("gpu_memory_utilization", 0.85)),
        tensor_parallel_size=int(vllm_cfg.get("tensor_parallel_size", 1)),
        enforce_eager=bool(vllm_cfg.get("enforce_eager", False)),
        max_model_len=int(vllm_cfg.get("max_model_len", 4096)),
        prompt_token_ids=prompt_token_ids,
    )

    v_raw = _run_in_subprocess(_vanilla_worker, common, "vllm-vanilla")
    if v_raw.get("ok"):
        vanilla = VllmEngineResult(ok=True, elapsed_s=v_raw["elapsed_s"], tokens=v_raw["tokens"])
    else:
        vanilla = VllmEngineResult(ok=False, error=v_raw.get("error") or "unknown error")

    s_raw = _run_in_subprocess(_sd_worker, common, "vllm-sd")
    if s_raw.get("ok"):
        spec_stats = VllmSpecStats(**s_raw["spec_stats"]) if s_raw.get("spec_stats") else None
        sd = VllmEngineResult(
            ok=True, elapsed_s=s_raw["elapsed_s"], tokens=s_raw["tokens"], spec_stats=spec_stats
        )
    else:
        sd = VllmEngineResult(ok=False, error=s_raw.get("error") or "unknown error")

    return VllmSpeedupResult(
        vanilla=vanilla,
        sd=sd,
        repeats=int(common["repeats"]),
        n_warmup=int(common["n_warmup"]),
    )
