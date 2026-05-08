"""vLLM speculative-decoding speedup probe (thin CLI wrapper).

Mirrors scripts/hf_sd_speedup.py but uses vLLM. The actual runner lives in
`kdsd.sd.vllm_runner`; this script just sets up HF_HOME, parses args,
tokenizes the prompts, and prints the report. The eval pipeline
(`scripts/evaluate_sd.py speedup=vllm`) reuses the same module.

Vocab-size constraint:
    vLLM 0.7.3 asserts that draft and target have identical lm_head vocab
    sizes. Qwen2.5: 3B/1.5B/0.5B all have vocab=151936; 7B/14B+ are 152064 —
    pair within the same group, or use --ngram below.

Two SD modes:
    --draft <hf-id>      use a draft model (vLLM's standard draft SD)
    --ngram              use vLLM's [ngram] prompt-lookup speculator
                         (no draft model, no vocab issue)

Usage:
    python scripts/vllm_sd_speedup.py \\
        --target Qwen/Qwen2.5-3B-Instruct \\
        --draft  Qwen/Qwen2.5-0.5B-Instruct \\
        --gamma 4 --max-new-tokens 128 --repeats 3
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make `src/` importable when running this file directly.
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ── Must set HF_HOME *before* importing huggingface_hub / transformers.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--hf-cache", default=None)
_pre_args, _ = _pre.parse_known_args()
if _pre_args.hf_cache:
    _hf_home = os.path.expanduser(_pre_args.hf_cache)
    os.makedirs(_hf_home, exist_ok=True)
    os.environ["HF_HOME"] = _hf_home
    os.environ["HF_HUB_CACHE"] = os.path.join(_hf_home, "hub")
    os.environ["HF_DATASETS_CACHE"] = os.path.join(_hf_home, "datasets")

PROMPTS = [
    "Explain the theory of relativity in simple terms.",
    "Write a Python function to compute the Fibonacci sequence.",
    "What are the main differences between TCP and UDP?",
    "Summarise the plot of Romeo and Juliet in one paragraph.",
    "What is the capital of France and why is it historically significant?",
    "Describe the water cycle in detail.",
    "What are the main causes of World War I?",
    "Explain how neural networks learn from data.",
]


def _tokenize_prompts(target_id: str, prompts: list[str]) -> list[list[int]]:
    """Apply the target's chat template + tokenize once, in the parent process.

    Mirrors `kdsd.eval.runner._format_chat` + the `add_special_tokens=False`
    call in `_generate_one`, so that vLLM (with skip_tokenizer_init=True) sees
    byte-identical input to the HF custom impl.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(target_id, trust_remote_code=False)
    out: list[list[int]] = []
    for p in prompts:
        if getattr(tok, "chat_template", None):
            text = tok.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = p
        ids = tok(text, add_special_tokens=False)["input_ids"]
        out.append(list(ids))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--target", required=True, help="Target model HF ID or local path")
    ap.add_argument("--draft", default=None,
                    help="Draft model HF ID or local path (required unless --ngram)")
    ap.add_argument("--ngram", action="store_true",
                    help="Use vLLM's [ngram] prompt-lookup speculator instead of a draft model")
    ap.add_argument("--ngram-prompt-lookup-max", type=int, default=4)
    ap.add_argument("--ngram-prompt-lookup-min", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--gamma", type=int, default=4, help="num_speculative_tokens per SD step")
    ap.add_argument("--repeats", type=int, default=3, help="Timed batched passes")
    ap.add_argument("--n-warmup", type=int, default=1, help="Warmup passes (not timed)")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--gpu-memory-utilization", type=float, default=0.6,
        help=(
            "vLLM GPU memory fraction (of total). The runner clamps this down "
            "to actually-free memory at start, so a high value is safe on a "
            "dedicated GPU; on a shared GPU keep it at the default."
        ),
    )
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=4096,
                    help="Cap context length to keep KV cache small")
    ap.add_argument("--enforce-eager", action="store_true",
                    help="Disable CUDA graphs (lower mem, slower)")
    ap.add_argument("--hf-cache", default=None, help="Override HF_HOME (e.g. /scratch/hf_cache on RunAI)")
    args = ap.parse_args()

    if not args.ngram and args.draft is None:
        ap.error("--draft is required unless --ngram is set")

    from kdsd.sd.vllm_runner import run_vllm_speedup

    print(f"\n[1/2] Tokenizing {len(PROMPTS)} prompts with {args.target}'s tokenizer …")
    prompt_token_ids = _tokenize_prompts(args.target, PROMPTS)

    sd_label = "[ngram]" if args.ngram else args.draft
    mode = "greedy" if args.temperature <= 0 else "sampling"
    print(f"\n[2/2] Running vLLM passes: target={args.target}  draft={sd_label}  gamma={args.gamma}")

    vllm_cfg = dict(
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=args.enforce_eager,
        dtype=args.dtype,
        ngram=args.ngram,
        ngram_prompt_lookup_max=args.ngram_prompt_lookup_max,
        ngram_prompt_lookup_min=args.ngram_prompt_lookup_min,
        n_warmup=args.n_warmup,
        repeats=args.repeats,
    )
    result = run_vllm_speedup(
        prompt_token_ids=prompt_token_ids,
        target_id=args.target,
        draft_id=args.draft,
        gamma=args.gamma,
        max_new_tokens=args.max_new_tokens,
        mode=mode,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        vllm_cfg=vllm_cfg,
    )

    if not result.vanilla.ok:
        print(f"\n[vanilla] FAILED: {result.vanilla.error}")
    if not result.sd.ok:
        print(f"\n[sd] FAILED: {result.sd.error}")
    if not (result.vanilla.ok and result.sd.ok):
        sys.exit(1)

    v_tps = result.vanilla_tokens_per_second
    s_tps = result.sd_tokens_per_second
    sep = "─" * 62

    print(f"\n{sep}")
    print(f"  Target : {args.target}")
    print(f"  Draft  : {sd_label}")
    print(f"  gamma={args.gamma}  max_new_tokens={args.max_new_tokens}  "
          f"repeats={args.repeats}  dtype={args.dtype}")
    print(sep)
    print(f"  Vanilla throughput   : {v_tps:>8.1f} tok/s "
          f"({result.vanilla.tokens} tokens in {result.vanilla.elapsed_s:.2f}s)")
    print(f"  SD throughput        : {s_tps:>8.1f} tok/s "
          f"({result.sd.tokens} tokens in {result.sd.elapsed_s:.2f}s)")
    print(f"  Speedup              : {result.speedup:>8.2f}x")
    print(sep)
    ss = result.sd.spec_stats
    if ss is not None and ss.draft_acceptance_rate is not None:
        print(f"  Draft acceptance rate : {ss.draft_acceptance_rate:.4f}")
        if ss.system_efficiency is not None:
            print(f"  System efficiency     : {ss.system_efficiency:.4f}")
        if ss.num_accepted_tokens and ss.num_draft_tokens:
            print(f"  Accepted / Draft tok  : {ss.num_accepted_tokens} / {ss.num_draft_tokens}")
        if ss.num_emitted_tokens:
            print(f"  Emitted tokens (total): {ss.num_emitted_tokens}")
    else:
        print("  Draft acceptance rate : N/A — not found in vLLM logs")
        print("  Tip: run with VLLM_LOGGING_LEVEL=DEBUG for verbose spec-decode output")
    print(sep)


if __name__ == "__main__":
    main()
