"""HF speculative-decoding speedup probe — no kdsd dependency.

Uses transformers' built-in speculative decoding (assistant_model) to check
whether a draft/target pair actually accelerates inference before committing
to a custom SD loop.

Usage:
    python scripts/hf_sd_speedup.py \
        --target Qwen/Qwen2.5-7B-Instruct \
        --draft  Qwen/Qwen2.5-0.5B-Instruct \
        --gamma 4 --max-new-tokens 128 --repeats 5
"""

from __future__ import annotations

import argparse
import os
import time

# ── Must set HF_HOME *before* importing huggingface_hub / transformers.
# huggingface_hub reads HF_HOME into module-level constants at import time;
# any os.environ assignment after that is silently ignored.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--hf-cache", default=None)
_pre_args, _ = _pre.parse_known_args()
if _pre_args.hf_cache:
    _hf_home = os.path.expanduser(_pre_args.hf_cache)
    os.makedirs(_hf_home, exist_ok=True)
    os.environ["HF_HOME"] = _hf_home
    os.environ["HF_HUB_CACHE"] = os.path.join(_hf_home, "hub")
    os.environ["HF_DATASETS_CACHE"] = os.path.join(_hf_home, "datasets")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def fmt_prompt(tokenizer: AutoTokenizer, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def run_timed(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    gen_kwargs: dict,
    n_repeats: int,
) -> tuple[float, int]:
    """Return (mean wall-clock seconds, mean new tokens) over n_repeats runs."""
    total_s = 0.0
    total_tok = 0
    with torch.inference_mode():
        for _ in range(n_repeats):
            cuda_sync()
            t0 = time.perf_counter()
            out = model.generate(input_ids, **gen_kwargs)
            cuda_sync()
            total_s += time.perf_counter() - t0
            total_tok += out.shape[1] - input_ids.shape[1]
    return total_s / n_repeats, total_tok // n_repeats


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--target", required=True, help="Target model HF ID or local path")
    ap.add_argument("--draft", required=True, help="Draft (assistant) model HF ID or local path")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--gamma", type=int, default=4, help="num_assistant_tokens per SD step")
    ap.add_argument("--repeats", type=int, default=5, help="Timed repeats per prompt")
    ap.add_argument("--n-warmup", type=int, default=2, help="Warmup runs before timing")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--hf-cache", default=None, help="Override HF_HOME (e.g. /scratch/hf_cache on RunAI)")
    args = ap.parse_args()

    dtype ={"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    print(f"Loading tokenizer from {args.target} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading tokenizer from {args.draft} ...")
    draft_tokenizer = AutoTokenizer.from_pretrained(args.draft, trust_remote_code=True)
    if draft_tokenizer.pad_token_id is None:
        draft_tokenizer.pad_token_id = draft_tokenizer.eos_token_id

    print(f"Loading target: {args.target}")
    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=dtype, device_map=args.device, trust_remote_code=True
    ).eval()

    print(f"Loading draft:  {args.draft}")
    draft = AutoModelForCausalLM.from_pretrained(
        args.draft, dtype=dtype, device_map=args.device, trust_remote_code=True
    ).eval()

    base_kwargs: dict = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    sd_kwargs: dict = {
        **base_kwargs,
        "assistant_model": draft,
        "num_assistant_tokens": args.gamma,
        "tokenizer": tokenizer,
        "assistant_tokenizer": draft_tokenizer,
    }

    # Warmup both paths on the first prompt so cuBLAS / kernel JIT don't
    # pollute the first measured run.
    print(f"Warming up ({args.n_warmup} runs each path) ...")
    warmup_ids = tokenizer(
        fmt_prompt(tokenizer, PROMPTS[0]),
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(args.device)
    with torch.inference_mode():
        for _ in range(args.n_warmup):
            cuda_sync(); target.generate(warmup_ids, **base_kwargs); cuda_sync()
            cuda_sync(); target.generate(warmup_ids, **sd_kwargs);   cuda_sync()

    # ── Measurement ──────────────────────────────────────────────────────────
    W = 52
    print(f"\n{'Prompt':<{W}} {'Vanilla tok/s':>13} {'HF-SD tok/s':>11} {'Speedup':>9}")
    print("─" * (W + 37))

    agg_v_s = agg_v_tok = agg_sd_s = agg_sd_tok = 0.0

    for prompt in PROMPTS:
        ids = tokenizer(
            fmt_prompt(tokenizer, prompt),
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(args.device)

        v_s, v_tok   = run_timed(target, ids, base_kwargs, args.repeats)
        sd_s, sd_tok = run_timed(target, ids, sd_kwargs,   args.repeats)

        v_tps  = v_tok  / v_s  if v_s  > 0 else 0.0
        sd_tps = sd_tok / sd_s if sd_s > 0 else 0.0
        speedup = sd_tps / v_tps if v_tps > 0 else float("nan")

        agg_v_s   += v_s;   agg_v_tok  += v_tok
        agg_sd_s  += sd_s;  agg_sd_tok += sd_tok

        label = (prompt[: W - 3] + "...") if len(prompt) > W else prompt
        print(f"{label:<{W}} {v_tps:>13.1f} {sd_tps:>11.1f} {speedup:>8.2f}x")

    print("─" * (W + 37))
    agg_v_tps  = agg_v_tok  / agg_v_s
    agg_sd_tps = agg_sd_tok / agg_sd_s
    agg_speedup = agg_sd_tps / agg_v_tps
    print(f"{'AGGREGATE':<{W}} {agg_v_tps:>13.1f} {agg_sd_tps:>11.1f} {agg_speedup:>8.2f}x")

    print(f"\ntarget : {args.target}")
    print(f"draft  : {args.draft}")
    print(f"gamma={args.gamma}  max_new_tokens={args.max_new_tokens}  "
          f"repeats={args.repeats}  dtype={args.dtype}  device={args.device}")


if __name__ == "__main__":
    main()
