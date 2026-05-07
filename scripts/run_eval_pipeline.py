"""Drive both eval phases (HF + vLLM) as separate processes.

vLLM cannot release its CUDA context cleanly, so it must not share one with HF.
This script invokes scripts/evaluate_sd.py twice as subprocesses with the same
config and run_name:

    1. engine=hf   → loads HF target+draft, runs the instrumented loop, writes
                     results/<run_name>/eval_summary.json (+ generations,
                     timing, config snapshot).
    2. engine=vllm → reads that eval_summary.json, runs vLLM vanilla+SD passes
                     in spawn subprocesses, merges the result back into the
                     same file.

Any extra arguments are forwarded verbatim as Hydra overrides to both phases.

Usage:
    uv run python scripts/run_eval_pipeline.py \\
        run_name=spec_smoke draft=Qwen/Qwen2.5-0.5B-Instruct \\
        prompts.jsonl=data/processed/eval.jsonl prompts.limit=20

Skip a phase with --skip-hf or --skip-vllm (runs only the other one).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_EVAL = _ROOT / "scripts" / "evaluate_sd.py"


def _run_phase(engine: str, hydra_overrides: list[str]) -> int:
    cmd = [sys.executable, str(_EVAL), f"engine={engine}", *hydra_overrides]
    print(f"\n[run_eval_pipeline] >>> {' '.join(cmd)}\n", flush=True)
    proc = subprocess.run(cmd, cwd=_ROOT)
    return proc.returncode


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--skip-hf", action="store_true",
                    help="Skip the HF phase (vLLM will need a prior eval_summary.json)")
    ap.add_argument("--skip-vllm", action="store_true",
                    help="Skip the vLLM phase (only HF metrics will be produced)")
    ap.add_argument("hydra_overrides", nargs=argparse.REMAINDER,
                    help="Hydra overrides forwarded to evaluate_sd.py "
                         "(e.g. run_name=foo draft=Qwen/...)")
    args = ap.parse_args()

    overrides = list(args.hydra_overrides or [])
    # argparse.REMAINDER keeps a leading '--' if the user typed one; strip it.
    if overrides and overrides[0] == "--":
        overrides = overrides[1:]

    if args.skip_hf and args.skip_vllm:
        ap.error("--skip-hf and --skip-vllm together leave nothing to run")

    if not args.skip_hf:
        rc = _run_phase("hf", overrides)
        if rc != 0:
            print(f"[run_eval_pipeline] HF phase exited with code {rc}; "
                  "skipping vLLM phase.", file=sys.stderr)
            sys.exit(rc)

    if not args.skip_vllm:
        rc = _run_phase("vllm", overrides)
        if rc != 0:
            print(f"[run_eval_pipeline] vLLM phase exited with code {rc}; "
                  "HF results are still on disk.", file=sys.stderr)
            sys.exit(rc)

    print("\n[run_eval_pipeline] all requested phases completed.", flush=True)


if __name__ == "__main__":
    main()
