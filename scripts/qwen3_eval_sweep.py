"""Discover Qwen3 checkpoints, run evals, and write a compact CSV summary."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV_NAME = "qwen3_checkpoint_eval_summary.csv"


CSV_COLUMNS = [
    "run",
    "checkpoint_run",
    "checkpoint_dir_name",
    "status",
    "returncode",
    "elapsed_s",
    "target_model",
    "draft_init",
    "loss_kind",
    "loss_alpha",
    "loss_temperature",
    "data_id",
    "response_source",
    "seed",
    "train_loss_final",
    "steps",
    "eval_backend",
    "runtime_mode",
    "runtime_temperature",
    "runtime_top_p",
    "gamma",
    "max_new_tokens",
    "run_vanilla_baseline",
    "speedup",
    "acceptance_rate",
    "avg_accepted_tokens",
    "tokens_per_second",
    "sd_time_s",
    "vanilla_time_s",
    "n_prompts",
    "n_warmup",
    "n_repeats",
    "vllm_num_drafts",
    "vllm_num_draft_tokens",
    "vllm_num_accepted_tokens",
    "vllm_request_batch_size",
    "checkpoint_dir",
    "model_dir",
    "config_path",
    "prompts_jsonl",
    "summary_path",
]


def str_to_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def has_model_weights(model_dir: Path) -> bool:
    return any(
        (
            any(model_dir.glob("*.safetensors")),
            any(model_dir.glob("pytorch_model*.bin")),
            (model_dir / "model.safetensors.index.json").exists(),
            (model_dir / "pytorch_model.bin.index.json").exists(),
        )
    )


def cfg_select(cfg, key: str, default=None):
    try:
        value = OmegaConf.select(cfg, key)
    except Exception:
        return default
    return default if value is None else value


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        value = json.load(fh)
    return value if isinstance(value, dict) else {}


def _string_or_empty(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _infer_data_id(run_name: str) -> str:
    if "ultrachat_50k_target_gen" in run_name:
        return "ultrachat_50k_target_gen"
    if "ultrachat_50k" in run_name:
        return "ultrachat_50k"
    if "ultrachat_25k" in run_name:
        return "ultrachat_25k"
    if "ultrachat_10k" in run_name:
        return "ultrachat_10k"
    if "alpaca_50k" in run_name:
        return "alpaca_50k"
    return ""


def _infer_response_source(data_id: str, run_name: str) -> str:
    if data_id.endswith("_target_gen") or "_tgen_" in run_name:
        return "target_generated"
    return "original"


def _resolve_path(value: str, *, repo_root: Path) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(value)))
    if not path.is_absolute():
        path = repo_root / path
    return path


def resolve_prompt_path(
    cfg,
    *,
    data_id: str,
    repo_root: Path = REPO_ROOT,
    override: str | None = None,
) -> Path:
    if override:
        return _resolve_path(override, repo_root=repo_root)

    eval_path = cfg_select(cfg, "data.eval_path")
    if eval_path is not None and "${" not in str(eval_path):
        return _resolve_path(str(eval_path), repo_root=repo_root)

    data_root = Path(str(cfg_select(cfg, "data_root", "/scratch/cs552-data"))).expanduser()
    base_id = str(cfg_select(cfg, "data.base_id", data_id))
    processed_id = base_id or data_id
    return data_root / "processed" / processed_id / "eval.jsonl"


def discover_checkpoints(
    *,
    checkpoint_root: Path,
    results_root: Path,
    hydra_root: Path,
    gamma: int,
    max_new_tokens: int,
    prompts_jsonl_override: str | None = None,
    repo_root: Path = REPO_ROOT,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    checkpoint_root = checkpoint_root.expanduser()
    if not checkpoint_root.exists():
        raise FileNotFoundError(f"checkpoint root does not exist: {checkpoint_root}")

    for run_dir in sorted(path for path in checkpoint_root.iterdir() if path.is_dir()):
        run_name = run_dir.name
        model_dir = run_dir / "model"
        config_path = run_dir / "config.yaml"
        reasons: list[str] = []
        if not config_path.exists():
            reasons.append("missing config.yaml")
        if not (model_dir / "config.json").exists():
            reasons.append("missing model/config.json")
        if not has_model_weights(model_dir):
            reasons.append("missing model weights")
        if reasons:
            skipped.append(
                {
                    "checkpoint_dir_name": run_name,
                    "checkpoint_dir": str(run_dir),
                    "reason": "; ".join(reasons),
                }
            )
            continue

        cfg = OmegaConf.load(config_path)
        meta = _read_json(run_dir / "meta.json")
        checkpoint_run = str(cfg_select(cfg, "run_name", meta.get("run_name") or run_name))
        data_id = str(
            cfg_select(cfg, "data.id", meta.get("dataset_id") or _infer_data_id(checkpoint_run))
        )
        response_source = str(
            cfg_select(cfg, "data.response_source", _infer_response_source(data_id, checkpoint_run))
        )
        prompt_path = resolve_prompt_path(
            cfg,
            data_id=data_id,
            repo_root=repo_root,
            override=prompts_jsonl_override,
        )
        eval_run = f"{checkpoint_run}_eval_g{int(gamma)}_max{int(max_new_tokens)}"
        results_dir = results_root / eval_run

        records.append(
            {
                "run": eval_run,
                "checkpoint_run": checkpoint_run,
                "checkpoint_dir_name": run_name,
                "target_model": str(cfg_select(cfg, "model.target", "Qwen/Qwen3-8B")),
                "draft_init": str(
                    meta.get("draft_init")
                    or cfg_select(cfg, "train.draft_init", None)
                    or cfg_select(cfg, "model.draft_default", "")
                ),
                "loss_kind": str(cfg_select(cfg, "loss.kind", "")),
                "loss_alpha": cfg_select(cfg, "loss.alpha", ""),
                "loss_temperature": cfg_select(cfg, "loss.temperature", ""),
                "data_id": data_id,
                "response_source": response_source,
                "seed": cfg_select(cfg, "seed", ""),
                "train_loss_final": meta.get("train_loss_final", ""),
                "steps": meta.get("steps", cfg_select(cfg, "train.steps", "")),
                "checkpoint_dir": str(run_dir),
                "model_dir": str(model_dir),
                "config_path": str(config_path),
                "prompts_jsonl": str(prompt_path),
                "results_dir": str(results_dir),
                "summary_path": str(results_dir / "eval_summary.json"),
                "hydra_dir": str(hydra_root / eval_run),
            }
        )

    return records, skipped


def build_eval_command(record: dict[str, Any], args: argparse.Namespace) -> list[str]:
    target_override = []
    if args.target_id:
        target_override.append(f"model.target={args.target_id}")

    return [
        sys.executable,
        "scripts/evaluate_sd.py",
        "model=qwen3",
        f"data={record['data_id']}",
        *target_override,
        f"draft={record['model_dir']}",
        f"pretrained_checkpoint_root={args.pretrained_checkpoint_root}",
        f"prompts.jsonl={record['prompts_jsonl']}",
        "prompts.hf_dataset=null",
        f"prompts.limit={args.prompts_limit}",
        f"runtime.mode={args.mode}",
        f"runtime.temperature={args.temperature}",
        f"runtime.top_p={args.top_p}",
        f"runtime.gamma={args.gamma}",
        f"runtime.max_new_tokens={args.max_new_tokens}",
        f"eval.backend={args.backend}",
        f"eval.n_warmup={args.warmup}",
        f"eval.n_repeats={args.repeats}",
        f"eval.run_vanilla_baseline={_bool_text(args.run_vanilla_baseline)}",
        f"wandb.enabled={_bool_text(args.report_to_wandb)}",
        f"results_dir={record['results_dir']}",
        f"hydra.run.dir={record['hydra_dir']}",
        f"run_name={record['run']}",
    ]


def _bool_text(value: bool) -> str:
    return "true" if bool(value) else "false"


def _metric_number(value: Any) -> Any:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return value
    return ""


def flatten_summary(summary_path: Path) -> dict[str, Any]:
    if not summary_path.exists():
        return {}
    with summary_path.open("r", encoding="utf-8") as fh:
        summary = json.load(fh)

    engine = (summary.get("engines") or {}).get("vllm") or {}
    row: dict[str, Any] = {
        "speedup": _metric_number(summary.get("speedup")),
        "acceptance_rate": _metric_number(summary.get("acceptance_rate")),
        "avg_accepted_tokens": _metric_number(summary.get("avg_accepted_tokens")),
        "tokens_per_second": _metric_number(summary.get("tokens_per_second")),
        "sd_time_s": _metric_number(summary.get("sd_time_s")),
        "vanilla_time_s": _metric_number(summary.get("vanilla_time_s")),
        "n_prompts": summary.get("n_prompts", ""),
        "n_warmup": summary.get("n_warmup", ""),
        "n_repeats": summary.get("n_repeats", ""),
        "vllm_num_drafts": engine.get("num_drafts", ""),
        "vllm_num_draft_tokens": engine.get("num_draft_tokens", ""),
        "vllm_num_accepted_tokens": engine.get("num_accepted_tokens", ""),
        "vllm_request_batch_size": engine.get("request_batch_size", ""),
    }
    for name, value in (summary.get("quality_score") or {}).items():
        row[f"quality_score.{name}"] = value
    return row


def write_summary_csv(
    *,
    records: list[dict[str, Any]],
    statuses: dict[str, dict[str, Any]],
    csv_path: Path,
    eval_config: dict[str, Any],
) -> None:
    rows: list[dict[str, Any]] = []
    extra_columns: list[str] = []

    for record in records:
        status = statuses.get(record["run"], {})
        row = {key: record.get(key, "") for key in CSV_COLUMNS}
        row.update(
            {
                "status": status.get("status", "not_run"),
                "returncode": status.get("returncode", ""),
                "elapsed_s": status.get("elapsed_s", ""),
                **eval_config,
            }
        )
        metrics = flatten_summary(Path(record["summary_path"]))
        row.update(metrics)
        for key in metrics:
            if key.startswith("quality_score.") and key not in extra_columns:
                extra_columns.append(key)
        rows.append(row)

    columns = [*CSV_COLUMNS, *sorted(extra_columns)]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def report_cached_eval_to_wandb(record: dict[str, Any]) -> None:
    from omegaconf import OmegaConf

    from scripts import evaluate_sd

    summary_path = Path(record["summary_path"])
    with summary_path.open("r", encoding="utf-8") as fh:
        summary = json.load(fh)
    draft = summary.get("draft")
    checkpoint_meta_path, checkpoint_meta = evaluate_sd._checkpoint_metadata_from_draft(draft)
    cfg = OmegaConf.create(
        {
            "run_name": record["run"],
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

    class Log:
        @staticmethod
        def warning(*args, **kwargs):
            print("WARNING:", *args, **kwargs)

    evaluate_sd._report_eval_to_wandb(
        cfg=cfg,
        summary=summary,
        out_dir=summary_path.parent,
        checkpoint_meta=checkpoint_meta,
        checkpoint_meta_path=checkpoint_meta_path,
        log=Log(),
    )


def _eval_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "eval_backend": args.backend,
        "runtime_mode": args.mode,
        "runtime_temperature": args.temperature,
        "runtime_top_p": args.top_p,
        "gamma": args.gamma,
        "max_new_tokens": args.max_new_tokens,
        "run_vanilla_baseline": _bool_text(args.run_vanilla_baseline),
    }


def run_sweep(args: argparse.Namespace) -> int:
    records, skipped = discover_checkpoints(
        checkpoint_root=Path(args.checkpoint_root),
        results_root=Path(args.results_root),
        hydra_root=Path(args.hydra_root),
        gamma=args.gamma,
        max_new_tokens=args.max_new_tokens,
        prompts_jsonl_override=args.prompts_jsonl_override or None,
    )
    for skipped_row in skipped:
        print(
            ">>> WARNING: skipping "
            f"{skipped_row['checkpoint_dir_name']}: {skipped_row['reason']}",
            file=sys.stderr,
        )
    if not records:
        print(f"ERROR: no complete checkpoints found under {args.checkpoint_root}", file=sys.stderr)
        return 1

    Path(args.results_root).mkdir(parents=True, exist_ok=True)
    Path(args.hydra_root).mkdir(parents=True, exist_ok=True)
    print(f">>> Found {len(records)} complete checkpoints")
    for record in records:
        print(
            ">>> checkpoint "
            f"{record['checkpoint_run']} loss={record['loss_kind']} "
            f"source={record['response_source']} prompts={record['prompts_jsonl']}"
        )

    statuses: dict[str, dict[str, Any]] = {}
    csv_path = Path(args.summary_csv or Path(args.results_root) / DEFAULT_CSV_NAME)
    eval_config = _eval_config_from_args(args)

    for idx, record in enumerate(records, start=1):
        summary_path = Path(record["summary_path"])
        if summary_path.exists() and not args.force_rerun:
            print(f">>> [{idx}/{len(records)}] Skipping cached {record['run']}")
            if args.report_to_wandb and args.report_cached_to_wandb:
                print(f">>> Reporting cached {record['run']} to W&B")
                report_cached_eval_to_wandb(record)
            statuses[record["run"]] = {"status": "cached", "returncode": 0, "elapsed_s": 0.0}
            write_summary_csv(records=records, statuses=statuses, csv_path=csv_path, eval_config=eval_config)
            continue

        cmd = build_eval_command(record, args)
        env = os.environ.copy()
        env["WANDB_JOB_TYPE"] = "eval"
        if args.wandb_group:
            env["WANDB_GROUP"] = args.wandb_group
        print(f">>> [{idx}/{len(records)}] Evaluating {record['run']}")
        print(f">>> command: {shlex.join(cmd)}")
        start = time.perf_counter()
        completed = subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=False)
        elapsed = time.perf_counter() - start
        status = "ok" if completed.returncode == 0 and summary_path.exists() else "failed"
        statuses[record["run"]] = {
            "status": status,
            "returncode": completed.returncode,
            "elapsed_s": f"{elapsed:.1f}",
        }
        write_summary_csv(records=records, statuses=statuses, csv_path=csv_path, eval_config=eval_config)
        if status == "failed":
            print(
                f">>> ERROR: {record['run']} failed with return code {completed.returncode}",
                file=sys.stderr,
            )

    print(f">>> Wrote CSV summary: {csv_path}")
    return 1 if any(row.get("status") == "failed" for row in statuses.values()) else 0


def print_discovery(args: argparse.Namespace) -> int:
    records, skipped = discover_checkpoints(
        checkpoint_root=Path(args.checkpoint_root),
        results_root=Path(args.results_root),
        hydra_root=Path(args.hydra_root),
        gamma=args.gamma,
        max_new_tokens=args.max_new_tokens,
        prompts_jsonl_override=args.prompts_jsonl_override or None,
    )
    payload = {"checkpoints": records, "skipped": skipped}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"complete_checkpoints={len(records)}")
        for record in records:
            print(
                "\t".join(
                    [
                        record["checkpoint_run"],
                        record["loss_kind"],
                        record["data_id"],
                        record["response_source"],
                        record["model_dir"],
                    ]
                )
            )
        if skipped:
            print(f"skipped={len(skipped)}", file=sys.stderr)
            for row in skipped:
                print(
                    f"{row['checkpoint_dir_name']}\t{row['reason']}",
                    file=sys.stderr,
                )
    return 0


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--hydra-root", required=True)
    parser.add_argument("--gamma", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--prompts-jsonl-override", default="")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover = subparsers.add_parser("discover")
    add_common_args(discover)
    discover.add_argument("--json", action="store_true")
    discover.set_defaults(func=print_discovery)

    run = subparsers.add_parser("run")
    add_common_args(run)
    run.add_argument("--summary-csv", default="")
    run.add_argument("--target-id", default="")
    run.add_argument("--pretrained-checkpoint-root", required=True)
    run.add_argument("--prompts-limit", required=True)
    run.add_argument("--warmup", required=True)
    run.add_argument("--repeats", required=True)
    run.add_argument("--backend", default="vllm")
    run.add_argument("--mode", default="greedy")
    run.add_argument("--temperature", default="0.0")
    run.add_argument("--top-p", default="1.0")
    run.add_argument("--report-to-wandb", type=str_to_bool, default=True)
    run.add_argument("--report-cached-to-wandb", type=str_to_bool, default=True)
    run.add_argument("--run-vanilla-baseline", type=str_to_bool, default=True)
    run.add_argument("--force-rerun", type=str_to_bool, default=False)
    run.add_argument("--wandb-group", default="")
    run.set_defaults(func=run_sweep)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
