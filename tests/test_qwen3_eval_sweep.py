import csv
import json
from pathlib import Path

from scripts import qwen3_eval_sweep


def _write_checkpoint(
    root: Path,
    name: str,
    *,
    data_id: str = "ultrachat_50k",
    response_source: str | None = "original",
    loss_kind: str = "fkl",
    complete: bool = True,
) -> Path:
    run_dir = root / name
    model_dir = run_dir / "model"
    model_dir.mkdir(parents=True)
    data_lines = [
        f"id: {data_id}",
        "base_id: ultrachat_50k" if data_id.endswith("_target_gen") else "",
        f"response_source: {response_source}" if response_source is not None else "",
        "eval_path: ${data.processed_dir}/eval.jsonl",
    ]
    data_yaml = "\n".join(f"  {line}" for line in data_lines if line)
    (run_dir / "config.yaml").write_text(
        f"""
run_name: {name}
seed: 42
data_root: /scratch/cs552-data
model:
  target: Qwen/Qwen3-8B
  draft_default: Qwen/Qwen3-0.6B
loss:
  kind: {loss_kind}
  alpha: 1.0
  temperature: 1.0
data:
{data_yaml}
""".lstrip(),
        encoding="utf-8",
    )
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "run_name": name,
                "dataset_id": data_id,
                "draft_init": "Qwen/Qwen3-0.6B",
                "train_loss_final": 0.25,
                "steps": 100,
            }
        ),
        encoding="utf-8",
    )
    if complete:
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        (model_dir / "model.safetensors").write_text("weights", encoding="utf-8")
    return run_dir


def test_discover_checkpoints_skips_incomplete_and_resolves_target_gen_prompts(tmp_path):
    checkpoint_root = tmp_path / "checkpoints"
    complete = _write_checkpoint(
        checkpoint_root,
        "qwen3_8btarget_0p6b_tgen_fkl_ultrachat_50k_target_gen_seed42",
        data_id="ultrachat_50k_target_gen",
        response_source="target_generated",
    )
    _write_checkpoint(checkpoint_root, "incomplete", complete=False)

    records, skipped = qwen3_eval_sweep.discover_checkpoints(
        checkpoint_root=checkpoint_root,
        results_root=tmp_path / "results",
        hydra_root=tmp_path / "hydra",
        gamma=4,
        max_new_tokens=256,
    )

    assert len(records) == 1
    assert len(skipped) == 1
    record = records[0]
    assert record["checkpoint_dir"] == str(complete)
    assert record["loss_kind"] == "fkl"
    assert record["data_id"] == "ultrachat_50k_target_gen"
    assert record["response_source"] == "target_generated"
    assert record["prompts_jsonl"] == "/scratch/cs552-data/processed/ultrachat_50k/eval.jsonl"
    assert record["run"].endswith("_eval_g4_max256")


def test_discover_checkpoints_infers_response_source_when_missing(tmp_path):
    checkpoint_root = tmp_path / "checkpoints"
    _write_checkpoint(
        checkpoint_root,
        "qwen3_8btarget_0p6b_tgen_jsd_ultrachat_50k_target_gen_seed42",
        data_id="ultrachat_50k_target_gen",
        response_source=None,
    )

    records, skipped = qwen3_eval_sweep.discover_checkpoints(
        checkpoint_root=checkpoint_root,
        results_root=tmp_path / "results",
        hydra_root=tmp_path / "hydra",
        gamma=4,
        max_new_tokens=256,
    )

    assert skipped == []
    assert records[0]["response_source"] == "target_generated"


def test_write_summary_csv_includes_checkpoint_eval_config_and_vllm_metrics(tmp_path):
    checkpoint_root = tmp_path / "checkpoints"
    _write_checkpoint(checkpoint_root, "qwen3_8btarget_0p6b_fkl_ultrachat_50k_seed42")
    records, _ = qwen3_eval_sweep.discover_checkpoints(
        checkpoint_root=checkpoint_root,
        results_root=tmp_path / "results",
        hydra_root=tmp_path / "hydra",
        gamma=4,
        max_new_tokens=256,
    )
    summary_path = Path(records[0]["summary_path"])
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(
        json.dumps(
            {
                "speedup": 1.5,
                "acceptance_rate": 0.5,
                "avg_accepted_tokens": 2.0,
                "tokens_per_second": 18.0,
                "sd_time_s": 10.0,
                "vanilla_time_s": 15.0,
                "n_prompts": 128,
                "n_warmup": 1,
                "n_repeats": 1,
                "quality_score": {"exact_match_vs_target": 0.25},
                "engines": {
                    "vllm": {
                        "num_drafts": 8,
                        "num_draft_tokens": 32,
                        "num_accepted_tokens": 16,
                        "request_batch_size": 8,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    csv_path = tmp_path / "summary.csv"
    qwen3_eval_sweep.write_summary_csv(
        records=records,
        statuses={records[0]["run"]: {"status": "ok", "returncode": 0, "elapsed_s": "12.3"}},
        csv_path=csv_path,
        eval_config={
            "eval_backend": "vllm",
            "runtime_mode": "greedy",
            "runtime_temperature": "0.0",
            "runtime_top_p": "1.0",
            "gamma": 4,
            "max_new_tokens": 256,
            "run_vanilla_baseline": "true",
        },
    )

    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "ok"
    assert row["target_model"] == "Qwen/Qwen3-8B"
    assert row["loss_kind"] == "fkl"
    assert row["response_source"] == "original"
    assert row["eval_backend"] == "vllm"
    assert row["runtime_mode"] == "greedy"
    assert row["runtime_temperature"] == "0.0"
    assert row["vllm_num_drafts"] == "8"
    assert row["vllm_num_draft_tokens"] == "32"
    assert row["vllm_num_accepted_tokens"] == "16"
    assert row["quality_score.exact_match_vs_target"] == "0.25"
