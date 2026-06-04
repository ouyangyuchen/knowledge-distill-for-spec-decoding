import json
import sys
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

from scripts import evaluate_sd


def test_checkpoint_metadata_from_model_dir(tmp_path):
    ckpt = tmp_path / "checkpoints" / "run-a"
    model_dir = ckpt / "model"
    model_dir.mkdir(parents=True)
    meta = {
        "run_name": "run-a",
        "wandb": {"id": "abc123", "project": "proj", "entity": "team"},
    }
    (ckpt / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    path, loaded = evaluate_sd._checkpoint_metadata_from_draft(model_dir)

    assert path == ckpt / "meta.json"
    assert loaded == meta


def test_checkpoint_metadata_from_checkpoint_dir(tmp_path):
    ckpt = tmp_path / "checkpoints" / "run-b"
    ckpt.mkdir(parents=True)
    meta = {"run_name": "run-b"}
    (ckpt / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    path, loaded = evaluate_sd._checkpoint_metadata_from_draft(ckpt)

    assert path == ckpt / "meta.json"
    assert loaded == meta


def test_prepare_draft_for_eval_materializes_pretrained_checkpoint(monkeypatch, tmp_path):
    calls = []

    class FakeModel:
        def save_pretrained(self, path, safe_serialization=True):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "config.json").write_text("{}", encoding="utf-8")
            (Path(path) / "model.safetensors").write_text("weights", encoding="utf-8")

    class FakeTokenizer:
        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "tokenizer.json").write_text("{}", encoding="utf-8")

    def fake_model_from_pretrained(model_id, **kwargs):
        calls.append(("model", model_id, kwargs))
        return FakeModel()

    def fake_tokenizer_from_pretrained(model_id, **kwargs):
        calls.append(("tokenizer", model_id, kwargs))
        return FakeTokenizer()

    fake_transformers = SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=fake_model_from_pretrained),
        AutoTokenizer=SimpleNamespace(from_pretrained=fake_tokenizer_from_pretrained),
        PreTrainedTokenizerBase=object,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    cfg = OmegaConf.create(
        {
            "draft": "pretrained",
            "pretrained_checkpoint_root": str(tmp_path / "pretrained"),
            "model": {
                "target": "Qwen/target",
                "draft_default": "Qwen/draft",
                "dtype": "bfloat16",
                "attn_impl": "sdpa",
                "trust_remote_code": False,
            },
        }
    )
    log = SimpleNamespace(info=lambda *args, **kwargs: None)

    model_dir = Path(evaluate_sd._prepare_draft_for_eval(cfg, log))

    assert model_dir.parent.parent == tmp_path / "pretrained"
    assert (model_dir / "config.json").exists()
    assert (model_dir / "model.safetensors").exists()
    assert (model_dir / "tokenizer.json").exists()
    meta = json.loads((model_dir.parent / "meta.json").read_text(encoding="utf-8"))
    assert meta["pretrained_checkpoint"] is True
    assert meta["steps"] == 0
    assert meta["draft_init"] == "Qwen/draft"
    assert [call[0] for call in calls] == ["model", "tokenizer"]

    calls.clear()
    assert Path(evaluate_sd._prepare_draft_for_eval(cfg, log)) == model_dir
    assert calls == []


def test_prepare_draft_for_eval_keeps_existing_local_checkpoint(tmp_path):
    model_dir = tmp_path / "checkpoints" / "run-a" / "model"
    model_dir.mkdir(parents=True)
    cfg = OmegaConf.create(
        {
            "draft": str(model_dir),
            "model": {"draft_default": "Qwen/draft"},
        }
    )
    log = SimpleNamespace(info=lambda *args, **kwargs: None)

    assert evaluate_sd._prepare_draft_for_eval(cfg, log) == str(model_dir)


def test_flatten_wandb_metrics_includes_summary_engine_and_quality():
    metrics = evaluate_sd._flatten_wandb_metrics(
        {
            "acceptance_rate": 0.5,
            "avg_accepted_tokens": 2.0,
            "speedup": 1.4,
            "tokens_per_second": 18.6,
            "sd_time_s": 10.0,
            "vanilla_time_s": 14.0,
            "n_prompts": 8,
            "n_warmup": 1,
            "n_repeats": 3,
            "engines": {
                "hf": {
                    "speedup": 1.4,
                    "target_calls": 12,
                    "batched": False,
                    "note": "ignored",
                }
            },
            "quality_score": {"mt_bench": 6.8},
        }
    )

    assert metrics["eval/acceptance_rate"] == 0.5
    assert metrics["eval/speedup"] == 1.4
    assert metrics["eval/hf/target_calls"] == 12
    assert metrics["eval/quality/mt_bench"] == 6.8
    assert "eval/hf/batched" not in metrics
    assert "eval/hf/note" not in metrics


def test_report_eval_to_wandb_uses_checkpoint_run_identity(monkeypatch, tmp_path):
    calls = {"init": None, "log": None, "finished": False}

    fake_wandb = SimpleNamespace(
        init=lambda **kwargs: calls.__setitem__("init", kwargs) or object(),
        log=lambda metrics: calls.__setitem__("log", metrics),
        finish=lambda: calls.__setitem__("finished", True),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    cfg = OmegaConf.create(
        {
            "run_name": "eval-run",
            "draft": str(tmp_path / "checkpoints" / "train-run" / "model"),
            "wandb": {
                "project": "cfg-project",
                "entity": "",
                "dir": str(tmp_path / "wandb"),
                "mode": "offline",
                "resume": "allow",
            },
        }
    )
    summary = {
        "target": "target",
        "draft": str(tmp_path / "checkpoints" / "train-run" / "model"),
        "acceptance_rate": 0.25,
        "avg_accepted_tokens": 1.0,
        "speedup": 1.2,
        "tokens_per_second": 9.0,
        "sd_time_s": 3.0,
        "vanilla_time_s": 3.6,
        "n_prompts": 2,
        "n_warmup": 0,
        "n_repeats": 1,
        "quality_score": {},
        "decoding": {"mode": "greedy"},
    }
    checkpoint_meta = {
        "run_name": "train-run",
        "wandb": {
            "id": "wandb-id",
            "name": "train-run",
            "project": "train-project",
            "entity": "train-entity",
        },
    }
    log = SimpleNamespace(warning=lambda *args, **kwargs: None)

    evaluate_sd._report_eval_to_wandb(
        cfg=cfg,
        summary=summary,
        out_dir=tmp_path / "results",
        checkpoint_meta=checkpoint_meta,
        checkpoint_meta_path=tmp_path / "checkpoints" / "train-run" / "meta.json",
        log=log,
    )

    assert calls["init"]["id"] == "wandb-id"
    assert calls["init"]["name"] == "train-run"
    assert calls["init"]["project"] == "train-project"
    assert calls["init"]["entity"] == "train-entity"
    assert calls["init"]["resume"] == "allow"
    assert calls["init"]["mode"] == "offline"
    assert calls["log"]["eval/speedup"] == 1.2
    assert calls["log"]["eval/acceptance_rate"] == 0.25
    assert calls["finished"] is True
