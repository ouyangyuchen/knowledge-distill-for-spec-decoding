import inspect
from types import SimpleNamespace

import torch
from torch import nn
from transformers import TrainingArguments

from kdsd.train import OPDTrainer, collect_opd_examples


class ConstantLM(nn.Module):
    def __init__(self, preferred_token: int, vocab_size: int = 8):
        super().__init__()
        self.config = SimpleNamespace(use_cache=False)
        self.preferred_token = int(preferred_token)
        self.vocab_size = int(vocab_size)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        logits = torch.zeros(
            input_ids.shape[0],
            input_ids.shape[1],
            self.vocab_size,
            device=input_ids.device,
        )
        logits[..., self.preferred_token] = 10.0
        return SimpleNamespace(logits=logits)


class TinyLM(nn.Module):
    def __init__(self, vocab_size: int = 8, hidden_size: int = 6):
        super().__init__()
        self.config = SimpleNamespace(use_cache=False)
        self.emb = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        return SimpleNamespace(logits=self.proj(self.emb(input_ids)))


def test_collect_opd_examples_shapes_and_replay():
    target = ConstantLM(preferred_token=3)
    draft = ConstantLM(preferred_token=5)
    prompt = torch.tensor([2, 4], dtype=torch.long)

    examples, stats = collect_opd_examples(
        target=target,
        draft=draft,
        prompt_input_ids=prompt,
        gamma=2,
        rollout_max_new_tokens=2,
        max_seq_len=8,
        mode="greedy",
        temperature=1.0,
        top_p=1.0,
        max_replay_examples=1,
    )

    assert stats.replay_count == 1
    assert len(examples) == 2
    for ex in examples:
        assert ex["input_ids"].shape == ex["labels"].shape
        assert ex["input_ids"].shape == ex["attention_mask"].shape
        assert ex["response_mask"].dtype is torch.bool
        assert ex["response_mask"].any()


def test_opd_trainer_smoke_with_tiny_models(tmp_path):
    dataset = [
        {
            "input_ids": torch.tensor([2, 3, 4]),
            "attention_mask": torch.ones(3, dtype=torch.long),
        },
        {
            "input_ids": torch.tensor([2, 5, 6]),
            "attention_mask": torch.ones(3, dtype=torch.long),
        },
    ]
    kwargs = {
        "output_dir": str(tmp_path),
        "max_steps": 1,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "report_to": [],
        "remove_unused_columns": False,
        "save_strategy": "no",
        "logging_steps": 1,
    }
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "no"
    else:
        kwargs["evaluation_strategy"] = "no"
    if "use_cpu" in params:
        kwargs["use_cpu"] = True
    elif "no_cuda" in params:
        kwargs["no_cuda"] = True

    trainer = OPDTrainer(
        model=TinyLM(),
        target_model=TinyLM(),
        args=TrainingArguments(**kwargs),
        train_dataset=dataset,
        kd_cfg={"kind": "fkl", "alpha": 1.0, "temperature": 1.0},
        opd_cfg={
            "gamma": 1,
            "rollout_max_new_tokens": 2,
            "max_seq_len": 8,
            "mode": "greedy",
            "temperature": 1.0,
            "top_p": 1.0,
            "max_replay_examples": 1,
            "eos_token_id": None,
        },
        pad_token_id=0,
    )

    result = trainer.train()

    assert result.training_loss >= 0
    train_log = next(row for row in trainer.state.log_history if "loss" in row)
    assert "opd_acceptance_rate" in train_log
