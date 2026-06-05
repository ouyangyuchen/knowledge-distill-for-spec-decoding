"""On-policy distillation utilities for speculative draft training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch.nn import functional as F
from transformers import Trainer

from kdsd.losses import kd_loss
from kdsd.sd.instrument import _proposal_dist, _sample_logits


@dataclass
class OPDRolloutStats:
    accepted_lens: list[int] = field(default_factory=list)
    proposed_count: int = 0
    target_calls: int = 0
    draft_calls: int = 0
    replay_count: int = 0
    generated_tokens: int = 0

    @property
    def acceptance_rate(self) -> float:
        if self.proposed_count <= 0:
            return 0.0
        return float(sum(self.accepted_lens)) / float(self.proposed_count)


def collect_opd_examples(
    *,
    target: torch.nn.Module,
    draft: torch.nn.Module,
    prompt_input_ids: torch.Tensor,
    gamma: int,
    rollout_max_new_tokens: int,
    max_seq_len: int,
    mode: str = "greedy",
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
    max_replay_examples: int = 1,
) -> tuple[list[dict[str, torch.Tensor]], OPDRolloutStats]:
    """Collect OPD training examples from one prompt.

    The main example follows target-assisted speculative rollout. Replay
    examples are one-step draft-induced contexts at rejected positions, with the
    target correction token used as the label so CE mixtures remain sensible.
    """
    if prompt_input_ids.dim() != 1:
        raise ValueError("prompt_input_ids must be a 1D tensor")
    if gamma < 1:
        raise ValueError("gamma must be >= 1")
    if rollout_max_new_tokens < 1:
        raise ValueError("rollout_max_new_tokens must be >= 1")

    device = prompt_input_ids.device
    prompt = prompt_input_ids.long()
    seq = prompt.clone()
    prompt_len = int(prompt.numel())
    stats = OPDRolloutStats()
    replay_examples: list[dict[str, torch.Tensor]] = []

    while (
        stats.generated_tokens < int(rollout_max_new_tokens)
        and int(seq.numel()) < int(max_seq_len)
    ):
        remaining = min(
            int(gamma),
            int(rollout_max_new_tokens) - stats.generated_tokens,
            int(max_seq_len) - int(seq.numel()),
        )
        if remaining <= 0:
            break
        stats.proposed_count += remaining

        candidates: list[torch.Tensor] = []
        q_dists: list[torch.Tensor] = []
        draft_seq = seq.clone()
        for _ in range(remaining):
            q_logits = _last_logits(draft, draft_seq)
            stats.draft_calls += 1
            q_dists.append(
                _proposal_dist(q_logits, mode=mode, temperature=temperature, top_p=top_p)
            )
            token = _sample_logits(q_logits, mode=mode, temperature=temperature, top_p=top_p)
            token = token.to(device=device, dtype=torch.long)
            candidates.append(token)
            draft_seq = torch.cat([draft_seq, token.view(1)], dim=0)

        target_seq = torch.cat([seq, torch.stack(candidates).view(-1)], dim=0)
        target_logits = _all_logits(target, target_seq)
        stats.target_calls += 1

        p_dists = []
        for i in range(remaining):
            logit_pos = int(seq.numel()) - 1 + i
            p_dists.append(
                _proposal_dist(
                    target_logits[logit_pos],
                    mode=mode,
                    temperature=temperature,
                    top_p=top_p,
                )
            )

        accept_mask = []
        for token, p_dist, q_dist in zip(candidates, p_dists, q_dists):
            idx = token.view(1)
            p_x = p_dist.gather(0, idx).clamp_min(0.0)
            q_x = q_dist.gather(0, idx).clamp_min(1e-12)
            accept_mask.append(bool((torch.rand((), device=device) * q_x <= p_x).item()))

        accepted = 0
        for ok in accept_mask:
            if ok:
                accepted += 1
            else:
                break
        stats.accepted_lens.append(accepted)

        accepted_tokens = candidates[:accepted]
        if accepted == remaining:
            bonus_logits = target_logits[int(seq.numel()) + remaining - 1]
            next_token = _sample_logits(
                bonus_logits,
                mode=mode,
                temperature=temperature,
                top_p=top_p,
            ).to(device=device, dtype=torch.long)
        else:
            reject_context = torch.cat(
                [seq, torch.stack(accepted_tokens).view(-1)]
                if accepted_tokens
                else [seq],
                dim=0,
            )
            next_token = _target_correction_token(
                p_dists[accepted],
                q_dists[accepted],
                mode=mode,
            ).to(device=device, dtype=torch.long)
            if len(replay_examples) < int(max_replay_examples):
                replay_examples.append(
                    _supervised_example(reject_context, next_token, int(reject_context.numel()))
                )
                stats.replay_count += 1

        segment_parts = accepted_tokens + [next_token]
        segment = torch.stack(segment_parts).view(-1)
        if stats.generated_tokens + int(segment.numel()) > int(rollout_max_new_tokens):
            keep = int(rollout_max_new_tokens) - stats.generated_tokens
            segment = segment[:keep]
        if int(seq.numel()) + int(segment.numel()) > int(max_seq_len):
            keep = int(max_seq_len) - int(seq.numel())
            segment = segment[:keep]
        if segment.numel() == 0:
            break

        seq = torch.cat([seq, segment], dim=0)
        stats.generated_tokens = int(seq.numel()) - prompt_len

        if eos_token_id is not None and bool(segment.eq(int(eos_token_id)).any().item()):
            eos_offsets = segment.eq(int(eos_token_id)).nonzero(as_tuple=False)
            end = int(seq.numel()) - int(segment.numel()) + int(eos_offsets[0].item()) + 1
            seq = seq[:end]
            stats.generated_tokens = int(seq.numel()) - prompt_len
            break

    examples = [_supervised_example(seq, None, prompt_len)]
    examples.extend(replay_examples)
    return examples, stats


class OPDTrainer(Trainer):
    def __init__(
        self,
        *args: Any,
        target_model: torch.nn.Module,
        kd_cfg: dict,
        opd_cfg: dict,
        pad_token_id: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.target = target_model.eval().requires_grad_(False)
        self.kd_cfg = dict(kd_cfg)
        self.opd_cfg = dict(opd_cfg)
        self.pad_token_id = int(pad_token_id)
        self.model_accepts_loss_kwargs = False
        self._loss_part_sums: dict[str, float] = {"loss_ce": 0.0, "loss_kd": 0.0}
        self._rollout_sums: dict[str, float] = {
            "opd_acceptance_rate": 0.0,
            "opd_avg_accepted_tokens": 0.0,
            "opd_replay_examples": 0.0,
            "opd_generated_tokens": 0.0,
            "opd_target_calls": 0.0,
            "opd_draft_calls": 0.0,
        }
        self._loss_part_count = 0

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs: Any,
    ):
        del kwargs
        was_training = model.training
        model.eval()
        examples: list[dict[str, torch.Tensor]] = []
        batch_stats: list[OPDRolloutStats] = []
        with torch.no_grad():
            for prompt_ids in _unpadded_prompts(inputs["input_ids"], inputs["attention_mask"]):
                rows, stats = collect_opd_examples(
                    target=self.target,
                    draft=model,
                    prompt_input_ids=prompt_ids,
                    gamma=int(self.opd_cfg.get("gamma", 4)),
                    rollout_max_new_tokens=int(self.opd_cfg.get("rollout_max_new_tokens", 128)),
                    max_seq_len=int(self.opd_cfg.get("max_seq_len", 1024)),
                    mode=str(self.opd_cfg.get("mode", "greedy")),
                    temperature=float(self.opd_cfg.get("temperature", 1.0)),
                    top_p=float(self.opd_cfg.get("top_p", 1.0)),
                    eos_token_id=self.opd_cfg.get("eos_token_id"),
                    max_replay_examples=int(self.opd_cfg.get("max_replay_examples", 1)),
                )
                examples.extend(rows)
                batch_stats.append(stats)
        if was_training:
            model.train()

        kd_inputs = _collate_kd_examples(examples, pad_token_id=self.pad_token_id)
        kd_inputs = {k: v.to(inputs["input_ids"].device) for k, v in kd_inputs.items()}
        labels = kd_inputs.pop("labels")
        response_mask = kd_inputs.pop("response_mask")

        student_out = model(**kd_inputs)
        teacher_logits = None
        if self.kd_cfg["kind"] != "ce":
            with torch.no_grad():
                teacher_logits = self.target(**kd_inputs).logits

        loss_parts = kd_loss(
            student_out.logits,
            teacher_logits,
            None,
            None,
            labels,
            kind=self.kd_cfg["kind"],
            temperature=float(self.kd_cfg.get("temperature", 1.0)),
            alpha=float(self.kd_cfg.get("alpha", 0.5)),
            loss_mask=response_mask,
            chunk_size=self.kd_cfg.get("chunk_size"),
        )

        if model.training:
            self._loss_part_sums["loss_ce"] += float(loss_parts["ce"].detach().cpu())
            self._loss_part_sums["loss_kd"] += float(loss_parts["kd"].detach().cpu())
            self._rollout_sums["opd_acceptance_rate"] += _mean_acceptance(batch_stats)
            self._rollout_sums["opd_avg_accepted_tokens"] += _mean_accepted_tokens(batch_stats)
            self._rollout_sums["opd_replay_examples"] += float(
                sum(s.replay_count for s in batch_stats)
            )
            self._rollout_sums["opd_generated_tokens"] += float(
                sum(s.generated_tokens for s in batch_stats)
            )
            self._rollout_sums["opd_target_calls"] += float(
                sum(s.target_calls for s in batch_stats)
            )
            self._rollout_sums["opd_draft_calls"] += float(
                sum(s.draft_calls for s in batch_stats)
            )
            self._loss_part_count += 1
        if return_outputs:
            return loss_parts["loss"], student_out
        return loss_parts["loss"]

    def log(self, logs: dict[str, float], *args: Any, **kwargs: Any) -> None:
        if "loss" in logs and self._loss_part_count > 0:
            logs = {
                **logs,
                **{
                    k: v / self._loss_part_count
                    for k, v in self._loss_part_sums.items()
                },
                **{
                    k: v / self._loss_part_count
                    for k, v in self._rollout_sums.items()
                },
            }
            self._loss_part_sums = {"loss_ce": 0.0, "loss_kd": 0.0}
            self._rollout_sums = {
                "opd_acceptance_rate": 0.0,
                "opd_avg_accepted_tokens": 0.0,
                "opd_replay_examples": 0.0,
                "opd_generated_tokens": 0.0,
                "opd_target_calls": 0.0,
                "opd_draft_calls": 0.0,
            }
            self._loss_part_count = 0
        super().log(logs, *args, **kwargs)


def _last_logits(model: torch.nn.Module, seq: torch.Tensor) -> torch.Tensor:
    return _all_logits(model, seq)[-1]


def _all_logits(model: torch.nn.Module, seq: torch.Tensor) -> torch.Tensor:
    attention_mask = torch.ones(1, int(seq.numel()), dtype=torch.long, device=seq.device)
    out = model(input_ids=seq.view(1, -1), attention_mask=attention_mask)
    return out.logits[0]


def _target_correction_token(p_dist: torch.Tensor, q_dist: torch.Tensor, *, mode: str) -> torch.Tensor:
    if mode == "greedy":
        return p_dist.argmax(dim=-1)
    diff = (p_dist - q_dist).clamp_min(0.0)
    total = diff.sum()
    if float(total.item()) > 0:
        return torch.multinomial(diff / total, num_samples=1).squeeze(-1)
    return p_dist.argmax(dim=-1)


def _supervised_example(
    seq: torch.Tensor,
    final_label_token: torch.Tensor | None,
    prompt_len: int,
) -> dict[str, torch.Tensor]:
    input_ids = seq.detach().long().cpu()
    if final_label_token is not None:
        input_ids = torch.cat(
            [input_ids, final_label_token.detach().long().cpu().view(1)],
            dim=0,
        )
    labels = torch.full_like(input_ids, -100)
    response_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    if final_label_token is None:
        labels[int(prompt_len):] = input_ids[int(prompt_len):]
        response_mask[int(prompt_len):] = True
    else:
        labels[-1] = final_label_token.detach().long().cpu()
        response_mask[-1] = True
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids, dtype=torch.long),
        "labels": labels,
        "response_mask": response_mask,
    }


def _unpadded_prompts(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> list[torch.Tensor]:
    out = []
    for ids, mask in zip(input_ids, attention_mask):
        out.append(ids[mask.bool()].detach())
    return out


def _collate_kd_examples(
    examples: list[dict[str, torch.Tensor]],
    *,
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    max_len = max(int(ex["input_ids"].numel()) for ex in examples)
    batch: dict[str, list[torch.Tensor]] = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
        "response_mask": [],
    }
    for ex in examples:
        pad = max_len - int(ex["input_ids"].numel())
        batch["input_ids"].append(F.pad(ex["input_ids"], (0, pad), value=int(pad_token_id)))
        batch["attention_mask"].append(F.pad(ex["attention_mask"], (0, pad), value=0))
        batch["labels"].append(F.pad(ex["labels"], (0, pad), value=-100))
        batch["response_mask"].append(F.pad(ex["response_mask"], (0, pad), value=0))
    return {
        "input_ids": torch.stack(batch["input_ids"]).long(),
        "attention_mask": torch.stack(batch["attention_mask"]).long(),
        "labels": torch.stack(batch["labels"]).long(),
        "response_mask": torch.stack(batch["response_mask"]).bool(),
    }


def _mean_acceptance(stats: list[OPDRolloutStats]) -> float:
    accepted = 0
    proposed = 0
    for s in stats:
        accepted += sum(s.accepted_lens)
        proposed += s.proposed_count
    return float(accepted) / float(proposed) if proposed else 0.0


def _mean_accepted_tokens(stats: list[OPDRolloutStats]) -> float:
    values = [x for s in stats for x in s.accepted_lens]
    return float(sum(values)) / float(len(values)) if values else 0.0
