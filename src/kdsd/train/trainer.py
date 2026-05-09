"""HF Trainer subclass for online target-forward KD."""

from __future__ import annotations

from typing import Any

import torch
from transformers import Trainer

from kdsd.losses import kd_loss


class KDTrainer(Trainer):
    def __init__(self, *args: Any, target_model: torch.nn.Module, kd_cfg: dict, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.target = target_model.eval().requires_grad_(False)
        self.kd_cfg = dict(kd_cfg)
        self._last_loss_parts: dict[str, float] = {}

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs: Any,
    ):
        labels = inputs.pop("labels")
        response_mask = inputs.pop("response_mask", labels.ne(-100))

        student_out = model(**inputs)
        with torch.no_grad():
            teacher_out = self.target(**inputs)

        loss_parts = kd_loss(
            student_out.logits,
            teacher_out.logits,
            None,
            None,
            labels,
            kind=self.kd_cfg["kind"],
            temperature=float(self.kd_cfg.get("temperature", 1.0)),
            alpha=float(self.kd_cfg.get("alpha", 0.5)),
            loss_mask=response_mask,
        )
        self._last_loss_parts = {
            "loss_ce": float(loss_parts["ce"].detach().cpu()),
            "loss_kd": float(loss_parts["kd"].detach().cpu()),
        }
        if return_outputs:
            return loss_parts["loss"], student_out
        return loss_parts["loss"]

    def log(self, logs: dict[str, float], *args: Any, **kwargs: Any) -> None:
        if self._last_loss_parts:
            logs = {**logs, **self._last_loss_parts}
        super().log(logs, *args, **kwargs)
