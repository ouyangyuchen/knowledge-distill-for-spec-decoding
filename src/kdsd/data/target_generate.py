"""Target response generation for response-source ablations."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from tqdm.auto import tqdm

from kdsd.data.dataset import format_prompt


@torch.inference_mode()
def generate_target_responses(
    records: Iterable[dict[str, Any]],
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    max_new_tokens: int,
    mode: str = "greedy",
    temperature: float = 0.0,
    top_p: float = 1.0,
    device: str = "cuda",
    show_progress: bool = True,
    progress_desc: str = "target responses",
) -> list[dict[str, Any]]:
    rows = list(records)
    batch_size = max(1, int(batch_size))
    out: list[dict[str, Any]] = []
    old_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    try:
        starts = range(0, len(rows), batch_size)
        progress = tqdm(
            starts,
            total=(len(rows) + batch_size - 1) // batch_size,
            desc=progress_desc,
            unit="batch",
            disable=not show_progress,
        )
        for start in progress:
            batch = rows[start:start + batch_size]
            prompts = [format_prompt(tokenizer, str(row["prompt_text"])) for row in batch]
            enc = tokenizer(
                prompts,
                padding=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            input_len = int(enc["input_ids"].shape[1])
            do_sample = mode == "sampling" and float(temperature) > 0
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": int(max_new_tokens),
                "do_sample": do_sample,
                "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if do_sample:
                gen_kwargs["temperature"] = float(temperature)
                gen_kwargs["top_p"] = float(top_p)
            generated = model.generate(**enc, **gen_kwargs)
            new_tokens = generated[:, input_len:]
            texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            for row, text in zip(batch, texts):
                out.append(_target_generated_row(row, text))
            progress.set_postfix(prompts=len(out))
    finally:
        tokenizer.padding_side = old_padding_side
    return out


def _target_generated_row(row: dict[str, Any], text: str) -> dict[str, Any]:
    next_row = dict(row)
    meta = dict(next_row.get("metadata") or {})
    meta["original_response_text"] = next_row.get("response_text")
    meta["response_generated_at"] = datetime.now(timezone.utc).isoformat()
    next_row["response_text"] = text.strip()
    next_row["source"] = "target"
    next_row["metadata"] = meta
    return next_row


def _vllm_sampling_params(
    sampling_params_cls: type,
    *,
    max_new_tokens: int,
    mode: str,
    temperature: float,
    top_p: float,
    seed: int | None,
) -> Any:
    if mode not in {"greedy", "sampling"}:
        raise ValueError(f"Unsupported target generation mode: {mode}")
    kwargs: dict[str, Any] = {
        "max_tokens": int(max_new_tokens),
        "temperature": 0.0 if mode == "greedy" else float(temperature),
        "top_p": 1.0 if mode == "greedy" else float(top_p),
    }
    if mode == "sampling" and seed is not None:
        kwargs["seed"] = int(seed)
    return sampling_params_cls(**kwargs)


def create_vllm_target_engine(
    *,
    model_id: str,
    max_new_tokens: int,
    mode: str = "greedy",
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int | None = None,
    dtype: str = "bfloat16",
    trust_remote_code: bool = False,
    tensor_parallel_size: int = 1,
    max_model_len: int = 2048,
    gpu_memory_utilization: float = 0.9,
    swap_space: float = 0,
    enforce_eager: bool = False,
) -> tuple[Any, Any]:
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError(
            "target_generation.backend=vllm requires vLLM. Run this command in the "
            "RunAI/course vLLM image, or install vLLM in the active environment."
        ) from exc

    sampling_params = _vllm_sampling_params(
        SamplingParams,
        max_new_tokens=int(max_new_tokens),
        mode=str(mode),
        temperature=float(temperature),
        top_p=float(top_p),
        seed=seed,
    )
    llm = LLM(
        model=str(model_id),
        tokenizer=str(model_id),
        dtype=str(dtype),
        trust_remote_code=bool(trust_remote_code),
        tensor_parallel_size=int(tensor_parallel_size),
        max_model_len=int(max_model_len),
        gpu_memory_utilization=float(gpu_memory_utilization),
        swap_space=float(swap_space),
        enforce_eager=bool(enforce_eager),
        seed=None if seed is None else int(seed),
    )
    return llm, sampling_params


def generate_target_responses_vllm(
    records: Iterable[dict[str, Any]],
    *,
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    request_batch_size: int,
    max_new_tokens: int,
    mode: str = "greedy",
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int | None = None,
    dtype: str = "bfloat16",
    trust_remote_code: bool = False,
    tensor_parallel_size: int = 1,
    max_model_len: int = 2048,
    gpu_memory_utilization: float = 0.9,
    swap_space: float = 0,
    enforce_eager: bool = False,
    llm: Any | None = None,
    sampling_params: Any | None = None,
    show_progress: bool = True,
    progress_desc: str = "target responses",
) -> list[dict[str, Any]]:
    rows = list(records)
    request_batch_size = max(1, int(request_batch_size))
    if llm is None or sampling_params is None:
        llm, sampling_params = create_vllm_target_engine(
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            mode=mode,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            swap_space=swap_space,
            enforce_eager=enforce_eager,
        )

    out: list[dict[str, Any]] = []
    starts = range(0, len(rows), request_batch_size)
    progress = tqdm(
        starts,
        total=(len(rows) + request_batch_size - 1) // request_batch_size,
        desc=progress_desc,
        unit="batch",
        disable=not show_progress,
    )
    for start in progress:
        batch = rows[start:start + request_batch_size]
        prompts = [format_prompt(tokenizer, str(row["prompt_text"])) for row in batch]
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        for row, request_output in zip(batch, outputs):
            text = request_output.outputs[0].text if request_output.outputs else ""
            out.append(_target_generated_row(row, text))
        progress.set_postfix(prompts=len(out))
    return out


def generation_meta(
    *,
    target_model: str,
    seed: int,
    generation_cfg: dict[str, Any],
    source_path: str,
    n_records: int,
) -> dict[str, Any]:
    return {
        "target_model": target_model,
        "seed": int(seed),
        "generation": generation_cfg,
        "source_path": source_path,
        "n_records": int(n_records),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
