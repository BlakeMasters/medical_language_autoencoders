"""Base-model probing for MedNLA evaluation runs."""

from __future__ import annotations

import gc
import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import transformers
import yaml
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.arch_adapters import resolve_decoder_layers, resolve_text_config
from nla.mednla.answer_parse import parse_answer
from nla.mednla.prompts import apply_chat, build_variant
from nla.mednla.schema import MedItem, Prediction, VariantName

logger = logging.getLogger("nla.mednla.probe")

PROBE_NAME = "pre_answer_last_prompt_token"
_BATCH_SIZE = 64


class ProbeValidationError(RuntimeError):
    """Raised when a probe artifact violates the T3 schema/shape contract."""


def torch_dtype_from_name(dtype: str) -> torch.dtype:
    try:
        value = getattr(torch, dtype)
    except AttributeError as exc:
        raise ValueError(f"unknown torch dtype: {dtype!r}") from exc
    if not isinstance(value, torch.dtype):
        raise ValueError(f"not a torch dtype: {dtype!r}")
    return value


def load_nla_sidecar_d_model(checkpoint_ref: str | Path) -> int:
    """Read d_model from an NLA actor sidecar, supporting local paths and HF IDs."""
    ref = str(checkpoint_ref)
    local_path = Path(ref)
    if local_path.exists():
        meta_path = local_path / "nla_meta.yaml"
        if not meta_path.exists():
            raise FileNotFoundError(f"missing nla_meta.yaml under {local_path}")
    else:
        meta_path = Path(hf_hub_download(repo_id=ref, filename="nla_meta.yaml"))

    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise ValueError(f"invalid nla_meta.yaml at {meta_path}")
    if meta.get("kind") == "nla_dataset":
        return int(meta["extraction"]["d_model"])
    return int(meta["d_model"])


def activation_schema(d_model: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("activation_id", pa.string()),
            pa.field("prediction_id", pa.string()),
            pa.field("item_id", pa.string()),
            pa.field("model_short_name", pa.string()),
            pa.field("layer_index", pa.int64()),
            pa.field("probe", pa.string()),
            pa.field("token_index", pa.int64()),
            pa.field("activation_vector", pa.list_(pa.float32(), list_size=d_model)),
        ]
    )


class ActivationParquetWriter:
    """Buffered fixed-size-list parquet writer for raw activation vectors."""

    def __init__(self, path: str | Path, d_model: int, *, batch_size: int = _BATCH_SIZE) -> None:
        self.path = Path(path)
        self.tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        self.d_model = d_model
        self.batch_size = batch_size
        self.schema = activation_schema(d_model)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = pq.ParquetWriter(self.tmp_path, self.schema)
        self._rows: list[tuple[Prediction, np.ndarray]] = []
        self.count = 0

    def append(self, prediction: Prediction, vector: np.ndarray) -> None:
        if vector.shape != (self.d_model,):
            raise ProbeValidationError(f"activation width {vector.shape} != ({self.d_model},)")
        if vector.dtype != np.float32:
            vector = vector.astype(np.float32, copy=False)
        self._rows.append((prediction, vector))
        if len(self._rows) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self._rows:
            return

        predictions = [row[0] for row in self._rows]
        vectors = np.stack([row[1] for row in self._rows]).astype(np.float32, copy=False)
        flat = pa.array(vectors.reshape(-1), type=pa.float32())
        fsl = pa.FixedSizeListArray.from_arrays(flat, list_size=self.d_model)
        batch = pa.RecordBatch.from_arrays(
            [
                pa.array([pred.activation_id for pred in predictions], type=pa.string()),
                pa.array([pred.prediction_id for pred in predictions], type=pa.string()),
                pa.array([pred.item_id for pred in predictions], type=pa.string()),
                pa.array([pred.model_short_name for pred in predictions], type=pa.string()),
                pa.array([pred.layer_index for pred in predictions], type=pa.int64()),
                pa.array([pred.probe for pred in predictions], type=pa.string()),
                pa.array([pred.token_index for pred in predictions], type=pa.int64()),
                fsl,
            ],
            schema=self.schema,
        )
        self._writer.write_batch(batch)
        self.count += len(self._rows)
        self._rows.clear()

    def close(self) -> None:
        if self._writer is None:
            return
        self.flush()
        self._writer.close()
        self._writer = None

    def commit(self) -> None:
        self.close()
        self.tmp_path.replace(self.path)

    def __enter__(self) -> "ActivationParquetWriter":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class BaseModelProbe:
    def __init__(
        self,
        model_id: str,
        layer_index: int,
        d_model: int,
        *,
        dtype: str = "bfloat16",
        device: str = "cuda",
        seed: int = 42,
    ) -> None:
        self.model_id = model_id
        self.layer_index = layer_index
        self.d_model = int(d_model)
        self.dtype = dtype
        self.device = torch.device(device)
        self.seed = seed
        self._closed = False
        torch.manual_seed(seed)

        torch_dtype = torch_dtype_from_name(dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        self.model.to(self.device)
        self.model.eval()

        text_config = resolve_text_config(self.model.config)
        self._hidden_size = int(text_config.hidden_size)
        if self._hidden_size != self.d_model:
            raise ProbeValidationError(
                f"d_model mismatch: model={self._hidden_size} arg={self.d_model}"
            )

        self.layers = resolve_decoder_layers(self.model)
        if not 0 <= self.layer_index < len(self.layers):
            raise ProbeValidationError(
                f"layer_index {self.layer_index} >= num_layers {len(self.layers)}"
            )
        logger.info(
            "loaded_probe model_id=%s layer_index=%d d_model=%d layer_class=%s",
            self.model_id,
            self.layer_index,
            self.d_model,
            self.layers[self.layer_index].__class__.__name__,
        )

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    def _register_hook(self, captured: dict[str, torch.Tensor]) -> torch.utils.hooks.RemovableHandle:
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            captured["hidden"] = hidden.detach().clone()

        return self.layers[self.layer_index].register_forward_hook(hook)

    def probe_one(
        self,
        item: MedItem,
        variant: VariantName,
        variant_seed: int,
        *,
        max_new_tokens: int,
        model_short_name: str,
    ) -> tuple[Prediction, np.ndarray]:
        if self._closed:
            raise RuntimeError("BaseModelProbe is closed")

        prompt_body, choice_map = build_variant(item, variant, variant_seed)
        prompt_text = apply_chat(self.tokenizer, prompt_body)
        variant_choices, variant_answer_key = _apply_variant_keymap(item, choice_map)
        inputs = self.tokenizer(prompt_text, return_tensors="pt", padding=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        captured: dict[str, torch.Tensor] = {}
        handle = self._register_hook(captured)
        old_use_cache = getattr(self.model.config, "use_cache", None)
        try:
            if old_use_cache is not None:
                self.model.config.use_cache = False
            with torch.inference_mode():
                self.model(**inputs, use_cache=False)
        finally:
            handle.remove()
            if old_use_cache is not None:
                self.model.config.use_cache = old_use_cache

        hidden = captured.get("hidden")
        if hidden is None:
            raise ProbeValidationError(
                f"forward hook on decoder layer {self.layer_index} did not fire"
            )
        if hidden.shape[-1] != self.d_model:
            raise ProbeValidationError(
                f"d_model mismatch: captured={hidden.shape[-1]} arg={self.d_model}"
            )

        token_index = int(inputs["attention_mask"].sum(dim=1).item()) - 1
        vec = hidden[0, token_index, :].float().cpu().numpy().astype(np.float32, copy=False)
        if vec.shape != (self.d_model,):
            raise ProbeValidationError(f"activation shape {vec.shape} != ({self.d_model},)")
        if not np.isfinite(vec).all():
            raise ProbeValidationError("activation vector has NaN/Inf")

        prompt_len = int(inputs["input_ids"].shape[1])
        with torch.inference_mode():
            gen_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated_text = self.tokenizer.decode(
            gen_ids[0, prompt_len:],
            skip_special_tokens=True,
        )
        selected = parse_answer(
            generated_text,
            list(variant_choices.keys()),
            choice_texts=variant_choices,
        )
        prediction_id = _prediction_id(item.item_id, model_short_name, variant, variant_seed)
        prediction = Prediction(
            prediction_id=prediction_id,
            item_id=item.item_id,
            model_id=self.model_id,
            model_short_name=model_short_name,
            layer_index=self.layer_index,
            prompt_variant=variant,
            variant_seed=variant_seed,
            prompt_text=prompt_text,
            generated_text=generated_text,
            selected_answer=selected,
            selected_answer_text=variant_choices.get(selected) if selected else None,
            correct=selected == variant_answer_key if selected is not None else False,
            probe=PROBE_NAME,
            token_index=token_index,
            activation_id=_activation_id(prediction_id),
            original_to_variant_choice_map=choice_map,
            source_metadata={
                "transformers_version": transformers.__version__,
                "torch_version": torch.__version__,
                "model_name_or_path": self.model_id,
                "dtype": self.dtype,
                "seed": self.seed,
                "hf_revision": item.source_metadata.get("hf_revision"),
            },
        )
        return prediction, vec

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        model = getattr(self, "model", None)
        if model is not None:
            self.model = None
        gc.collect()
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> "BaseModelProbe":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def _prediction_id(
    item_id: str,
    model_short_name: str,
    variant: VariantName,
    variant_seed: int,
) -> str:
    return f"{item_id}::{model_short_name}::{variant}::{variant_seed}"


def _activation_id(prediction_id: str) -> str:
    return "act:" + hashlib.sha1(prediction_id.encode()).hexdigest()[:16]


def _apply_variant_keymap(
    item: MedItem,
    original_to_variant_choice_map: dict[str, str],
) -> tuple[dict[str, str], str]:
    choices = {
        variant_key: item.choices[original_key]
        for original_key, variant_key in original_to_variant_choice_map.items()
    }
    choices = dict(sorted(choices.items(), key=lambda pair: pair[0]))
    return choices, original_to_variant_choice_map[item.answer_key]


def assert_activation_parquet(path: str | Path, *, d_model: int, expected_rows: int) -> None:
    table = pq.read_table(path)
    av_type = table.schema.field("activation_vector").type
    if av_type.list_size != d_model:
        raise ProbeValidationError(f"activation_vector list_size {av_type.list_size} != {d_model}")
    if table.num_rows != expected_rows:
        raise ProbeValidationError(f"activation rows {table.num_rows} != predictions {expected_rows}")


__all__ = [
    "ActivationParquetWriter",
    "BaseModelProbe",
    "PROBE_NAME",
    "ProbeValidationError",
    "activation_schema",
    "assert_activation_parquet",
    "load_nla_sidecar_d_model",
    "torch_dtype_from_name",
]
