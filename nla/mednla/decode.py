"""Decode MedNLA activation parquet rows with released NLA checkpoints."""

from __future__ import annotations

import gc
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import snapshot_download

from nla.mednla.schema import DecodeRecord
from nla_inference import EXPLANATION_RE, NLAClient, NLACritic

logger = logging.getLogger("nla.mednla.decode")

WARNING_MISSING_EXPLANATION_TAGS = "missing_explanation_tags"
WARNING_CJK_INJECTION_FAILURE = "cjk_injection_failure"
WARNING_INJECTION_MARKER_DESCRIBED = "injection_marker_described"
WARNING_LOW_RECONSTRUCTION_COSINE = "low_reconstruction_cosine"
WARNING_VERY_SHORT_EXPLANATION = "very_short_explanation"
WARNING_EMPTY_RAW_OUTPUT = "empty_raw_output"

DECODE_WARNING_SET = frozenset(
    {
        WARNING_MISSING_EXPLANATION_TAGS,
        WARNING_CJK_INJECTION_FAILURE,
        WARNING_INJECTION_MARKER_DESCRIBED,
        WARNING_LOW_RECONSTRUCTION_COSINE,
        WARNING_VERY_SHORT_EXPLANATION,
        WARNING_EMPTY_RAW_OUTPUT,
    }
)

_PARQUET_META_COLUMNS = (
    "activation_id",
    "prediction_id",
    "item_id",
    "model_short_name",
    "layer_index",
    "probe",
)


class DecodeSchemaError(RuntimeError):
    """Raised when an activation parquet is incompatible with the NLA sidecar."""


def resolve_checkpoint_path(checkpoint_ref: str | Path) -> Path:
    """Return a local checkpoint directory for a local path or Hugging Face ID."""
    path = Path(str(checkpoint_ref))
    if path.exists():
        return path
    return Path(snapshot_download(str(checkpoint_ref)))


def torch_dtype_from_name(dtype: str) -> torch.dtype:
    try:
        value = getattr(torch, dtype)
    except AttributeError as exc:
        raise ValueError(f"unknown torch dtype: {dtype!r}") from exc
    if not isinstance(value, torch.dtype):
        raise ValueError(f"not a torch dtype: {dtype!r}")
    return value


class MedNLADecoder:
    def __init__(
        self,
        *,
        av_path: str,
        ar_path: str | None,
        sglang_url: str,
        ar_device: str = "cuda",
        ar_dtype: str = "bfloat16",
    ) -> None:
        self.av_path = av_path
        self.ar_path = ar_path
        self.av_checkpoint_path = resolve_checkpoint_path(av_path)
        self.ar_checkpoint_path = resolve_checkpoint_path(ar_path) if ar_path else None
        self.client = NLAClient(self.av_checkpoint_path, sglang_url=sglang_url, device="cpu")
        self.critic = (
            NLACritic(
                self.ar_checkpoint_path,
                device=ar_device,
                dtype=torch_dtype_from_name(ar_dtype),
            )
            if self.ar_checkpoint_path is not None
            else None
        )
        self.injection_char = self.client.cfg.injection_char
        self._closed = False

    @property
    def d_model(self) -> int:
        return int(self.client.cfg.d_model)

    def decode_batch(
        self,
        activations_path: str | Path,
        *,
        batch_size: int = 16,
        temperature: float = 0.7,
        max_new_tokens: int = 200,
        cjk_threshold: float = 0.30,
        limit: int | None = None,
    ) -> Iterator[DecodeRecord]:
        if self._closed:
            raise RuntimeError("MedNLADecoder is closed")

        pf = pq.ParquetFile(activations_path)
        field_names = set(pf.schema_arrow.names)
        missing = {"activation_vector", *_PARQUET_META_COLUMNS} - field_names
        if missing:
            raise DecodeSchemaError(f"activations parquet missing columns: {sorted(missing)}")

        av_type = pf.schema_arrow.field("activation_vector").type
        if not hasattr(av_type, "list_size") or av_type.list_size != self.d_model:
            actual = getattr(av_type, "list_size", av_type)
            raise DecodeSchemaError(
                f"activations parquet d_model={actual} != AV sidecar d_model={self.d_model}"
            )

        yielded = 0
        for batch in pf.iter_batches(batch_size=batch_size):
            av_col = batch.column("activation_vector")
            flat = av_col.flatten().to_numpy(zero_copy_only=False).astype(np.float32)
            vectors = flat.reshape(len(av_col), self.d_model)
            meta_columns = {
                name: batch.column(name).to_pylist()
                for name in _PARQUET_META_COLUMNS
            }
            for row_idx, vec in enumerate(vectors):
                if limit is not None and yielded >= limit:
                    return
                meta = {name: meta_columns[name][row_idx] for name in _PARQUET_META_COLUMNS}
                yield self._decode_one(
                    meta,
                    vec,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    cjk_threshold=cjk_threshold,
                )
                yielded += 1

    def _decode_one(
        self,
        meta: dict[str, Any],
        vec: np.ndarray,
        *,
        temperature: float,
        max_new_tokens: int,
        cjk_threshold: float,
    ) -> DecodeRecord:
        warnings: list[str] = []
        raw_text = self.client.generate(
            vec,
            extract_explanation=False,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

        if not raw_text.strip():
            warnings.append(WARNING_EMPTY_RAW_OUTPUT)

        match = EXPLANATION_RE.search(raw_text)
        if match is None:
            explanation = None
            parse_ok = False
            warnings.append(WARNING_MISSING_EXPLANATION_TAGS)
        else:
            explanation = match.group(1).strip()
            parse_ok = True
            if len(explanation) < 10:
                warnings.append(WARNING_VERY_SHORT_EXPLANATION)

        if _cjk_fraction(raw_text) >= cjk_threshold:
            warnings.append(WARNING_CJK_INJECTION_FAILURE)

        if self.injection_char and self.injection_char in raw_text:
            warnings.append(WARNING_INJECTION_MARKER_DESCRIBED)

        if self.critic is not None and explanation:
            mse, cos = self.critic.score(explanation, vec)
            reconstruction_mse = float(mse)
            reconstruction_cos = float(cos)
            if reconstruction_cos < 0.10:
                warnings.append(WARNING_LOW_RECONSTRUCTION_COSINE)
        else:
            reconstruction_mse = None
            reconstruction_cos = None

        _assert_closed_warning_set(warnings)
        return DecodeRecord(
            activation_id=str(meta["activation_id"]),
            prediction_id=str(meta["prediction_id"]),
            model_short_name=str(meta["model_short_name"]),
            nla_actor=self.av_path,
            nla_critic=self.ar_path,
            raw_av_text=raw_text,
            explanation=explanation,
            parse_ok=parse_ok,
            reconstruction_mse=reconstruction_mse,
            reconstruction_cos=reconstruction_cos,
            decode_warnings=warnings,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        http_client = getattr(getattr(self, "client", None), "_http", None)
        if http_client is not None:
            http_client.close()
        self.client = None
        self.critic = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> "MedNLADecoder":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def _cjk_fraction(text: str) -> float:
    if not text:
        return 0.0
    cjk = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    return cjk / len(text)


def _assert_closed_warning_set(warnings: list[str]) -> None:
    unknown = set(warnings) - DECODE_WARNING_SET
    if unknown:
        raise AssertionError(f"unknown decode warning(s): {sorted(unknown)}")


__all__ = [
    "DECODE_WARNING_SET",
    "DecodeSchemaError",
    "MedNLADecoder",
    "WARNING_CJK_INJECTION_FAILURE",
    "WARNING_EMPTY_RAW_OUTPUT",
    "WARNING_INJECTION_MARKER_DESCRIBED",
    "WARNING_LOW_RECONSTRUCTION_COSINE",
    "WARNING_MISSING_EXPLANATION_TAGS",
    "WARNING_VERY_SHORT_EXPLANATION",
    "resolve_checkpoint_path",
    "torch_dtype_from_name",
]
