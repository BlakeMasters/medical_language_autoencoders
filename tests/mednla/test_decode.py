"""Tests for MedNLA NLA decode streaming."""

from __future__ import annotations

import argparse
import types
from pathlib import Path

import numpy as np
import orjson
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import nla.mednla.decode as decode_mod
import scripts.mednla.run_nla_decode as decode_cli
from nla.mednla.decode import (
    WARNING_CJK_INJECTION_FAILURE,
    WARNING_EMPTY_RAW_OUTPUT,
    WARNING_MISSING_EXPLANATION_TAGS,
    WARNING_VERY_SHORT_EXPLANATION,
    DecodeSchemaError,
    MedNLADecoder,
    resolve_checkpoint_path,
)
from nla.mednla.schema import DecodeRecord


D_MODEL = 8


class FakeClient:
    cfg = types.SimpleNamespace(
        d_model=D_MODEL,
        injection_char="\u320e",
        injection_scale=10.0,
    )

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self._http = types.SimpleNamespace(headers={}, close=lambda: None)

    def generate(self, vec, *, extract_explanation, **kwargs):
        del kwargs
        assert extract_explanation is False
        v0 = int(float(vec[0]))
        return {
            0: "<explanation>Aspirin reduces platelet aggregation.</explanation>",
            1: "no tags here",
            2: "\u5b8c\u5168\u4e2d\u6587\u8f93\u51fa\u8868\u793a\u6ce8\u5165\u5931\u8d25",
            3: "<explanation>x</explanation>",
            4: "",
        }[v0]


class FakeCritic:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def score(self, explanation, vec):
        del explanation, vec
        return 0.2, 0.9


@pytest.fixture(autouse=True)
def fake_nla(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(decode_mod, "NLAClient", FakeClient)
    monkeypatch.setattr(decode_mod, "NLACritic", FakeCritic)
    monkeypatch.setattr(decode_mod, "resolve_checkpoint_path", lambda ref: Path(str(ref)))


def _write_activation_parquet(path: Path, first_values: list[int], *, d_model: int = D_MODEL) -> None:
    schema = pa.schema(
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
    vectors = np.zeros((len(first_values), d_model), dtype=np.float32)
    vectors[:, 0] = np.asarray(first_values, dtype=np.float32)
    fsl = pa.FixedSizeListArray.from_arrays(
        pa.array(vectors.reshape(-1), type=pa.float32()),
        list_size=d_model,
    )
    batch = pa.RecordBatch.from_arrays(
        [
            pa.array([f"act:{idx}" for idx in range(len(first_values))]),
            pa.array([f"pred:{idx}" for idx in range(len(first_values))]),
            pa.array([f"item:{idx}" for idx in range(len(first_values))]),
            pa.array(["fake"] * len(first_values)),
            pa.array([1] * len(first_values), type=pa.int64()),
            pa.array(["pre_answer_last_prompt_token"] * len(first_values)),
            pa.array(list(range(len(first_values))), type=pa.int64()),
            fsl,
        ],
        schema=schema,
    )
    with pq.ParquetWriter(path, schema) as writer:
        writer.write_batch(batch)


def test_decode_batch_happy_path_with_critic(tmp_path: Path) -> None:
    path = tmp_path / "activations.parquet"
    _write_activation_parquet(path, [0])

    decoder = MedNLADecoder(
        av_path="actor",
        ar_path="critic",
        sglang_url="http://localhost:30000",
        ar_device="cpu",
    )
    records = list(decoder.decode_batch(path))

    assert len(records) == 1
    record = records[0]
    assert record.parse_ok is True
    assert record.explanation == "Aspirin reduces platelet aggregation."
    assert record.decode_warnings == []
    assert record.reconstruction_mse == pytest.approx(0.2)
    assert record.reconstruction_cos == pytest.approx(0.9)


def test_decode_batch_missing_tags_and_cjk_and_short_and_empty(tmp_path: Path) -> None:
    path = tmp_path / "activations.parquet"
    _write_activation_parquet(path, [1, 2, 3, 4])

    decoder = MedNLADecoder(
        av_path="actor",
        ar_path=None,
        sglang_url="http://localhost:30000",
    )
    records = list(decoder.decode_batch(path))

    assert records[0].parse_ok is False
    assert records[0].explanation is None
    assert WARNING_MISSING_EXPLANATION_TAGS in records[0].decode_warnings
    assert WARNING_CJK_INJECTION_FAILURE in records[1].decode_warnings
    assert WARNING_VERY_SHORT_EXPLANATION in records[2].decode_warnings
    assert WARNING_EMPTY_RAW_OUTPUT in records[3].decode_warnings


def test_no_critic_leaves_reconstruction_fields_none(tmp_path: Path) -> None:
    path = tmp_path / "activations.parquet"
    _write_activation_parquet(path, [0])

    decoder = MedNLADecoder(
        av_path="actor",
        ar_path=None,
        sglang_url="http://localhost:30000",
    )
    record = next(decoder.decode_batch(path))

    assert record.reconstruction_mse is None
    assert record.reconstruction_cos is None


def test_decoder_applies_sglang_auth_token() -> None:
    decoder = MedNLADecoder(
        av_path="actor",
        ar_path=None,
        sglang_url="http://localhost:30000",
        sglang_auth_token="secret-token",
    )

    assert decoder.client._http.headers["Authorization"] == "Bearer secret-token"


def test_parquet_schema_mismatch_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.parquet"
    _write_activation_parquet(path, [0], d_model=D_MODEL + 1)
    decoder = MedNLADecoder(
        av_path="actor",
        ar_path=None,
        sglang_url="http://localhost:30000",
    )

    with pytest.raises(DecodeSchemaError, match="d_model"):
        list(decoder.decode_batch(path))


def test_limit_preserves_parquet_order(tmp_path: Path) -> None:
    path = tmp_path / "activations.parquet"
    _write_activation_parquet(path, [0, 1, 3])
    decoder = MedNLADecoder(
        av_path="actor",
        ar_path=None,
        sglang_url="http://localhost:30000",
    )

    records = list(decoder.decode_batch(path, batch_size=2, limit=2))

    assert [record.activation_id for record in records] == ["act:0", "act:1"]


def test_resolve_checkpoint_path_uses_snapshot_for_hf_ids(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    local = tmp_path / "local"
    local.mkdir()
    assert resolve_checkpoint_path(local) == local

    monkeypatch.setattr(decode_mod, "snapshot_download", lambda ref: str(tmp_path / ref.replace("/", "--")))
    assert resolve_checkpoint_path("owner/repo") == tmp_path / "owner--repo"


def _config(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "models:",
                "  - short_name: fake",
                "    nla_actor: actor",
                "    nla_critic: critic",
                "nla_decode:",
                "  sglang_url: http://localhost:30000",
                "  temperature: 0.7",
                "  max_new_tokens: 200",
                "  batch_size: 2",
            ]
        ),
        encoding="utf-8",
    )


class FakeCliDecoder:
    def __init__(self, *args, **kwargs):
        del args, kwargs

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def decode_batch(self, *args, **kwargs):
        del args, kwargs
        yield DecodeRecord(
            activation_id="act:0",
            prediction_id="pred:0",
            model_short_name="fake",
            nla_actor="actor",
            nla_critic=None,
            raw_av_text="<explanation>ok</explanation>",
            explanation="ok",
            parse_ok=True,
            reconstruction_mse=None,
            reconstruction_cos=None,
            decode_warnings=[],
        )


def _args(config: Path, activations: Path, out: Path, **overrides) -> argparse.Namespace:
    defaults = {
        "config": str(config),
        "model": "fake",
        "activations": str(activations),
        "out": str(out),
        "no_critic": True,
        "allow_cjk_warnings": False,
        "sglang_url": None,
        "auth_token": None,
        "auth_token_env": None,
        "limit": None,
        "batch_size": None,
        "manifest_out": None,
        "ar_device": "cpu",
        "ar_dtype": "bfloat16",
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_cli_atomic_write_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    activations = tmp_path / "activations.parquet"
    out = tmp_path / "decodes.jsonl"
    _config(config)
    _write_activation_parquet(activations, [0])
    monkeypatch.setattr(decode_cli, "MedNLADecoder", FakeCliDecoder)

    assert decode_cli._run(_args(config, activations, out)) == 0

    assert out.exists()
    assert not out.with_suffix(out.suffix + ".tmp").exists()
    row = orjson.loads(out.read_bytes().splitlines()[0])
    assert row["prediction_id"] == "pred:0"


def test_cli_sglang_url_and_auth_token_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class CapturingDecoder(FakeCliDecoder):
        kwargs = None

        def __init__(self, *args, **kwargs):
            del args
            type(self).kwargs = kwargs

    config = tmp_path / "config.yaml"
    activations = tmp_path / "activations.parquet"
    out = tmp_path / "decodes.jsonl"
    _config(config)
    _write_activation_parquet(activations, [0])
    monkeypatch.setenv("SGLANG_TOKEN", "secret-from-env")
    monkeypatch.setattr(decode_cli, "MedNLADecoder", CapturingDecoder)

    assert decode_cli._run(
        _args(
            config,
            activations,
            out,
            sglang_url="http://127.0.0.1:18000",
            auth_token_env="SGLANG_TOKEN",
        )
    ) == 0

    assert CapturingDecoder.kwargs["sglang_url"] == "http://127.0.0.1:18000"
    assert CapturingDecoder.kwargs["sglang_auth_token"] == "secret-from-env"


def test_cli_cjk_abort_leaves_tmp_and_no_final(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class CjkDecoder(FakeCliDecoder):
        def decode_batch(self, *args, **kwargs):
            del args, kwargs
            yield DecodeRecord(
                activation_id="act:0",
                prediction_id="pred:0",
                model_short_name="fake",
                nla_actor="actor",
                nla_critic=None,
                raw_av_text="\u5b8c\u5168\u4e2d\u6587",
                explanation=None,
                parse_ok=False,
                reconstruction_mse=None,
                reconstruction_cos=None,
                decode_warnings=[WARNING_CJK_INJECTION_FAILURE],
            )

    config = tmp_path / "config.yaml"
    activations = tmp_path / "activations.parquet"
    out = tmp_path / "decodes.jsonl"
    _config(config)
    _write_activation_parquet(activations, [0])
    monkeypatch.setattr(decode_cli, "MedNLADecoder", CjkDecoder)

    assert decode_cli._run(_args(config, activations, out)) == 6
    assert not out.exists()
    assert out.with_suffix(out.suffix + ".tmp").exists()


def test_cli_simulated_crash_leaves_tmp_and_no_final(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class CrashingDecoder(FakeCliDecoder):
        def decode_batch(self, *args, **kwargs):
            del args, kwargs
            yield from super().decode_batch()
            raise RuntimeError("boom")

    config = tmp_path / "config.yaml"
    activations = tmp_path / "activations.parquet"
    out = tmp_path / "decodes.jsonl"
    _config(config)
    _write_activation_parquet(activations, [0])
    monkeypatch.setattr(decode_cli, "MedNLADecoder", CrashingDecoder)

    with pytest.raises(RuntimeError, match="boom"):
        decode_cli._run(_args(config, activations, out))
    assert not out.exists()
    assert out.with_suffix(out.suffix + ".tmp").exists()
