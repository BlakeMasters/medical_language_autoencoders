"""Tests for MedNLA base-model probing."""

from __future__ import annotations

import hashlib
import types

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

import nla.mednla.probe as probe_mod
from nla.mednla.probe import (
    ActivationParquetWriter,
    BaseModelProbe,
    ProbeValidationError,
    load_nla_sidecar_d_model,
    torch_dtype_from_name,
)
from nla.mednla.schema import MedItem, Prediction


HIDDEN_SIZE = 8


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 0
    pad_token = "<pad>"
    eos_token = "<eos>"
    padding_side = "right"
    truncation_side = "right"

    def apply_chat_template(self, messages, *, tokenize: bool, add_generation_prompt: bool):
        assert add_generation_prompt is True
        text = messages[0]["content"] + "\n<assistant>"
        if tokenize:
            return list(range(1, len(text.split()) + 1))
        return text

    def __call__(self, text: str, *, return_tensors: str, padding: bool):
        del return_tensors, padding
        token_count = len(text.split())
        input_ids = torch.arange(1, token_count + 1, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, token_ids, *, skip_special_tokens: bool):
        del skip_special_tokens
        ids = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
        return "C" if 67 in ids else ""


class FakeLayer(torch.nn.Module):
    def __init__(self, offset: float) -> None:
        super().__init__()
        self.offset = offset
        self.call_count = 0

    def forward(self, hidden):
        self.call_count += 1
        return hidden + self.offset


class FakeBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([FakeLayer(1.0), FakeLayer(2.0)])


class FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=HIDDEN_SIZE, use_cache=True)
        self.model = FakeBackbone()

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        del attention_mask, use_cache
        hidden = input_ids.float().unsqueeze(-1).repeat(1, 1, HIDDEN_SIZE)
        for layer in self.model.layers:
            hidden = layer(hidden)
        return types.SimpleNamespace(last_hidden_state=hidden)

    def generate(self, input_ids, attention_mask=None, max_new_tokens=8, do_sample=False, pad_token_id=None):
        del attention_mask, max_new_tokens, do_sample, pad_token_id
        answer = torch.full((input_ids.shape[0], 1), 67, dtype=input_ids.dtype, device=input_ids.device)
        return torch.cat([input_ids, answer], dim=1)


@pytest.fixture
def item() -> MedItem:
    return MedItem(
        item_id="medqa:test:000001",
        dataset="medqa",
        split="test",
        subject="cardiology",
        question="Which drug reduces platelet aggregation?",
        choices={"A": "Aspirin", "B": "Ibuprofen", "C": "Warfarin", "D": "Metformin"},
        answer_key="A",
        gold_rationale=None,
        source_metadata={"hf_revision": "abc123"},
    )


@pytest.fixture
def fake_probe(monkeypatch: pytest.MonkeyPatch) -> BaseModelProbe:
    monkeypatch.setattr(probe_mod.AutoTokenizer, "from_pretrained", lambda *a, **k: FakeTokenizer())
    monkeypatch.setattr(probe_mod.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: FakeModel())
    return BaseModelProbe(
        "fake/model",
        layer_index=1,
        d_model=HIDDEN_SIZE,
        dtype="float32",
        device="cpu",
        seed=123,
    )


def test_probe_one_captures_last_prompt_token_vector(
    fake_probe: BaseModelProbe,
    item: MedItem,
) -> None:
    prediction, vec = fake_probe.probe_one(
        item,
        "canonical",
        0,
        max_new_tokens=8,
        model_short_name="fake",
    )

    assert fake_probe.layers[1].call_count == 1
    assert len(fake_probe.layers[1]._forward_hooks) == 0
    assert vec.shape == (HIDDEN_SIZE,)
    assert vec.dtype == np.float32
    assert np.isfinite(vec).all()
    assert prediction.token_index == len(prediction.prompt_text.split()) - 1
    assert prediction.probe == "pre_answer_last_prompt_token"


def test_hidden_size_property(fake_probe: BaseModelProbe) -> None:
    assert fake_probe.hidden_size == HIDDEN_SIZE


def test_prediction_and_activation_ids_are_deterministic(
    fake_probe: BaseModelProbe,
    item: MedItem,
) -> None:
    prediction, _ = fake_probe.probe_one(
        item,
        "canonical",
        0,
        max_new_tokens=8,
        model_short_name="fake",
    )

    assert prediction.prediction_id == "medqa:test:000001::fake::canonical::0"
    expected = hashlib.sha1(prediction.prediction_id.encode()).hexdigest()[:16]
    assert prediction.activation_id == f"act:{expected}"
    assert prediction.original_to_variant_choice_map == {"A": "A", "B": "B", "C": "C", "D": "D"}


def test_torch_dtype_from_name_rejects_unknown_dtype() -> None:
    assert torch_dtype_from_name("float32") == torch.float32
    with pytest.raises(ValueError, match="unknown torch dtype"):
        torch_dtype_from_name("not_a_dtype")


def test_probe_rejects_d_model_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(probe_mod.AutoTokenizer, "from_pretrained", lambda *a, **k: FakeTokenizer())
    monkeypatch.setattr(probe_mod.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: FakeModel())

    with pytest.raises(ProbeValidationError, match="d_model mismatch"):
        BaseModelProbe(
            "fake/model",
            layer_index=1,
            d_model=HIDDEN_SIZE + 1,
            dtype="float32",
            device="cpu",
        )


def test_correctness_uses_variant_remapped_answer_key(
    fake_probe: BaseModelProbe,
    item: MedItem,
) -> None:
    canonical_pred, _ = fake_probe.probe_one(
        item,
        "canonical",
        0,
        max_new_tokens=8,
        model_short_name="fake",
    )
    shuffled_pred, _ = fake_probe.probe_one(
        item,
        "option_shuffle",
        0,
        max_new_tokens=8,
        model_short_name="fake",
    )

    assert canonical_pred.selected_answer == "C"
    assert canonical_pred.correct is False
    assert shuffled_pred.original_to_variant_choice_map["A"] == "C"
    assert shuffled_pred.selected_answer == "C"
    assert shuffled_pred.selected_answer_text == "Aspirin"
    assert shuffled_pred.correct is True


def test_close_is_idempotent_and_hooks_are_removed(
    fake_probe: BaseModelProbe,
    item: MedItem,
) -> None:
    fake_probe.probe_one(item, "canonical", 0, max_new_tokens=8, model_short_name="fake")
    assert len(fake_probe.layers[1]._forward_hooks) == 0

    fake_probe.close()
    fake_probe.close()
    assert len(fake_probe.layers[1]._forward_hooks) == 0


def test_load_local_sidecar_d_model(tmp_path) -> None:
    ckpt = tmp_path / "actor"
    ckpt.mkdir()
    (ckpt / "nla_meta.yaml").write_text(
        "\n".join(
            [
                "kind: nla_model",
                "d_model: 1234",
            ]
        ),
        encoding="utf-8",
    )

    assert load_nla_sidecar_d_model(ckpt) == 1234


def test_activation_writer_rejects_wrong_vector_width(tmp_path) -> None:
    path = tmp_path / "bad.parquet"
    prediction = _prediction("item::0")

    with ActivationParquetWriter(path, HIDDEN_SIZE) as writer:
        with pytest.raises(ProbeValidationError, match="activation width"):
            writer.append(prediction, np.zeros((HIDDEN_SIZE + 1,), dtype=np.float32))


def test_activation_parquet_roundtrip(tmp_path) -> None:
    path = tmp_path / "activations.parquet"
    predictions = [_prediction(f"item::{idx}", token_index=idx) for idx in range(3)]

    with ActivationParquetWriter(path, HIDDEN_SIZE, batch_size=2) as writer:
        for idx, prediction in enumerate(predictions):
            writer.append(prediction, np.full((HIDDEN_SIZE,), idx, dtype=np.float32))
        writer.commit()

    table = pq.read_table(path)
    av_type = table.schema.field("activation_vector").type
    assert av_type.list_size == HIDDEN_SIZE
    assert av_type.value_type == pa.float32()
    assert table.num_rows == 3
    first = np.asarray(table.column("activation_vector")[0].as_py(), dtype=np.float32)
    assert first.shape == (HIDDEN_SIZE,)
    assert first.dtype == np.float32


def _prediction(prediction_id: str, *, token_index: int = 0) -> Prediction:
    return Prediction(
        prediction_id=prediction_id,
        item_id=prediction_id,
        model_id="fake/model",
        model_short_name="fake",
        layer_index=1,
        prompt_variant="canonical",
        variant_seed=0,
        prompt_text="prompt",
        generated_text="A",
        selected_answer="A",
        selected_answer_text="Choice A",
        correct=True,
        probe="pre_answer_last_prompt_token",
        token_index=token_index,
        activation_id=f"act:{token_index}",
        original_to_variant_choice_map={"A": "A"},
        source_metadata={},
    )
