"""Tests for MedNLA explanation scoring."""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import orjson
import pytest
import torch

import scripts.mednla.score_explanations as score_cli
from nla.mednla.schema import DecodeRecord, MedItem, Prediction, ScoreRecord, from_dict, to_dict
from nla.mednla.scoring import (
    HeuristicScorer,
    JudgeParseError,
    MedGemmaJudge,
    parse_judge_json,
)


def _item() -> MedItem:
    return MedItem(
        item_id="medqa:test:000001",
        dataset="medqa",
        split="test",
        subject="cardiology",
        question="Which drug reduces platelet aggregation in coronary artery disease?",
        choices={"A": "Aspirin", "B": "Ibuprofen", "C": "Warfarin", "D": "Metformin"},
        answer_key="A",
        gold_rationale="Aspirin irreversibly inhibits platelet cyclooxygenase and reduces aggregation.",
        source_metadata={"hf_revision": "abc123"},
    )


def _prediction(
    *,
    prediction_id: str = "pred:0",
    item_id: str = "medqa:test:000001",
    selected_answer: str | None = "A",
    selected_answer_text: str | None = "Aspirin",
    correct: bool = True,
) -> Prediction:
    return Prediction(
        prediction_id=prediction_id,
        item_id=item_id,
        model_id="fake/model",
        model_short_name="fake",
        layer_index=1,
        prompt_variant="canonical",
        variant_seed=0,
        prompt_text="prompt",
        generated_text=selected_answer or "",
        selected_answer=selected_answer,
        selected_answer_text=selected_answer_text,
        correct=correct,
        probe="pre_answer_last_prompt_token",
        token_index=0,
        activation_id=f"act:{prediction_id}",
        original_to_variant_choice_map={"A": "A", "B": "B", "C": "C", "D": "D"},
        source_metadata={},
    )


def _decode(
    *,
    prediction_id: str = "pred:0",
    explanation: str | None = "Aspirin reduces platelet aggregation in coronary artery disease.",
    parse_ok: bool = True,
) -> DecodeRecord:
    return DecodeRecord(
        activation_id=f"act:{prediction_id}",
        prediction_id=prediction_id,
        model_short_name="fake",
        nla_actor="actor",
        nla_critic=None,
        raw_av_text=f"<explanation>{explanation or ''}</explanation>",
        explanation=explanation,
        parse_ok=parse_ok,
        reconstruction_mse=None,
        reconstruction_cos=None,
        decode_warnings=[],
    )


def test_heuristic_failed_parse_taxonomy_honors_correct() -> None:
    score = HeuristicScorer().score(_item(), _prediction(correct=False), _decode(parse_ok=False, explanation=None))

    assert score.medical_relevance == 0
    assert score.answer_support == "unrelated"
    assert score.nla_quality_binary == "weak"
    assert score.taxonomy_cell == "incorrect_weak"
    assert score.scorer_notes == "explanation absent"


def test_heuristic_empty_explanation_is_absent() -> None:
    score = HeuristicScorer().score(_item(), _prediction(), _decode(explanation=""))

    assert score.medical_relevance == 0
    assert score.taxonomy_cell == "correct_weak"
    assert score.shortcut_suspected is True


def test_heuristic_aligned_supports_selected() -> None:
    score = HeuristicScorer().score(
        _item(),
        _prediction(),
        _decode(explanation="Aspirin reduces platelet aggregation in coronary artery disease."),
    )

    assert score.medical_relevance == 2
    assert score.rationale_alignment == 2
    assert score.answer_support == "supports_selected"
    assert score.nla_quality_binary == "aligned"
    assert score.taxonomy_cell == "correct_aligned"


def test_heuristic_shortcut_suspected() -> None:
    score = HeuristicScorer().score(
        _item(),
        _prediction(selected_answer="C", selected_answer_text="Warfarin", correct=False),
        _decode(explanation="The answer is option C because the letter is the best choice."),
    )

    assert score.medical_relevance == 0
    assert score.shortcut_suspected is True
    assert score.taxonomy_cell == "incorrect_weak"


def test_heuristic_supports_other_when_correct_option_overlaps() -> None:
    score = HeuristicScorer().score(
        _item(),
        _prediction(selected_answer="B", selected_answer_text="Ibuprofen", correct=False),
        _decode(explanation="Aspirin reduces platelet aggregation in coronary disease drug therapy."),
    )

    assert score.answer_support == "supports_other"
    assert score.nla_quality_binary == "aligned"
    assert score.taxonomy_cell == "incorrect_aligned"


def test_heuristic_stopword_behavior() -> None:
    weak = HeuristicScorer().score(_item(), _prediction(), _decode(explanation="the and of to in on at"))
    strong = HeuristicScorer().score(
        _item(),
        _prediction(),
        _decode(explanation="drug reduces platelet aggregation coronary artery disease aspirin"),
    )

    assert weak.medical_relevance == 0
    assert strong.medical_relevance == 2


class FakeInputs(dict):
    def to(self, device):
        del device
        return self


class FakeProcessor:
    raw = '{"medical_relevance":2,"rationale_alignment":1,"answer_support":"supports_selected","medically_invalid":false,"shortcut_suspected":false,"evidence":"ok"}'

    @classmethod
    def from_pretrained(cls, model_id):
        del model_id
        return cls()

    def apply_chat_template(self, *args, **kwargs):
        del args, kwargs
        return FakeInputs({"input_ids": torch.tensor([[1, 2]])})

    def decode(self, *args, **kwargs):
        del args, kwargs
        return type(self).raw


class FakeModel:
    device = "cpu"

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        del args, kwargs
        return cls()

    def eval(self):
        return self

    def generate(self, **kwargs):
        del kwargs
        return torch.tensor([[1, 2, 3]])


@pytest.fixture
def fake_transformers(monkeypatch):
    module = types.SimpleNamespace(
        AutoProcessor=FakeProcessor,
        AutoModelForImageTextToText=FakeModel,
        AutoTokenizer=object,
        AutoModelForCausalLM=object,
    )
    monkeypatch.setitem(sys.modules, "transformers", module)
    FakeProcessor.raw = (
        '{"medical_relevance":2,"rationale_alignment":1,"answer_support":"supports_selected",'
        '"medically_invalid":false,"shortcut_suspected":false,"evidence":"ok"}'
    )
    return module


def test_judge_skips_failed_parse_without_model_call(fake_transformers) -> None:
    judge = MedGemmaJudge(device="cpu")
    score = judge.score(_item(), _prediction(), _decode(parse_ok=False, explanation=None))

    assert score.scorer == "medgemma_judge_v1"
    assert score.scorer_notes == "explanation absent"
    assert judge.parse_errors == 0


def test_judge_valid_json_populates_score(fake_transformers) -> None:
    judge = MedGemmaJudge(device="cpu")
    score = judge.score(_item(), _prediction(), _decode())

    assert score.medical_relevance == 2
    assert score.rationale_alignment == 1
    assert score.answer_support == "supports_selected"
    assert score.nla_quality_binary == "aligned"
    assert score.scorer_evidence == "ok"


def test_judge_fenced_json_parses(fake_transformers) -> None:
    FakeProcessor.raw = (
        '```json\n{"medical_relevance":1,"rationale_alignment":null,"answer_support":"unclear",'
        '"medically_invalid":false,"shortcut_suspected":true,"evidence":"fenced"}\n```'
    )
    judge = MedGemmaJudge(device="cpu")
    score = judge.score(_item(), _prediction(), _decode())

    assert score.medical_relevance == 1
    assert score.answer_support == "unclear"
    assert score.nla_quality_binary == "aligned"


def test_judge_malformed_json_failure_record(fake_transformers) -> None:
    FakeProcessor.raw = "not json"
    judge = MedGemmaJudge(device="cpu")
    score = judge.score(_item(), _prediction(), _decode())

    assert score.medical_relevance == 0
    assert score.answer_support == "unclear"
    assert score.scorer_notes.startswith("judge_parse_error:")
    assert judge.parse_errors == 1


def test_judge_unknown_enum_failure_record(fake_transformers) -> None:
    FakeProcessor.raw = (
        '{"medical_relevance":1,"rationale_alignment":1,"answer_support":"sort_of",'
        '"medically_invalid":false,"shortcut_suspected":false,"evidence":"bad enum"}'
    )
    judge = MedGemmaJudge(device="cpu")
    score = judge.score(_item(), _prediction(), _decode())

    assert score.scorer_notes.startswith("judge_parse_error:")
    assert judge.parse_errors == 1


def test_judge_strict_type_and_key_validation() -> None:
    with pytest.raises(JudgeParseError):
        parse_judge_json(
            '{"medical_relevance":true,"rationale_alignment":1,"answer_support":"unclear",'
            '"medically_invalid":false,"shortcut_suspected":false,"evidence":"bad"}'
        )
    with pytest.raises(JudgeParseError):
        parse_judge_json(
            '{"medical_relevance":1,"rationale_alignment":1,"answer_support":"unclear",'
            '"medically_invalid":false,"shortcut_suspected":false}'
        )


def test_judge_close_is_idempotent(fake_transformers) -> None:
    judge = MedGemmaJudge(device="cpu")

    judge.close()
    judge.close()

    assert judge._closed is True


def _write_jsonl(path: Path, rows: list[object]) -> None:
    path.write_bytes(b"".join(orjson.dumps(to_dict(row)) + b"\n" for row in rows))


def _write_config(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "scoring:",
                "  judge_model: google/medgemma-4b-it",
                "  judge_loader: auto",
                "  judge_max_new_tokens: 400",
            ]
        ),
        encoding="utf-8",
    )


def _cli_fixture(tmp_path: Path, *, n: int = 3) -> tuple[Path, Path, Path, Path, Path]:
    config = tmp_path / "config.yaml"
    items_path = tmp_path / "items.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    decodes_path = tmp_path / "decodes.jsonl"
    out = tmp_path / "scores.jsonl"
    _write_config(config)
    item = _item()
    _write_jsonl(items_path, [item])
    preds = [_prediction(prediction_id=f"pred:{idx}") for idx in range(n)]
    decodes = [_decode(prediction_id=f"pred:{idx}") for idx in range(n)]
    _write_jsonl(predictions_path, preds)
    _write_jsonl(decodes_path, decodes)
    return config, items_path, predictions_path, decodes_path, out


def _args(config: Path, items: Path, predictions: Path, decodes: Path, out: Path, **overrides) -> argparse.Namespace:
    defaults = {
        "config": str(config),
        "items": str(items),
        "predictions": str(predictions),
        "decodes": str(decodes),
        "out": str(out),
        "scorer": "heuristic_v1",
        "judge_model": None,
        "judge_loader": None,
        "device": "cpu",
        "dtype": "bfloat16",
        "limit": None,
        "manifest_out": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_cli_heuristic_writes_scores_summary_and_manifest(tmp_path: Path) -> None:
    config, items, predictions, decodes, out = _cli_fixture(tmp_path)
    manifest = tmp_path / "manifest.json"

    assert score_cli._run(_args(config, items, predictions, decodes, out, manifest_out=str(manifest))) == 0

    rows = [from_dict(ScoreRecord, orjson.loads(line)) for line in out.read_bytes().splitlines()]
    summary = orjson.loads((tmp_path / "_summary.json").read_bytes())
    manifest_data = orjson.loads(manifest.read_bytes())
    assert len(rows) == 3
    assert summary["scorer"] == "heuristic_v1"
    assert summary["n_records"] == 3
    assert manifest_data["stage"] == "mednla_score"


class FakeCliJudge:
    name = "medgemma_judge_v1"

    def __init__(self, *args, **kwargs):
        del args, kwargs
        self.calls = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def close(self):
        return None

    def score(self, item, pred, decode):
        del item, decode
        self.calls += 1
        if self.calls <= 2:
            return ScoreRecord(
                prediction_id=pred.prediction_id,
                medical_relevance=0,
                rationale_alignment=None,
                answer_support="unclear",
                medically_invalid=False,
                shortcut_suspected=False,
                nla_quality_binary="weak",
                taxonomy_cell="correct_weak",
                scorer=self.name,
                scorer_notes="judge_parse_error: bad",
                scorer_evidence="",
            )
        return HeuristicScorer().score(_item(), pred, _decode(prediction_id=pred.prediction_id))


def test_cli_judge_parse_error_threshold_exits_7(monkeypatch, tmp_path: Path) -> None:
    config, items, predictions, decodes, out = _cli_fixture(tmp_path)
    monkeypatch.setattr(score_cli, "MedGemmaJudge", FakeCliJudge)

    assert score_cli._run(_args(config, items, predictions, decodes, out, scorer="medgemma_judge_v1")) == 7

    assert out.exists()
    assert len(out.read_bytes().splitlines()) == 3


def test_cli_missing_join_fails_loudly(tmp_path: Path) -> None:
    config, items, predictions, decodes, out = _cli_fixture(tmp_path)
    _write_jsonl(decodes, [_decode(prediction_id="different")])

    with pytest.raises(ValueError, match="missing join"):
        score_cli._run(_args(config, items, predictions, decodes, out))


def test_cli_duplicate_prediction_fails_loudly(tmp_path: Path) -> None:
    config, items, predictions, decodes, out = _cli_fixture(tmp_path)
    duplicate = _prediction(prediction_id="pred:0")
    _write_jsonl(predictions, [duplicate, duplicate])

    with pytest.raises(ValueError, match="duplicate prediction_id"):
        score_cli._run(_args(config, items, predictions, decodes, out))
