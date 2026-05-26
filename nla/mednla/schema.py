"""MedNLA artifact dataclasses and JSON-serialization helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, TypeVar

VariantName = Literal["canonical", "option_shuffle", "compact"]
ProbeName = Literal["pre_answer_last_prompt_token", "first_answer_token", "question_final_token"]
AnswerSupport = Literal["supports_selected", "supports_other", "contradicts_selected", "unrelated", "unclear"]
TaxonomyCell = Literal["correct_aligned", "correct_weak", "incorrect_aligned", "incorrect_weak"]

T = TypeVar("T")


@dataclass(slots=True, frozen=True)
class MedItem:
    item_id: str
    dataset: str
    split: str
    subject: str | None
    question: str
    choices: dict[str, str]
    answer_key: str
    gold_rationale: str | None
    source_metadata: dict[str, Any]


@dataclass(slots=True, frozen=True)
class Prediction:
    prediction_id: str
    item_id: str
    model_id: str
    model_short_name: str
    layer_index: int
    prompt_variant: VariantName
    variant_seed: int
    prompt_text: str
    generated_text: str
    selected_answer: str | None
    selected_answer_text: str | None
    correct: bool
    probe: ProbeName
    token_index: int
    activation_id: str
    original_to_variant_choice_map: dict[str, str]
    source_metadata: dict[str, Any]


@dataclass(slots=True, frozen=True)
class DecodeRecord:
    activation_id: str
    prediction_id: str
    model_short_name: str
    nla_actor: str
    nla_critic: str | None
    raw_av_text: str
    explanation: str | None
    parse_ok: bool
    reconstruction_mse: float | None
    reconstruction_cos: float | None
    decode_warnings: list[str]


@dataclass(slots=True, frozen=True)
class ScoreRecord:
    prediction_id: str
    medical_relevance: int
    rationale_alignment: int | None
    answer_support: AnswerSupport
    medically_invalid: bool
    shortcut_suspected: bool
    nla_quality_binary: Literal["aligned", "weak"]
    taxonomy_cell: TaxonomyCell
    scorer: str
    scorer_notes: str
    scorer_evidence: str


_FIELD_WHITELISTS: dict[type[Any], tuple[str, ...]] = {
    MedItem: (
        "item_id",
        "dataset",
        "split",
        "subject",
        "question",
        "choices",
        "answer_key",
        "gold_rationale",
        "source_metadata",
    ),
    Prediction: (
        "prediction_id",
        "item_id",
        "model_id",
        "model_short_name",
        "layer_index",
        "prompt_variant",
        "variant_seed",
        "prompt_text",
        "generated_text",
        "selected_answer",
        "selected_answer_text",
        "correct",
        "probe",
        "token_index",
        "activation_id",
        "original_to_variant_choice_map",
        "source_metadata",
    ),
    DecodeRecord: (
        "activation_id",
        "prediction_id",
        "model_short_name",
        "nla_actor",
        "nla_critic",
        "raw_av_text",
        "explanation",
        "parse_ok",
        "reconstruction_mse",
        "reconstruction_cos",
        "decode_warnings",
    ),
    ScoreRecord: (
        "prediction_id",
        "medical_relevance",
        "rationale_alignment",
        "answer_support",
        "medically_invalid",
        "shortcut_suspected",
        "nla_quality_binary",
        "taxonomy_cell",
        "scorer",
        "scorer_notes",
        "scorer_evidence",
    ),
}


def to_dict(obj: Any) -> dict[str, Any]:
    if type(obj) not in _FIELD_WHITELISTS:
        raise TypeError(f"unsupported type for to_dict: {type(obj)!r}")
    return asdict(obj)


def from_dict(cls: type[T], data: dict[str, Any]) -> T:
    if cls not in _FIELD_WHITELISTS:
        raise TypeError(f"unsupported type for from_dict: {cls!r}")
    fields = _FIELD_WHITELISTS[cls]
    for key in data:
        if key not in fields:
            raise ValueError(key)
    for key in fields:
        if key not in data:
            raise ValueError(key)
    return cls(**{key: data[key] for key in fields})  # type: ignore[call-arg, return-value]


__all__ = [
    "AnswerSupport",
    "DecodeRecord",
    "MedItem",
    "Prediction",
    "ProbeName",
    "ScoreRecord",
    "TaxonomyCell",
    "VariantName",
    "from_dict",
    "to_dict",
]
