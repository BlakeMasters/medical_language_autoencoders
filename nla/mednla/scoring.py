"""Score MedNLA decoded explanations with heuristic and local judge paths."""

from __future__ import annotations

import gc
import re
from typing import Any, Literal

import orjson
import torch

from nla.mednla.schema import AnswerSupport, DecodeRecord, MedItem, Prediction, ScoreRecord, TaxonomyCell

NLAQuality = Literal["aligned", "weak"]
JudgeLoader = Literal["auto", "image_text", "causal_lm"]

_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "of",
        "to",
        "in",
        "on",
        "at",
        "and",
        "or",
        "but",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "with",
        "for",
        "by",
        "from",
        "as",
        "not",
        "no",
        "yes",
        "i",
        "you",
        "he",
        "she",
        "they",
        "we",
        "him",
        "her",
        "them",
        "us",
        "my",
        "your",
        "his",
        "their",
        "our",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "will",
        "would",
        "can",
        "could",
        "should",
        "may",
        "might",
        "if",
        "then",
        "than",
        "also",
        "such",
        "which",
        "who",
        "whose",
        "what",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "very",
        "just",
        "only",
        "into",
        "through",
        "over",
        "under",
        "between",
        "about",
    }
)

_SHORTCUT_MARKERS = ("the answer", "option", "letter", "best choice", "test answer")
_ANSWER_SUPPORT_VALUES = frozenset(
    {"supports_selected", "supports_other", "contradicts_selected", "unrelated", "unclear"}
)
_JUDGE_REQUIRED_KEYS = frozenset(
    {
        "medical_relevance",
        "rationale_alignment",
        "answer_support",
        "medically_invalid",
        "shortcut_suspected",
        "evidence",
    }
)

_JUDGE_PROMPT = """You are scoring an interpretability artifact for a medical QA evaluation.

The NLA explanation is NOT a chain of thought from the answering model. It is decoded from a single latent activation and may be noisy or confabulated. Score whether it contains medically relevant content aligned with the benchmark item. Do not reward generic medical-sounding text unless it connects to the question.

Question:
{question}

Choices:
{choices_block}

Correct answer: {answer_key}. {answer_text}
Model selected answer: {selected_answer_or_NONE}. {selected_answer_text_or_empty}
Gold rationale:
{gold_rationale_or_NONE}

NLA explanation:
{explanation}

Return ONLY a single JSON object (no prose, no markdown fences) with these keys:
{{
  "medical_relevance": 0 | 1 | 2,
  "rationale_alignment": 0 | 1 | 2 | null,
  "answer_support": "supports_selected" | "supports_other" | "contradicts_selected" | "unrelated" | "unclear",
  "medically_invalid": true | false,
  "shortcut_suspected": true | false,
  "evidence": "<<=240 chars>"
}}"""


class JudgeParseError(ValueError):
    """Raised when the MedGemma judge output is not the strict rubric JSON."""


def apply_aligned_rule(
    med_rel: int,
    med_invalid: bool,
    ans_sup: AnswerSupport,
    rat_align: int | None,
) -> NLAQuality:
    if (
        med_rel >= 1
        and med_invalid is False
        and ans_sup in {"supports_selected", "supports_other", "unclear"}
        and (rat_align is None or rat_align >= 1)
    ):
        return "aligned"
    return "weak"


def taxonomy_cell(correct: bool, quality: NLAQuality) -> TaxonomyCell:
    if correct:
        return "correct_aligned" if quality == "aligned" else "correct_weak"
    return "incorrect_aligned" if quality == "aligned" else "incorrect_weak"


class HeuristicScorer:
    name = "heuristic_v1"

    def score(self, item: MedItem, pred: Prediction, decode: DecodeRecord) -> ScoreRecord:
        explanation = (decode.explanation or "").strip()
        if not decode.parse_ok or not explanation:
            return _absent_score(pred, scorer=self.name)

        explanation_tokens = _concept_tokens(explanation)
        question_tokens = _concept_tokens(item.question)
        rationale_tokens = _concept_tokens(item.gold_rationale or "")
        selected_tokens = _concept_tokens(pred.selected_answer_text or "")
        correct_tokens = _concept_tokens(item.choices.get(item.answer_key, ""))

        q_overlap = len(explanation_tokens & question_tokens)
        r_overlap = len(explanation_tokens & rationale_tokens) if item.gold_rationale else None
        sel_overlap = len(explanation_tokens & selected_tokens)
        cor_overlap = len(explanation_tokens & correct_tokens)

        if q_overlap < 2:
            medical_relevance = 0
        elif q_overlap < 5:
            medical_relevance = 1
        else:
            medical_relevance = 2

        if r_overlap is None:
            rationale_alignment = None
        elif r_overlap == 0:
            rationale_alignment = 0
        elif r_overlap < 4:
            rationale_alignment = 1
        else:
            rationale_alignment = 2

        if sel_overlap > 0 and sel_overlap >= cor_overlap:
            answer_support: AnswerSupport = "supports_selected"
        elif cor_overlap > sel_overlap and cor_overlap > 0:
            answer_support = "supports_other"
        else:
            answer_support = "unrelated"

        shortcut_suspected = any(marker in explanation.lower() for marker in _SHORTCUT_MARKERS) and q_overlap < 2
        quality = apply_aligned_rule(medical_relevance, False, answer_support, rationale_alignment)
        return ScoreRecord(
            prediction_id=pred.prediction_id,
            medical_relevance=medical_relevance,
            rationale_alignment=rationale_alignment,
            answer_support=answer_support,
            medically_invalid=False,
            shortcut_suspected=shortcut_suspected,
            nla_quality_binary=quality,
            taxonomy_cell=taxonomy_cell(pred.correct, quality),
            scorer=self.name,
            scorer_notes=f"q={q_overlap} r={r_overlap} sel={sel_overlap} cor={cor_overlap}",
            scorer_evidence="",
        )


class MedGemmaJudge:
    name = "medgemma_judge_v1"

    def __init__(
        self,
        model_id: str = "google/medgemma-4b-it",
        *,
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_new_tokens: int = 400,
        loader: JudgeLoader = "auto",
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.loader: Literal["image_text", "causal_lm"] = (
            "image_text" if loader == "auto" and model_id.endswith("medgemma-4b-it") else loader
        )  # type: ignore[assignment]
        if self.loader == "auto":
            self.loader = "causal_lm"
        torch_dtype = torch_dtype_from_name(dtype)

        if self.loader == "image_text":
            from transformers import AutoModelForImageTextToText, AutoProcessor

            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map=device,
            ).eval()
            self.tokenizer = None
        elif self.loader == "causal_lm":
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map=device,
            ).eval()
            self.processor = None
        else:
            raise ValueError(f"unknown judge loader: {loader!r}")
        self.parse_errors = 0
        self._closed = False

    def score(self, item: MedItem, pred: Prediction, decode: DecodeRecord) -> ScoreRecord:
        explanation = (decode.explanation or "").strip()
        if not decode.parse_ok or not explanation:
            return _absent_score(pred, scorer=self.name)
        raw = self._generate_raw(build_judge_prompt(item, pred, explanation))
        try:
            parsed = parse_judge_json(raw)
        except JudgeParseError:
            self.parse_errors += 1
            return _judge_parse_failure(pred, raw)
        quality = apply_aligned_rule(
            parsed["medical_relevance"],
            parsed["medically_invalid"],
            parsed["answer_support"],
            parsed["rationale_alignment"],
        )
        return ScoreRecord(
            prediction_id=pred.prediction_id,
            medical_relevance=parsed["medical_relevance"],
            rationale_alignment=parsed["rationale_alignment"],
            answer_support=parsed["answer_support"],
            medically_invalid=parsed["medically_invalid"],
            shortcut_suspected=parsed["shortcut_suspected"],
            nla_quality_binary=quality,
            taxonomy_cell=taxonomy_cell(pred.correct, quality),
            scorer=self.name,
            scorer_notes="",
            scorer_evidence=parsed["evidence"],
        )

    def _generate_raw(self, prompt: str) -> str:
        if self.loader == "image_text":
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            ).to(self.model.device)
            out_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
            return self.processor.decode(out_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)

        ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)
        out_ids = self.model.generate(ids, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.tokenizer.decode(out_ids[0, ids.shape[1] :], skip_special_tokens=True)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.model = None
        self.processor = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> "MedGemmaJudge":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def build_judge_prompt(item: MedItem, pred: Prediction, explanation: str) -> str:
    choices_block = "\n".join(f"{key}. {value}" for key, value in sorted(item.choices.items()))
    selected_answer = pred.selected_answer if pred.selected_answer is not None else "NONE"
    return _JUDGE_PROMPT.format(
        question=item.question,
        choices_block=choices_block,
        answer_key=item.answer_key,
        answer_text=item.choices.get(item.answer_key, ""),
        selected_answer_or_NONE=selected_answer,
        selected_answer_text_or_empty=pred.selected_answer_text or "",
        gold_rationale_or_NONE=item.gold_rationale or "NONE",
        explanation=explanation,
    )


def parse_judge_json(raw: str) -> dict[str, Any]:
    text = _strip_json_fence(raw)
    try:
        data = orjson.loads(text)
    except orjson.JSONDecodeError as exc:
        raise JudgeParseError(str(exc)) from exc
    if not isinstance(data, dict):
        raise JudgeParseError("judge output is not a JSON object")
    keys = set(data)
    if keys != _JUDGE_REQUIRED_KEYS:
        raise JudgeParseError(f"judge keys mismatch: {sorted(keys)}")

    med_rel = data["medical_relevance"]
    if type(med_rel) is not int or med_rel not in {0, 1, 2}:
        raise JudgeParseError("medical_relevance must be 0, 1, or 2")

    rat_align = data["rationale_alignment"]
    if rat_align is not None and (type(rat_align) is not int or rat_align not in {0, 1, 2}):
        raise JudgeParseError("rationale_alignment must be 0, 1, 2, or null")

    answer_support = data["answer_support"]
    if not isinstance(answer_support, str) or answer_support not in _ANSWER_SUPPORT_VALUES:
        raise JudgeParseError("invalid answer_support")

    medically_invalid = data["medically_invalid"]
    if type(medically_invalid) is not bool:
        raise JudgeParseError("medically_invalid must be boolean")

    shortcut_suspected = data["shortcut_suspected"]
    if type(shortcut_suspected) is not bool:
        raise JudgeParseError("shortcut_suspected must be boolean")

    evidence = data["evidence"]
    if not isinstance(evidence, str):
        raise JudgeParseError("evidence must be string")

    return {
        "medical_relevance": med_rel,
        "rationale_alignment": rat_align,
        "answer_support": answer_support,
        "medically_invalid": medically_invalid,
        "shortcut_suspected": shortcut_suspected,
        "evidence": evidence[:240],
    }


def torch_dtype_from_name(dtype: str) -> torch.dtype:
    try:
        value = getattr(torch, dtype)
    except AttributeError as exc:
        raise ValueError(f"unknown torch dtype: {dtype!r}") from exc
    if not isinstance(value, torch.dtype):
        raise ValueError(f"not a torch dtype: {dtype!r}")
    return value


def _concept_tokens(text: str) -> set[str]:
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return {
        token
        for token in normalized.split()
        if len(token) >= 3 and token not in _STOPWORDS and not (len(token) == 1 and token in "abcde")
    }


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    return fenced.group(1).strip() if fenced else stripped


def _absent_score(pred: Prediction, *, scorer: str) -> ScoreRecord:
    quality: NLAQuality = "weak"
    return ScoreRecord(
        prediction_id=pred.prediction_id,
        medical_relevance=0,
        rationale_alignment=None,
        answer_support="unrelated",
        medically_invalid=False,
        shortcut_suspected=True,
        nla_quality_binary=quality,
        taxonomy_cell=taxonomy_cell(pred.correct, quality),
        scorer=scorer,
        scorer_notes="explanation absent",
        scorer_evidence="",
    )


def _judge_parse_failure(pred: Prediction, raw: str) -> ScoreRecord:
    quality: NLAQuality = "weak"
    return ScoreRecord(
        prediction_id=pred.prediction_id,
        medical_relevance=0,
        rationale_alignment=None,
        answer_support="unclear",
        medically_invalid=False,
        shortcut_suspected=False,
        nla_quality_binary=quality,
        taxonomy_cell=taxonomy_cell(pred.correct, quality),
        scorer=MedGemmaJudge.name,
        scorer_notes=f"judge_parse_error: {raw[:120]}",
        scorer_evidence="",
    )


__all__ = [
    "HeuristicScorer",
    "JudgeParseError",
    "MedGemmaJudge",
    "apply_aligned_rule",
    "build_judge_prompt",
    "parse_judge_json",
    "taxonomy_cell",
    "torch_dtype_from_name",
]
