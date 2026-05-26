"""Tests for medical QA prompt construction."""

from __future__ import annotations

import re
from unittest.mock import MagicMock

import pytest

from nla.mednla.prompts import apply_chat, build_canonical_prompt, build_variant
from nla.mednla.schema import MedItem

SAMPLE_ITEM = MedItem(
    item_id="medqa:test:000001",
    dataset="medqa",
    split="test",
    subject="cardiology",
    question="Which drug is first-line for hypertension?",
    choices={
        "A": "Lisinopril",
        "B": "Metformin",
        "C": "Atorvastatin",
        "D": "Levothyroxine",
    },
    answer_key="A",
    gold_rationale="ACE inhibitors are first-line for uncomplicated hypertension.",
    source_metadata={"hf_index": 1},
)

BANNED_PHRASES = ("step by step", "chain of thought", "reason")
VARIANTS = ("canonical", "option_shuffle", "compact")


@pytest.fixture
def item() -> MedItem:
    return SAMPLE_ITEM


@pytest.mark.parametrize("variant", VARIANTS)
def test_choice_text_appears_exactly_once(item: MedItem, variant: str) -> None:
    prompt, _ = build_variant(item, variant, variant_seed=7)  # type: ignore[arg-type]
    for choice_text in item.choices.values():
        matches = re.findall(re.escape(choice_text), prompt)
        assert len(matches) == 1, f"{choice_text!r} found {len(matches)} times in {variant}"


@pytest.mark.parametrize("seed", range(16))
def test_shuffle_answer_key_remaps_to_same_option_text(item: MedItem, seed: int) -> None:
    prompt, choice_map = build_variant(item, "option_shuffle", variant_seed=seed)
    assert prompt
    new_answer_key = choice_map[item.answer_key]
    original_text = item.choices[item.answer_key]
    shuffled_choices = {choice_map[key]: text for key, text in item.choices.items()}
    assert shuffled_choices[new_answer_key] == original_text


@pytest.mark.parametrize("variant", VARIANTS)
def test_build_variant_is_deterministic(item: MedItem, variant: str) -> None:
    first = build_variant(item, variant, variant_seed=3)  # type: ignore[arg-type]
    second = build_variant(item, variant, variant_seed=3)  # type: ignore[arg-type]
    assert first == second


@pytest.mark.parametrize("variant", ("canonical", "compact"))
def test_prompts_exclude_cot_phrases(item: MedItem, variant: str) -> None:
    prompt, _ = build_variant(item, variant, variant_seed=0)  # type: ignore[arg-type]
    lowered = prompt.lower()
    for phrase in BANNED_PHRASES:
        assert phrase not in lowered


@pytest.mark.parametrize("variant", VARIANTS)
def test_gold_rationale_not_leaked(item: MedItem, variant: str) -> None:
    assert item.gold_rationale is not None
    prompt, _ = build_variant(item, variant, variant_seed=0)  # type: ignore[arg-type]
    assert item.gold_rationale not in prompt


def test_build_canonical_prompt_format(item: MedItem) -> None:
    prompt = build_canonical_prompt(item)
    assert prompt.startswith("Question:\n")
    assert "Answer with the single best option letter only." in prompt


def test_apply_chat_uses_generation_prompt() -> None:
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "chat-prompt"
    result = apply_chat(tokenizer, "plain prompt")
    assert result == "chat-prompt"
    tokenizer.apply_chat_template.assert_called_once_with(
        [{"role": "user", "content": "plain prompt"}],
        tokenize=False,
        add_generation_prompt=True,
    )
