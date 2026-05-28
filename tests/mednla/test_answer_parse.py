"""Tests for answer letter parsing."""

from __future__ import annotations

import pytest

from nla.mednla.answer_parse import parse_answer

CHOICE_KEYS = ["A", "B", "C", "D"]
ASPIRIN_CHOICES = {"A": "Aspirin", "B": "Ibuprofen"}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("A", "A"),
        ("(A)", "A"),
        ("A.", "A"),
        (" A ", "A"),
        ("Answer: A", "A"),
        ("Final answer: (B)", "B"),
        ("Answer is C", "C"),
        ("I choose option D because it fits best.", "D"),
        ("The best answer is D because A is contraindicated.", "D"),
        ("the answer is C", "C"),
        ("", None),
        ("A or B", None),
        ("Both A and B are correct", None),
    ],
)
def test_parse_answer_table(text: str, expected: str | None) -> None:
    assert parse_answer(text, CHOICE_KEYS) == expected


def test_parse_answer_choice_text_exact_match() -> None:
    assert (
        parse_answer(
            "Aspirin",
            list(ASPIRIN_CHOICES.keys()),
            choice_texts=ASPIRIN_CHOICES,
        )
        == "A"
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The answer is A, not Ibuprofen.", "A"),
        ("A. Not Ibuprofen.", "A"),
        ("A patient should receive Ibuprofen.", "B"),
    ],
)
def test_explicit_letters_take_precedence_over_choice_text_mentions(
    text: str,
    expected: str,
) -> None:
    assert parse_answer(text, list(ASPIRIN_CHOICES.keys()), choice_texts=ASPIRIN_CHOICES) == expected
