"""Medical QA prompt construction and chat templating."""

from __future__ import annotations

import random
from typing import Any

from nla.mednla.schema import MedItem, VariantName

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def build_canonical_prompt(item: MedItem) -> str:
    return _render_canonical_body(item.question, item.choices)


def build_variant(
    item: MedItem,
    variant: VariantName,
    variant_seed: int,
) -> tuple[str, dict[str, str]]:
    if variant == "canonical":
        return build_canonical_prompt(item), _identity_choice_map(item.choices)
    if variant == "compact":
        return _render_compact_body(item.question, item.choices), _identity_choice_map(item.choices)
    if variant == "option_shuffle":
        return _build_option_shuffle(item, variant_seed)
    raise ValueError(f"unknown variant: {variant!r}")


def apply_chat(tokenizer: Any, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _identity_choice_map(choices: dict[str, str]) -> dict[str, str]:
    return {key: key for key in sorted(choices)}


def _sorted_choice_pairs(choices: dict[str, str]) -> list[tuple[str, str]]:
    return sorted(choices.items(), key=lambda pair: pair[0])


def _render_canonical_body(question: str, choices: dict[str, str]) -> str:
    lines = [
        "Question:",
        question,
        "",
        "Options:",
    ]
    for key, text in _sorted_choice_pairs(choices):
        lines.append(f"{key}. {text}")
    lines.extend(["", "Answer with the single best option letter only."])
    return "\n".join(lines)


def _render_compact_body(question: str, choices: dict[str, str]) -> str:
    option_parts = [f"{key}. {text}" for key, text in _sorted_choice_pairs(choices)]
    return "\n".join(
        [
            question,
            "  ".join(option_parts),
            "Answer with the letter only.",
        ]
    )


def _build_option_shuffle(item: MedItem, variant_seed: int) -> tuple[str, dict[str, str]]:
    pairs = _sorted_choice_pairs(item.choices)
    shuffled = random.Random(variant_seed).sample(pairs, k=len(pairs))
    new_choices: dict[str, str] = {}
    original_to_variant: dict[str, str] = {}
    for index, (original_key, text) in enumerate(shuffled):
        new_key = _LETTERS[index]
        new_choices[new_key] = text
        original_to_variant[original_key] = new_key
    prompt = _render_canonical_body(item.question, new_choices)
    return prompt, original_to_variant


__all__ = ["apply_chat", "build_canonical_prompt", "build_variant"]
