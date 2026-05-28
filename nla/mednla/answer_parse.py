"""Parse model-generated answers into a single choice letter."""

from __future__ import annotations

import re

_FIRST_LINE_PATTERN = re.compile(
    r"^\s*\(([A-Z])\)\s*$|^\s*([A-Z])[\.\):]\s*$",
)
_LEADING_KEY_PATTERN = re.compile(r"^\s*\(?([A-Z])\)?[\.\):]\s+", re.IGNORECASE)
_PREFIX_PATTERN = re.compile(
    r"\b(?:Final answer|Selected answer|Correct answer|Best answer|Answer|Option|Choice)\b"
    r"\s*(?:is|:|\-)?\s*\(?([A-Z])\)?\b",
    re.IGNORECASE,
)
_CHOOSE_PATTERN = re.compile(
    r"\b(?:choose|select|selected|pick)\s+(?:option\s+)?\(?([A-Z])\)?\b",
    re.IGNORECASE,
)
_KEY_BOUNDARY_PATTERN = re.compile(r"\b([A-Z])\b", re.IGNORECASE)


def parse_answer(
    text: str,
    choice_keys: list[str],
    *,
    choice_texts: dict[str, str] | None = None,
) -> str | None:
    key_set = set(choice_keys)

    stripped = text.lstrip()
    if stripped.endswith("."):
        stripped = stripped[:-1]
    candidate = stripped.strip()
    if len(candidate) == 1 and candidate in key_set:
        return candidate

    first_line = text.splitlines()[0] if text else ""
    match = _FIRST_LINE_PATTERN.match(first_line)
    if match:
        letter = match.group(1) or match.group(2)
        if letter in key_set:
            return letter

    match = _LEADING_KEY_PATTERN.match(text)
    if match:
        letter = match.group(1).upper()
        if letter in key_set:
            return letter

    prefix_window = text[:100]
    for pattern in (_PREFIX_PATTERN, _CHOOSE_PATTERN):
        for match in pattern.finditer(prefix_window):
            letter = match.group(1).upper()
            if letter in key_set:
                return letter

    if choice_texts is not None:
        normalized = text.strip().lower()
        exact_matches = [
            key
            for key in choice_keys
            if key in choice_texts and normalized == choice_texts[key].strip().lower()
        ]
        if len(exact_matches) == 1:
            return exact_matches[0]

        lowered = text.lower()
        substring_matches = [
            key
            for key in choice_keys
            if key in choice_texts and choice_texts[key].strip().lower() in lowered
        ]
        if len(substring_matches) == 1:
            return substring_matches[0]
        if len(substring_matches) > 1:
            return None

    key_substring_matches = _substring_key_matches(text, choice_keys)
    if len(key_substring_matches) == 1:
        return key_substring_matches[0]

    return None


def _substring_key_matches(text: str, choice_keys: list[str]) -> list[str]:
    seen: list[str] = []
    for match in _KEY_BOUNDARY_PATTERN.finditer(text):
        letter = match.group(1).upper()
        if letter in choice_keys and letter not in seen:
            seen.append(letter)
    return seen


__all__ = ["parse_answer"]
