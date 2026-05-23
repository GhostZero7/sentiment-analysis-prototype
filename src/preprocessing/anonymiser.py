"""Manual anonymisation helpers for comment text."""

from __future__ import annotations

import re
from typing import Iterable

from .cleaner import SPACE_PATTERN


# Manual, corpus-driven redaction list.
# These are names and name-like phrases observed in the collected comments.
NAME_PHRASES: tuple[str, ...] = (
    "angela baker",
    "chomba lesa malembeka",
    "christine joanna julio",
    "dennis zulu",
    "gusha lawrence kasongo",
    "isaac zulu",
    "jay mtonga",
    "jay praise nkhata",
    "joyce nkhoma",
    "joshua mayuya",
    "maclean sandra phiri",
    "mark mwandila",
    "mweene omedy keny",
    "nhamo afiya chiwila",
    "precious chanda",
    "saviour alingo chisanga",
    "susan j stephens",
    "tri cia phiri",
)


NAME_TOKEN_PATTERNS: tuple[str, ...] = tuple(sorted({token for phrase in NAME_PHRASES for token in phrase.split()}))


def _remove_phrase(text: str, phrase: str) -> str:
    pattern = re.compile(rf"\b{re.escape(phrase)}\b", flags=re.IGNORECASE)
    return pattern.sub(" ", text)


def anonymise_text(text: str, *, extra_phrases: Iterable[str] | None = None) -> str:
    """Remove manually identified names from a lowercased comment string."""
    if not text:
        return ""

    cleaned = str(text)
    phrases = tuple(NAME_PHRASES) + tuple(extra_phrases or ())
    for phrase in phrases:
        cleaned = _remove_phrase(cleaned, phrase)

    for token in NAME_TOKEN_PATTERNS:
        token_pattern = re.compile(rf"\b{re.escape(token)}\b", flags=re.IGNORECASE)
        cleaned = token_pattern.sub(" ", cleaned)

    cleaned = SPACE_PATTERN.sub(" ", cleaned)
    return cleaned.strip()
