"""Local sentiment correction rules for Zambian code-switching."""

from __future__ import annotations

import re
from dataclasses import dataclass


TOKEN_PATTERN = re.compile(r"\b[\w']+\b", re.UNICODE)


@dataclass(frozen=True, slots=True)
class LocalLexiconMatch:
    token: str
    weight: float
    label: str


# Keep this list intentionally small and explicit so it is easy to review.
LOCAL_SENTIMENT_LEXICON: dict[str, LocalLexiconMatch] = {
    "fyabupuba": LocalLexiconMatch(token="fyabupuba", weight=-0.8, label="negative"),
    "bwino": LocalLexiconMatch(token="bwino", weight=0.5, label="positive"),
    "awe": LocalLexiconMatch(token="awe", weight=-0.4, label="negative"),
}


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    return TOKEN_PATTERN.findall(text.lower())


def apply_local_correction(text: str, base_compound: float, base_label: str) -> dict[str, object]:
    """
    Apply a small, auditable sentiment correction based on local lexicon matches.

    The correction is intentionally conservative:
    - only a few high-confidence local terms are used
    - the output keeps the raw VADER values alongside the corrected values
    """
    tokens = tokenize(text)
    matches = [LOCAL_SENTIMENT_LEXICON[token] for token in tokens if token in LOCAL_SENTIMENT_LEXICON]

    if not matches:
        return {
            "corrected_compound": float(base_compound),
            "corrected_label": base_label,
            "local_correction_applied": False,
            "local_correction_score": 0.0,
            "local_correction_terms": "",
        }

    if "fyabupuba" in tokens:
        return {
            "corrected_compound": -1.0,
            "corrected_label": "negative",
            "local_correction_applied": True,
            "local_correction_score": -1.0,
            "local_correction_terms": "fyabupuba",
        }

    correction_score = sum(match.weight for match in matches)
    corrected_compound = max(-1.0, min(1.0, float(base_compound) + correction_score))

    if corrected_compound >= 0.05:
        corrected_label = "positive"
    elif corrected_compound <= -0.05:
        corrected_label = "negative"
    else:
        corrected_label = "neutral"

    return {
        "corrected_compound": corrected_compound,
        "corrected_label": corrected_label,
        "local_correction_applied": True,
        "local_correction_score": correction_score,
        "local_correction_terms": ",".join(sorted({match.token for match in matches})),
    }
