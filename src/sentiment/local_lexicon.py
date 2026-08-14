"""Local sentiment correction rules for Zambian code-switching."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


TOKEN_PATTERN = re.compile(r"\b[\w']+\b", re.UNICODE)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEXICON_PATH = PROJECT_ROOT / "data" / "lexicons" / "local_sentiment_lexicon.csv"


@dataclass(frozen=True, slots=True)
class LocalLexiconMatch:
    token: str
    weight: float
    label: str


FALLBACK_LOCAL_SENTIMENT_LEXICON: dict[str, LocalLexiconMatch] = {
    "fyabupuba": LocalLexiconMatch(token="fyabupuba", weight=-1.0, label="negative"),
    "bwino": LocalLexiconMatch(token="bwino", weight=-0.5, label="sarcasm_marker"),
    "awe": LocalLexiconMatch(token="awe", weight=-0.7, label="negative"),
}

HARD_NEGATIVE_TERMS = {
    "fyabupuba",
    "ifyabupuba",
    "fyabupuba fye",
    "ifya bupuba fye",
}


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    return TOKEN_PATTERN.findall(text.lower())


def _normalise_term(term: str) -> str:
    return " ".join(TOKEN_PATTERN.findall(str(term).lower()))


def _expand_variants(term: str) -> list[str]:
    variants = [part.strip() for part in str(term).split("/") if part.strip()]
    return [_normalise_term(variant) for variant in variants if _normalise_term(variant)]


def _weight_for_sentiment(sentiment: str, raw_weight: str) -> float:
    try:
        weight = float(raw_weight)
    except (TypeError, ValueError):
        weight = 0.0

    if sentiment == "sarcasm_marker" and weight != 0:
        return -abs(weight)
    return weight


@lru_cache(maxsize=1)
def load_local_lexicon(path: str | Path = DEFAULT_LEXICON_PATH) -> dict[str, LocalLexiconMatch]:
    """Load the auditable local lexicon from CSV, falling back to core rules."""
    lexicon_path = Path(path)
    if not lexicon_path.is_file():
        return FALLBACK_LOCAL_SENTIMENT_LEXICON

    loaded: dict[str, LocalLexiconMatch] = {}
    with lexicon_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            sentiment = str(row.get("sentiment", "")).strip().lower()
            weight = _weight_for_sentiment(sentiment, str(row.get("weight", "0")))
            for term in _expand_variants(str(row.get("term", ""))):
                loaded[term] = LocalLexiconMatch(token=term, weight=weight, label=sentiment)

    return loaded or FALLBACK_LOCAL_SENTIMENT_LEXICON


def _match_lexicon_terms(text: str, lexicon: dict[str, LocalLexiconMatch]) -> list[LocalLexiconMatch]:
    normalised_text = " ".join(tokenize(text))
    if not normalised_text:
        return []
    matches: list[LocalLexiconMatch] = []
    occupied_spans: list[tuple[int, int]] = []
    sorted_terms = sorted(lexicon.items(), key=lambda item: (len(item[0].split()), len(item[0])), reverse=True)
    for term, match in sorted_terms:
        if match.weight == 0:
            continue
        found = re.search(rf"(?<!\w){re.escape(term)}(?!\w)", normalised_text)
        if not found:
            continue
        span = found.span()
        if any(max(span[0], used[0]) < min(span[1], used[1]) for used in occupied_spans):
            continue
        occupied_spans.append(span)
        matches.append(match)
    return matches


def apply_local_correction(text: str, base_compound: float, base_label: str) -> dict[str, object]:
    """
    Apply a small, auditable sentiment correction based on local lexicon matches.

    The correction is auditable:
    - local terms live in data/lexicons/local_sentiment_lexicon.csv
    - slash-separated variants and short phrases are supported
    - the output keeps the raw VADER values alongside the corrected values
    """
    lexicon = load_local_lexicon()
    matches = _match_lexicon_terms(text, lexicon)

    if not matches:
        return {
            "corrected_compound": float(base_compound),
            "corrected_label": base_label,
            "local_correction_applied": False,
            "local_correction_score": 0.0,
            "local_correction_terms": "",
        }

    matched_terms = sorted({match.token for match in matches})
    if any(term in HARD_NEGATIVE_TERMS for term in matched_terms):
        return {
            "corrected_compound": -1.0,
            "corrected_label": "negative",
            "local_correction_applied": True,
            "local_correction_score": -1.0,
            "local_correction_terms": ",".join(matched_terms),
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
        "local_correction_terms": ",".join(matched_terms),
    }
