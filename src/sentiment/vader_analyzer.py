"""VADER sentiment scoring helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import nltk

from .local_lexicon import apply_local_correction

try:
    from nltk.sentiment import SentimentIntensityAnalyzer
except Exception as exc:  # pragma: no cover - import-time fallback
    SentimentIntensityAnalyzer = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@dataclass(slots=True)
class VaderScore:
    compound: float
    pos: float
    neu: float
    neg: float
    label: str


_ANALYZER: SentimentIntensityAnalyzer | None = None


def _load_analyzer() -> SentimentIntensityAnalyzer:
    """Load VADER, downloading the lexicon if needed."""
    global _ANALYZER
    if _ANALYZER is not None:
        return _ANALYZER

    if SentimentIntensityAnalyzer is None:
        raise RuntimeError(f"nltk.sentiment.vader could not be imported: {_IMPORT_ERROR}")

    try:
        _ANALYZER = SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
        _ANALYZER = SentimentIntensityAnalyzer()

    return _ANALYZER


def get_vader_scores(text: str) -> dict[str, Any]:
    """Return VADER polarity scores and a discrete label."""
    analyzer = _load_analyzer()
    scores = analyzer.polarity_scores(text or "")
    raw_compound = float(scores["compound"])
    if raw_compound >= 0.05:
        raw_label = "positive"
    elif raw_compound <= -0.05:
        raw_label = "negative"
    else:
        raw_label = "neutral"

    correction = apply_local_correction(text or "", raw_compound, raw_label)
    compound = float(correction["corrected_compound"])
    label = str(correction["corrected_label"])

    return {
        "raw_compound": raw_compound,
        "raw_label": raw_label,
        "compound": compound,
        "pos": float(scores["pos"]),
        "neu": float(scores["neu"]),
        "neg": float(scores["neg"]),
        "label": label,
        **correction,
    }
