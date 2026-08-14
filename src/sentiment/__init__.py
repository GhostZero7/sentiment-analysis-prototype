"""Sentiment analysis package."""

from typing import Any

from .local_lexicon import apply_local_correction
from .nrc_analyzer import get_nrc_scores
from .vader_analyzer import VaderScore, get_vader_scores

__all__ = [
    "VaderScore",
    "apply_local_correction",
    "get_nrc_scores",
    "get_vader_scores",
    "label_roberta_dataframe",
    "label_roberta_text_batch",
]


def __getattr__(name: str) -> Any:
    """Load the optional transformer stack only when its public helpers are used."""
    if name == "label_roberta_dataframe":
        from .roberta_analyzer import label_dataframe

        return label_dataframe
    if name == "label_roberta_text_batch":
        from .roberta_analyzer import label_text_batch

        return label_text_batch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
