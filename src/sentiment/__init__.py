"""Sentiment analysis package."""

from .local_lexicon import apply_local_correction
from .vader_analyzer import VaderScore, get_vader_scores

__all__ = ["VaderScore", "apply_local_correction", "get_vader_scores"]
