"""Sentiment analysis package."""

from .local_lexicon import apply_local_correction
from .nrc_analyzer import get_nrc_scores
from .roberta_analyzer import label_dataframe as label_roberta_dataframe
from .roberta_analyzer import label_text_batch as label_roberta_text_batch
from .vader_analyzer import VaderScore, get_vader_scores

__all__ = [
    "VaderScore",
    "apply_local_correction",
    "get_nrc_scores",
    "get_vader_scores",
    "label_roberta_dataframe",
    "label_roberta_text_batch",
]
