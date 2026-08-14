"""Select readable comment text for rule-based language analysis."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd


SENTIMENT_TEXT_COLUMNS = ("text_raw", "text", "text_clean", "processed_text")


def select_sentiment_text(row: Mapping[str, Any]) -> str:
    """Prefer text that preserves punctuation, contractions, and negation."""
    for column in SENTIMENT_TEXT_COLUMNS:
        value = row.get(column)
        if value is None or pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            return text
    return ""
