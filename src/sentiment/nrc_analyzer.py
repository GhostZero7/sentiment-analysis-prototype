"""NRC emotion scoring helpers."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from nrclex import NRCLex

_TARGET_EMOTIONS = ("anger", "anticipation", "fear", "trust", "sadness")


@lru_cache(maxsize=1)
def _load_lexicon() -> dict[str, list[str]]:
    """Load the NRC lexicon once and reuse the underlying word-emotion map."""
    lexicon = NRCLex().__dict__.get("__lexicon__", {})
    if not isinstance(lexicon, dict):  # pragma: no cover - defensive fallback
        return {}
    return {str(word).lower(): [str(emotion).lower() for emotion in emotions] for word, emotions in lexicon.items()}


def _tokenize(text: str) -> list[str]:
    """Tokenize the already-cleaned text with a lightweight whitespace split."""
    return [token.strip().lower() for token in str(text or "").split() if token.strip()]


def get_nrc_scores(text: str) -> dict[str, Any]:
    """Return normalized NRC emotion scores for the target ZESCO workflow."""
    tokens = _tokenize(text)
    token_count = len(tokens)
    lexicon = _load_lexicon()

    counts = {emotion: 0 for emotion in _TARGET_EMOTIONS}
    matched_tokens = 0
    for token in tokens:
        emotions = lexicon.get(token)
        if not emotions:
            continue
        matched_tokens += 1
        for emotion in emotions:
            if emotion in counts:
                counts[emotion] += 1

    if token_count == 0:
        normalized = {emotion: 0.0 for emotion in _TARGET_EMOTIONS}
        hope = 0.0
        frustration = 0.0
    else:
        normalized = {emotion: counts[emotion] / token_count for emotion in _TARGET_EMOTIONS}
        hope = normalized["anticipation"]
        frustration = (normalized["anger"] + normalized["sadness"]) / 2.0

    return {
        "nrc_token_count": token_count,
        "nrc_matched_token_count": matched_tokens,
        "nrc_anger": normalized["anger"],
        "nrc_anticipation": normalized["anticipation"],
        "nrc_fear": normalized["fear"],
        "nrc_trust": normalized["trust"],
        "nrc_sadness": normalized["sadness"],
        "nrc_hope": hope,
        "nrc_frustration": frustration,
    }
