"""Combined NRC and local Zambian emotion scoring helpers."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from nrclex import NRCLex


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCAL_EMOTION_PATH = PROJECT_ROOT / "data" / "lexicons" / "local_emotion_lexicon.csv"
TOKEN_PATTERN = re.compile(r"\b[\w']+\b", re.UNICODE)
_NRC_EMOTIONS = ("anger", "anticipation", "fear", "trust", "sadness")
_LOCAL_EMOTIONS = ("anger", "fear", "trust", "hope", "sadness", "frustration")


@dataclass(frozen=True, slots=True)
class LocalEmotionMatch:
    term: str
    scores: dict[str, float]


@lru_cache(maxsize=1)
def _load_lexicon() -> dict[str, list[str]]:
    """Load NRC once and reuse its word-to-emotion map."""
    lexicon = NRCLex().__dict__.get("__lexicon__", {})
    if not isinstance(lexicon, dict):  # pragma: no cover - defensive fallback
        return {}
    return {
        str(word).lower(): [str(emotion).lower() for emotion in emotions]
        for word, emotions in lexicon.items()
    }


def _normalise_term(value: str) -> str:
    return " ".join(TOKEN_PATTERN.findall(str(value).lower()))


def _expand_variants(value: str) -> list[str]:
    return [term for part in str(value).split("/") if (term := _normalise_term(part))]


def _bounded_weight(value: str | float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


@lru_cache(maxsize=1)
def load_local_emotion_lexicon(
    path: str | Path = DEFAULT_LOCAL_EMOTION_PATH,
) -> dict[str, LocalEmotionMatch]:
    """Load local emotion terms, variants, and phrases from CSV."""
    lexicon_path = Path(path)
    if not lexicon_path.is_file():
        return {}

    loaded: dict[str, LocalEmotionMatch] = {}
    with lexicon_path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            scores = {emotion: _bounded_weight(row.get(emotion, 0.0)) for emotion in _LOCAL_EMOTIONS}
            for term in _expand_variants(row.get("term", "")):
                loaded[term] = LocalEmotionMatch(term=term, scores=scores)
    return loaded


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(str(text or "").lower())


def _match_local_terms(text: str) -> list[LocalEmotionMatch]:
    """Prefer longer phrases and prevent overlapping variants from double-counting."""
    normalised = " ".join(_tokenize(text))
    if not normalised:
        return []

    matches: list[LocalEmotionMatch] = []
    occupied_spans: list[tuple[int, int]] = []
    lexicon = load_local_emotion_lexicon()
    sorted_terms = sorted(lexicon.items(), key=lambda item: (len(item[0].split()), len(item[0])), reverse=True)
    for term, match in sorted_terms:
        found = re.search(rf"(?<!\w){re.escape(term)}(?!\w)", normalised)
        if not found:
            continue
        span = found.span()
        if any(max(span[0], used[0]) < min(span[1], used[1]) for used in occupied_spans):
            continue
        occupied_spans.append(span)
        matches.append(match)
    return matches


def _local_scores(matches: list[LocalEmotionMatch], *, is_sarcastic: bool) -> dict[str, float]:
    if not matches:
        return {emotion: 0.0 for emotion in _LOCAL_EMOTIONS}

    count = len(matches)
    scores = {
        emotion: min(1.0, sum(match.scores[emotion] for match in matches) / count)
        for emotion in _LOCAL_EMOTIONS
    }
    if is_sarcastic:
        scores["trust"] = 0.0
        scores["hope"] = 0.0
    return scores


def get_nrc_scores(text: str, *, is_sarcastic: bool = False) -> dict[str, Any]:
    """Return NRC scores supplemented by explainable local emotion weights."""
    tokens = _tokenize(text)
    token_count = len(tokens)
    nrc_lexicon = _load_lexicon()

    counts = {emotion: 0 for emotion in _NRC_EMOTIONS}
    matched_tokens = 0
    for token in tokens:
        emotions = nrc_lexicon.get(token)
        if not emotions:
            continue
        matched_tokens += 1
        for emotion in emotions:
            if emotion in counts:
                counts[emotion] += 1

    if token_count:
        base = {emotion: counts[emotion] / token_count for emotion in _NRC_EMOTIONS}
    else:
        base = {emotion: 0.0 for emotion in _NRC_EMOTIONS}
    base_hope = base["anticipation"]
    base_frustration = (base["anger"] + base["sadness"]) / 2.0

    local_matches = _match_local_terms(text)
    local = _local_scores(local_matches, is_sarcastic=is_sarcastic)

    combined = {
        "anger": min(1.0, base["anger"] + local["anger"]),
        "fear": min(1.0, base["fear"] + local["fear"]),
        "trust": min(1.0, base["trust"] + local["trust"]),
        "hope": min(1.0, base_hope + local["hope"]),
        "sadness": min(1.0, base["sadness"] + local["sadness"]),
        "frustration": min(1.0, base_frustration + local["frustration"]),
    }

    return {
        "nrc_token_count": token_count,
        "nrc_matched_token_count": matched_tokens,
        "nrc_base_anger": base["anger"],
        "nrc_base_anticipation": base["anticipation"],
        "nrc_base_fear": base["fear"],
        "nrc_base_trust": base["trust"],
        "nrc_base_sadness": base["sadness"],
        "nrc_base_hope": base_hope,
        "nrc_base_frustration": base_frustration,
        "local_emotion_applied": bool(local_matches),
        "local_emotion_terms": ",".join(sorted(match.term for match in local_matches)),
        "local_emotion_match_count": len(local_matches),
        **{f"local_{emotion}_score": local[emotion] for emotion in _LOCAL_EMOTIONS},
        "nrc_anger": combined["anger"],
        "nrc_anticipation": base["anticipation"],
        "nrc_fear": combined["fear"],
        "nrc_trust": combined["trust"],
        "nrc_sadness": combined["sadness"],
        "nrc_hope": combined["hope"],
        "nrc_frustration": combined["frustration"],
    }
