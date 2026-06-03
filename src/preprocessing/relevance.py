"""Rule-based relevance detection for ZESCO/electricity comments."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass


TOKEN_PATTERN = re.compile(r"\b[\w']+\b", re.UNICODE)

STRONG_TERMS = {
    "zesco",
    "electricity",
    "electric",
    "power",
    "loadshedding",
    "loadshedded",
    "loadsheding",
    "malaiti",
    "amalaiti",
    "amaiti",
    "meter",
    "metering",
    "units",
    "tariff",
    "erb",
    "grid",
    "solar",
    "transformer",
    "outage",
    "blackout",
    "substation",
    "hydro",
    "kariba",
    "generator",
    "genset",
    "inverter",
    "supply",
}

STRONG_PHRASES = {
    "load shedding",
    "power cut",
    "power outage",
    "power supply",
    "electricity supply",
    "net metering",
    "stable supply",
    "kafue gorge",
    "energy regulation",
    "energy sector",
}

CONTEXT_TERMS = {
    "schedule",
    "timetable",
    "fault",
    "restored",
    "restore",
    "cut",
    "off",
    "gone",
    "came",
    "come",
    "midnight",
    "hrs",
    "hours",
    "charge",
    "charging",
    "switch",
    "switched",
    "line",
    "group",
}

POST_CONTEXT_TERMS = {
    "update",
    "promise",
    "promised",
    "said",
    "october",
    "1st",
    "stability",
    "stable",
    "admin",
    "announcement",
    "countdown",
    "counting",
}

WEAK_OFF_TOPIC_TERMS = {
    "football",
    "church",
    "relationship",
    "music",
    "birthday",
    "movie",
    "school fees",
}


@dataclass(frozen=True, slots=True)
class RelevanceResult:
    is_relevant: bool
    relevance_score: int
    relevance_reason: str
    relevance_terms: str


def _normalise(text: str) -> str:
    return " ".join(TOKEN_PATTERN.findall(str(text).lower()))


def _matched_phrases(text: str, phrases: set[str]) -> list[str]:
    return sorted(phrase for phrase in phrases if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text))


def _matched_tokens(text: str, terms: set[str]) -> list[str]:
    tokens = set(TOKEN_PATTERN.findall(text))
    return sorted(tokens.intersection(terms))


def assess_relevance(text: str) -> dict[str, object]:
    """Assess whether a comment is about ZESCO, electricity, or energy-service experience."""
    normalised = _normalise(text)
    if not normalised:
        result = RelevanceResult(False, 0, "empty_or_unreadable", "")
        return asdict(result)

    strong_matches = _matched_tokens(normalised, STRONG_TERMS)
    phrase_matches = _matched_phrases(normalised, STRONG_PHRASES)
    context_matches = _matched_tokens(normalised, CONTEXT_TERMS)
    post_context_matches = _matched_tokens(normalised, POST_CONTEXT_TERMS)
    off_topic_matches = _matched_phrases(normalised, WEAK_OFF_TOPIC_TERMS)

    score = (
        (len(strong_matches) * 3)
        + (len(phrase_matches) * 4)
        + len(context_matches)
        + len(post_context_matches)
    )
    matched_terms = sorted(set(strong_matches + phrase_matches + context_matches + post_context_matches))

    if phrase_matches or strong_matches:
        reason = "matched_energy_terms"
        if context_matches:
            reason = "matched_energy_and_context_terms"
        result = RelevanceResult(True, score, reason, ",".join(matched_terms))
        return asdict(result)

    if len(context_matches) >= 3:
        result = RelevanceResult(True, score, "multiple_power_behavior_terms", ",".join(matched_terms))
        return asdict(result)

    if len(post_context_matches) >= 2:
        result = RelevanceResult(True, score, "matched_post_context_terms", ",".join(matched_terms))
        return asdict(result)

    if off_topic_matches:
        result = RelevanceResult(False, score, "matched_off_topic_terms_only", ",".join(off_topic_matches))
        return asdict(result)

    result = RelevanceResult(False, score, "no_energy_or_zesco_terms", ",".join(matched_terms))
    return asdict(result)
