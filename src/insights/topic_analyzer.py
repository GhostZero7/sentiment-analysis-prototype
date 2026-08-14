"""Explainable thematic grouping for Zambian energy discourse."""

from __future__ import annotations

import re
from collections.abc import Iterable
from functools import lru_cache

import pandas as pd


TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Load shedding and reliability": (
        "load shedding",
        "loadshedding",
        "power cut",
        "power cuts",
        "blackout",
        "outage",
        "no power",
        "restore power",
        "power supply",
        "hours of power",
        "power",
        "electricity",
        "hours",
        "hrs",
        "stability",
        "supply",
        "amaliti",
        "malaiti",
        "schedule",
        "supply hours",
        "loadsheeding",
    ),
    "Tariffs and affordability": (
        "tariff",
        "tariffs",
        "afford",
        "affordable",
        "affordability",
        "expensive",
        "price",
        "prices",
        "cost",
        "costs",
        "bill",
        "bills",
        "meter",
        "units",
        "kwh",
        "kwacha",
        "charge",
        "charges",
    ),
    "Renewable energy and solar": (
        "solar",
        "renewable",
        "green energy",
        "wind power",
        "wind energy",
        "hydro",
        "hydropower",
        "battery",
        "batteries",
        "inverter",
        "net metering",
        "clean energy",
    ),
    "Customer service and communication": (
        "customer service",
        "call centre",
        "call center",
        "response",
        "respond",
        "update",
        "updates",
        "notice",
        "announcement",
        "communicate",
        "communication",
        "explain",
        "press briefing",
        "complaint",
    ),
    "Faults, connections and infrastructure": (
        "fault",
        "faults",
        "transformer",
        "cable",
        "pole",
        "connection",
        "connections",
        "connect",
        "maintenance",
        "substation",
        "grid",
        "breakdown",
        "repair",
        "repairs",
    ),
    "Governance and public trust": (
        "government",
        "ministry",
        "erb",
        "policy",
        "policies",
        "regulation",
        "regulations",
        "corruption",
        "trust",
        "accountability",
        "transparent",
        "transparency",
        "promise",
        "promises",
    ),
    "Jobs and economic impact": (
        "job",
        "jobs",
        "employment",
        "business",
        "businesses",
        "production",
        "economy",
        "economic",
        "income",
        "workers",
        "market",
    ),
    "Environment and climate": (
        "climate",
        "drought",
        "water level",
        "kariba",
        "emissions",
        "pollution",
        "environment",
        "environmental",
        "sustainable",
        "sustainability",
        "sdg 7",
    ),
}

OTHER_TOPIC = "Other energy concerns"
TEXT_COLUMNS = ("text_raw", "text", "text_clean", "processed_text")
EMOTION_COLUMNS = (
    "nrc_anger",
    "nrc_fear",
    "nrc_trust",
    "nrc_sadness",
    "nrc_hope",
    "nrc_frustration",
)


@lru_cache(maxsize=4096)
def _normalise_text(value: str) -> str:
    text = str(value or "").lower().replace("-", " ")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text)).strip()


def _keyword_score(text: str, keywords: Iterable[str]) -> int:
    score = 0
    padded = f" {text} "
    for keyword in keywords:
        normalised = _normalise_text(keyword)
        if f" {normalised} " in padded:
            score += max(1, len(normalised.split()))
    return score


def assign_topic(text: object) -> str:
    """Assign one primary, auditable policy topic to a comment."""
    normalised = _normalise_text(str(text or ""))
    if not normalised:
        return OTHER_TOPIC

    scores = {
        topic: _keyword_score(normalised, keywords)
        for topic, keywords in TOPIC_KEYWORDS.items()
    }
    best_score = max(scores.values())
    if best_score <= 0:
        return OTHER_TOPIC

    candidates = [topic for topic, score in scores.items() if score == best_score]
    if len(candidates) > 1 and "Load shedding and reliability" in candidates:
        candidates.remove("Load shedding and reliability")
    return candidates[0]


def _first_text_value(row: pd.Series, text_columns: Iterable[str]) -> str:
    for column in text_columns:
        value = row.get(column, "")
        if pd.notna(value) and str(value).strip():
            return str(value)
    return ""


def add_topic_labels(
    frame: pd.DataFrame,
    *,
    text_columns: Iterable[str] = TEXT_COLUMNS,
) -> pd.DataFrame:
    """Return a copy with a `topic` column derived from available comment text."""
    labeled = frame.copy()
    if labeled.empty:
        labeled["topic"] = pd.Series(dtype="object")
        return labeled
    labeled["topic"] = labeled.apply(
        lambda row: assign_topic(_first_text_value(row, text_columns)),
        axis=1,
    )
    return labeled


def _label_percent(group: pd.DataFrame, label_column: str, label: str) -> float:
    if label_column not in group.columns or group.empty:
        return 0.0
    labels = group[label_column].fillna("unknown").astype(str).str.lower()
    return round(float(labels.eq(label).mean() * 100), 1)


def _leading_emotion(group: pd.DataFrame) -> tuple[str, float]:
    present = [column for column in EMOTION_COLUMNS if column in group.columns]
    if not present:
        return "n/a", 0.0
    averages = group[present].apply(pd.to_numeric, errors="coerce").fillna(0).mean()
    if averages.empty or float(averages.max()) <= 0:
        return "n/a", 0.0
    column = str(averages.idxmax())
    return column.removeprefix("nrc_").replace("_", " ").title(), float(averages[column])


def summarize_topics(
    frame: pd.DataFrame,
    *,
    label_column: str = "corrected_label",
) -> pd.DataFrame:
    """Summarize topic volume, sentiment, and leading emotion."""
    columns = [
        "topic",
        "comment_count",
        "share_percent",
        "negative_percent",
        "neutral_percent",
        "positive_percent",
        "leading_emotion",
        "emotion_score",
        "average_frustration",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    labeled = frame if "topic" in frame.columns else add_topic_labels(frame)
    total = len(labeled)
    rows: list[dict[str, object]] = []
    for topic, group in labeled.groupby("topic", dropna=False):
        leading_emotion, emotion_score = _leading_emotion(group)
        frustration = (
            pd.to_numeric(group["nrc_frustration"], errors="coerce").fillna(0).mean()
            if "nrc_frustration" in group.columns
            else 0.0
        )
        rows.append(
            {
                "topic": str(topic),
                "comment_count": int(len(group)),
                "share_percent": round(len(group) / total * 100, 1),
                "negative_percent": _label_percent(group, label_column, "negative"),
                "neutral_percent": _label_percent(group, label_column, "neutral"),
                "positive_percent": _label_percent(group, label_column, "positive"),
                "leading_emotion": leading_emotion,
                "emotion_score": round(emotion_score, 3),
                "average_frustration": round(float(frustration), 3),
            }
        )
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["comment_count", "negative_percent"], ascending=[False, False])
        .reset_index(drop=True)
    )
