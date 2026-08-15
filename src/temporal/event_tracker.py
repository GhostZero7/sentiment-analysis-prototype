"""Temporal aggregation helpers for public sentiment monitoring."""

from __future__ import annotations

import pandas as pd


TIMESTAMP_COLUMNS = ("timestamp", "analysis_timestamp", "created_at", "date")
FREQUENCIES = {
    "Day": "D",
    "Week": "W-SUN",
    "Month": "M",
}
EMOTION_COLUMNS = (
    "nrc_anger",
    "nrc_fear",
    "nrc_trust",
    "nrc_sadness",
    "nrc_hope",
    "nrc_frustration",
)
TREND_COLUMNS = [
    "period",
    "comment_count",
    "negative_percent",
    "neutral_percent",
    "positive_percent",
    "sarcasm_percent",
    *EMOTION_COLUMNS,
]


def find_timestamp_column(frame: pd.DataFrame) -> str | None:
    """Return the first timestamp column containing at least one valid value."""
    for column in TIMESTAMP_COLUMNS:
        if column not in frame.columns:
            continue
        parsed = pd.to_datetime(frame[column], errors="coerce", utc=True)
        if parsed.notna().any():
            return column
    return None


def add_event_time(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with parsed UTC timestamps in `_event_time`."""
    prepared = frame.copy()
    column = find_timestamp_column(prepared)
    if column is None:
        prepared["_event_time"] = pd.NaT
    else:
        prepared["_event_time"] = pd.to_datetime(
            prepared[column],
            errors="coerce",
            utc=True,
        )
    return prepared


def build_sentiment_trends(
    frame: pd.DataFrame,
    *,
    frequency: str = "Day",
    label_column: str = "corrected_label",
) -> pd.DataFrame:
    """Aggregate volume, sentiment share, sarcasm, and emotions over time."""
    if frame.empty or label_column not in frame.columns:
        return pd.DataFrame(columns=TREND_COLUMNS)

    prepared = add_event_time(frame)
    prepared = prepared[prepared["_event_time"].notna()].copy()
    if prepared.empty:
        return pd.DataFrame(columns=TREND_COLUMNS)

    period_frequency = FREQUENCIES.get(frequency, FREQUENCIES["Day"])
    local_time = prepared["_event_time"].dt.tz_convert(None)
    prepared["_period"] = local_time.dt.to_period(period_frequency).dt.start_time
    prepared["_label"] = (
        prepared[label_column]
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    rows = [
        _summarize_group(group, label_column="_label", period=period)
        for period, group in prepared.groupby("_period", sort=True)
    ]
    return pd.DataFrame(rows, columns=TREND_COLUMNS)


def _summarize_group(
    group: pd.DataFrame,
    *,
    label_column: str,
    period: object,
) -> dict[str, object]:
    count = len(group)
    labels = group[label_column].fillna("unknown").astype(str).str.strip().str.lower()
    row: dict[str, object] = {"period": period, "comment_count": int(count)}
    for label in ("negative", "neutral", "positive"):
        row[f"{label}_percent"] = round(float(labels.eq(label).mean() * 100), 1)

    if "is_sarcastic" in group.columns:
        sarcasm = group["is_sarcastic"]
        if sarcasm.dtype != bool:
            sarcasm = sarcasm.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})
        row["sarcasm_percent"] = round(float(sarcasm.mean() * 100), 1)
    else:
        row["sarcasm_percent"] = 0.0

    for column in EMOTION_COLUMNS:
        row[column] = (
            round(float(pd.to_numeric(group[column], errors="coerce").fillna(0).mean()), 3)
            if column in group.columns
            else 0.0
        )
    return row


def build_comment_progression(
    frame: pd.DataFrame,
    *,
    segments: int = 6,
    label_column: str = "corrected_label",
) -> pd.DataFrame:
    """Aggregate sentiment across ordered comment groups when time resolution is limited."""
    if frame.empty or label_column not in frame.columns:
        return pd.DataFrame(columns=TREND_COLUMNS)

    prepared = add_event_time(frame).reset_index(drop=True)
    prepared["_source_order"] = range(len(prepared))
    if prepared["_event_time"].nunique(dropna=True) > 1:
        prepared = prepared.sort_values(
            ["_event_time", "_source_order"],
            kind="stable",
            na_position="last",
        ).reset_index(drop=True)

    group_count = max(1, min(int(segments), len(prepared)))
    prepared["_segment"] = [index * group_count // len(prepared) for index in range(len(prepared))]
    rows: list[dict[str, object]] = []
    for segment, group in prepared.groupby("_segment", sort=True):
        start = int(group.index.min()) + 1
        end = int(group.index.max()) + 1
        rows.append(
            _summarize_group(
                group,
                label_column=label_column,
                period=f"Comments {start}-{end}",
            )
        )
    return pd.DataFrame(rows, columns=TREND_COLUMNS)


def describe_negative_trend(trends: pd.DataFrame) -> str:
    """Describe the change in negative sentiment between the first and last period."""
    if trends.empty:
        return "No valid timestamps were available for trend analysis."
    if len(trends) < 2:
        return "Only one time period is available, so a direction of change cannot yet be established."

    first = float(trends.iloc[0]["negative_percent"])
    last = float(trends.iloc[-1]["negative_percent"])
    change = round(last - first, 1)
    if abs(change) < 2:
        return f"Negative sentiment remained broadly stable ({change:+.1f} percentage points)."
    direction = "increased" if change > 0 else "decreased"
    return f"Negative sentiment {direction} by {abs(change):.1f} percentage points across the selected period."


def describe_comment_progression(progression: pd.DataFrame) -> str:
    """Describe negative sentiment movement between the first and last comment groups."""
    if progression.empty:
        return "There are not enough comments to calculate discussion progression."
    if len(progression) < 2:
        return "Only one comment group is available, so movement cannot be estimated."
    first = float(progression.iloc[0]["negative_percent"])
    last = float(progression.iloc[-1]["negative_percent"])
    change = round(last - first, 1)
    if abs(change) < 2:
        return f"Negative sentiment stayed broadly stable across the discussion ({change:+.1f} points)."
    direction = "rose" if change > 0 else "fell"
    return f"Negative sentiment {direction} by {abs(change):.1f} points from the first to last comment group."
