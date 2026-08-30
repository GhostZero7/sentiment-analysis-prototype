"""Temporal aggregation helpers for public sentiment monitoring."""

from __future__ import annotations

import pandas as pd


TIMESTAMP_COLUMNS = ("timestamp", "analysis_timestamp", "created_at", "date")
FREQUENCIES = {
    "Hour": "h",
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


def resolve_trend_frequency(frame: pd.DataFrame) -> str:
    """Choose the smallest useful calendar grouping for the available discussion span."""
    prepared = add_event_time(frame)
    valid = prepared["_event_time"].dropna().sort_values()
    if valid.empty:
        return "Day"
    span = valid.max() - valid.min()
    if span <= pd.Timedelta(days=2):
        return "Hour"
    if span <= pd.Timedelta(days=120):
        return "Day"
    if span <= pd.Timedelta(days=730):
        return "Week"
    return "Month"


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

    resolved_frequency = (
        resolve_trend_frequency(prepared) if frequency == "Automatic" else frequency
    )
    period_frequency = FREQUENCIES.get(resolved_frequency, FREQUENCIES["Day"])
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


def build_trend_events(
    frame: pd.DataFrame,
    trends: pd.DataFrame,
    *,
    frequency: str = "Automatic",
    label_column: str = "corrected_label",
    limit: int = 3,
) -> pd.DataFrame:
    """Explain the strongest period-to-period changes using discussion evidence."""
    columns = ["period", "movement", "evidence", "representative_comment"]
    if frame.empty or len(trends) < 2:
        return pd.DataFrame(columns=columns)

    prepared = add_event_time(frame)
    prepared = prepared[prepared["_event_time"].notna()].copy()
    if prepared.empty:
        return pd.DataFrame(columns=columns)
    resolved = resolve_trend_frequency(prepared) if frequency == "Automatic" else frequency
    period_frequency = FREQUENCIES.get(resolved, FREQUENCIES["Day"])
    prepared["_period"] = (
        prepared["_event_time"].dt.tz_convert(None).dt.to_period(period_frequency).dt.start_time
    )

    candidates: list[dict[str, object]] = []
    for index in range(1, len(trends)):
        previous = trends.iloc[index - 1]
        current = trends.iloc[index]
        negative_change = float(current["negative_percent"] - previous["negative_percent"])
        volume_change = int(current["comment_count"] - previous["comment_count"])
        score = abs(negative_change) + min(abs(volume_change), 100) * 0.1
        if abs(negative_change) < 5 and volume_change <= 0:
            continue

        period = pd.Timestamp(current["period"])
        group = prepared[prepared["_period"].eq(period)].copy()
        topic = ""
        if "topic" in group.columns:
            topics = group["topic"].fillna("").astype(str)
            topics = topics[topics.str.strip().ne("")]
            if not topics.empty:
                topic = str(topics.value_counts().index[0])

        emotion = ""
        present_emotions = [column for column in EMOTION_COLUMNS if column in group.columns]
        if present_emotions:
            averages = group[present_emotions].apply(pd.to_numeric, errors="coerce").fillna(0).mean()
            if not averages.empty and float(averages.max()) > 0:
                emotion = str(averages.idxmax()).removeprefix("nrc_").replace("_", " ").title()

        if abs(negative_change) >= 5:
            direction = "rose" if negative_change > 0 else "fell"
            movement = f"Negative sentiment {direction} by {abs(negative_change):.1f} points"
        else:
            movement = f"Comment volume increased by {volume_change}"
        evidence_parts = [
            f"{int(current['comment_count'])} comments in this period",
            f"volume changed by {volume_change:+d} from the previous period",
        ]
        if topic:
            evidence_parts.append(f"the leading discussion topic was {topic}")
        if emotion:
            evidence_parts.append(f"the leading detected emotion was {emotion}")

        representative = ""
        text_column = next(
            (
                column
                for column in ("text_raw", "text", "text_clean", "processed_text")
                if column in group.columns
            ),
            None,
        )
        if text_column:
            preferred_label = "negative" if negative_change > 0 else "positive"
            sample = group
            if label_column in group.columns:
                labels = group[label_column].fillna("").astype(str).str.lower()
                preferred = group[labels.eq(preferred_label)]
                if not preferred.empty:
                    sample = preferred
            values = sample[text_column].fillna("").astype(str).str.strip()
            values = values[values.ne("")]
            if not values.empty:
                representative = values.iloc[0][:220]

        candidates.append(
            {
                "period": period,
                "movement": movement,
                "evidence": "; ".join(evidence_parts) + ".",
                "representative_comment": representative,
                "_score": score,
            }
        )

    if not candidates:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(candidates)
        .sort_values(["_score", "period"], ascending=[False, True])
        .head(max(1, int(limit)))
        .drop(columns="_score")
        .reset_index(drop=True)
    )


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
