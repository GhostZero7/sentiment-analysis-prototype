"""Evidence-linked recommendations and stakeholder report generation."""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pandas as pd

from src.insights.topic_analyzer import EMOTION_COLUMNS, add_topic_labels, summarize_topics
from src.temporal.event_tracker import (
    build_comment_progression,
    build_sentiment_trends,
    describe_comment_progression,
    describe_negative_trend,
)


TOPIC_RECOMMENDATIONS: dict[str, str] = {
    "Load shedding and reliability": (
        "Publish consistent load-shedding schedules, explain unavoidable changes promptly, "
        "and provide restoration updates that communities can verify."
    ),
    "Tariffs and affordability": (
        "Pair tariff or pricing communication with plain-language cost examples, affordability "
        "impact analysis, and accessible public consultation channels."
    ),
    "Renewable energy and solar": (
        "Explain renewable-energy costs, financing options, expected reliability benefits, and "
        "realistic implementation timelines using locally relevant examples."
    ),
    "Customer service and communication": (
        "Set clear response standards for public enquiries and publish short, regular service "
        "updates that address the questions appearing most often."
    ),
    "Faults, connections and infrastructure": (
        "Separate fault, connection, and maintenance concerns in public reporting and communicate "
        "ownership, escalation routes, and expected resolution times."
    ),
    "Governance and public trust": (
        "Support announcements with progress measures, named institutional responsibilities, and "
        "follow-up evidence that allows the public to assess delivery."
    ),
    "Jobs and economic impact": (
        "Include expected employment, local procurement, and small-business impacts when "
        "communicating energy initiatives."
    ),
    "Environment and climate": (
        "Connect climate and sustainability messages to Zambia's lived concerns, including drought, "
        "hydropower risk, affordability, and electricity reliability."
    ),
    "Other energy concerns": (
        "Review representative comments in this emerging theme before designing a targeted response."
    ),
}

PRIORITY_ORDER = {"High": 0, "Medium": 1, "Monitor": 2}


def build_policy_recommendations(
    topic_summary: pd.DataFrame,
    *,
    trend_data: pd.DataFrame | None = None,
    total_comments: int | None = None,
    limit: int = 6,
) -> pd.DataFrame:
    """Build transparent communication and policy considerations from aggregated evidence."""
    columns = ["priority", "topic", "evidence", "recommendation"]
    if topic_summary.empty:
        return pd.DataFrame(columns=columns)

    total = int(total_comments or topic_summary["comment_count"].sum())
    minimum_evidence = max(3, math.ceil(total * 0.01))
    rows: list[dict[str, str]] = []

    for _, topic_row in topic_summary.iterrows():
        count = int(topic_row["comment_count"])
        if count < minimum_evidence:
            continue
        share = float(topic_row["share_percent"])
        negative = float(topic_row["negative_percent"])
        frustration = float(topic_row.get("average_frustration", 0.0))
        if negative >= 60 and share >= 5:
            priority = "High"
        elif negative >= 40 or share >= 15:
            priority = "Medium"
        else:
            priority = "Monitor"

        topic = str(topic_row["topic"])
        evidence = (
            f"{count} comments ({share:.1f}% of the selected data); "
            f"{negative:.1f}% negative sentiment"
        )
        if frustration >= 0.05:
            evidence += f"; frustration index {frustration:.3f}"
        evidence += "."
        rows.append(
            {
                "priority": priority,
                "topic": topic,
                "evidence": evidence,
                "recommendation": TOPIC_RECOMMENDATIONS.get(
                    topic,
                    TOPIC_RECOMMENDATIONS["Other energy concerns"],
                ),
            }
        )

    if trend_data is not None and len(trend_data) >= 2:
        change = float(
            trend_data.iloc[-1]["negative_percent"]
            - trend_data.iloc[0]["negative_percent"]
        )
        if change >= 10:
            rows.append(
                {
                    "priority": "High",
                    "topic": "Rising negative sentiment",
                    "evidence": (
                        f"Negative sentiment increased by {change:.1f} percentage points "
                        "across the selected period."
                    ),
                    "recommendation": (
                        "Review the comments and events behind the increase, then issue a focused "
                        "response addressing the most prominent concern before the next monitoring cycle."
                    ),
                }
            )

    recommendations = pd.DataFrame(rows, columns=columns)
    if recommendations.empty:
        return recommendations
    recommendations["_priority_order"] = recommendations["priority"].map(PRIORITY_ORDER)
    return (
        recommendations.sort_values(
            ["_priority_order", "topic"],
            ascending=[True, True],
        )
        .drop(columns="_priority_order")
        .head(limit)
        .reset_index(drop=True)
    )


def _sentiment_counts(frame: pd.DataFrame, label_column: str) -> pd.DataFrame:
    if label_column not in frame.columns:
        return pd.DataFrame(columns=["sentiment", "comments", "percent"])
    labels = frame[label_column].fillna("unknown").astype(str).str.lower()
    counts = labels.value_counts()
    ordered = [label for label in ("negative", "neutral", "positive", "unknown") if label in counts]
    return pd.DataFrame(
        {
            "sentiment": ordered,
            "comments": [int(counts[label]) for label in ordered],
            "percent": [round(float(counts[label] / len(frame) * 100), 1) for label in ordered],
        }
    )


def _emotion_summary(frame: pd.DataFrame) -> pd.DataFrame:
    present = [column for column in EMOTION_COLUMNS if column in frame.columns]
    if not present:
        return pd.DataFrame(columns=["emotion", "average_score"])
    averages = (
        frame[present]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .mean()
        .sort_values(ascending=False)
    )
    return pd.DataFrame(
        {
            "emotion": [column.removeprefix("nrc_").title() for column in averages.index],
            "average_score": [round(float(value), 3) for value in averages.values],
        }
    )


def _date_range(frame: pd.DataFrame) -> str:
    for column in ("timestamp", "analysis_timestamp", "created_at", "date"):
        if column not in frame.columns:
            continue
        values = pd.to_datetime(frame[column], errors="coerce", utc=True).dropna()
        if values.empty:
            continue
        start = values.min().date().isoformat()
        end = values.max().date().isoformat()
        return start if start == end else f"{start} to {end}"
    return "Date information unavailable"


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    if frame.empty:
        return ["No data available."]
    labels = [column.replace("_", " ").title() for column in columns]
    lines = [
        "| " + " | ".join(labels) + " |",
        "|" + "|".join("---" for _ in labels) + "|",
    ]
    for _, row in frame[columns].iterrows():
        values = [str(row[column]).replace("|", "/") for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def build_stakeholder_report(
    frame: pd.DataFrame,
    *,
    source_label: str,
    generated_at: datetime | None = None,
) -> str:
    """Create a downloadable Markdown report for the selected dashboard data."""
    generated = generated_at or datetime.now(timezone.utc)
    label_column = "corrected_label" if "corrected_label" in frame.columns else "label"
    labeled = frame if "topic" in frame.columns else add_topic_labels(frame)
    topics = summarize_topics(labeled, label_column=label_column)
    trends = build_sentiment_trends(labeled, frequency="Day", label_column=label_column)
    progression = build_comment_progression(labeled, label_column=label_column)
    if len(trends) >= 2:
        movement_summary = describe_negative_trend(trends)
        recommendation_trends = trends
    else:
        movement_summary = describe_comment_progression(progression)
        recommendation_trends = None
    recommendations = build_policy_recommendations(
        topics,
        trend_data=recommendation_trends,
        total_comments=len(labeled),
    )
    sentiment = _sentiment_counts(labeled, label_column)
    emotions = _emotion_summary(labeled)

    dominant_sentiment = (
        str(sentiment.sort_values("comments", ascending=False).iloc[0]["sentiment"]).title()
        if not sentiment.empty
        else "Unavailable"
    )
    top_topic = str(topics.iloc[0]["topic"]) if not topics.empty else "Unavailable"
    leading_emotion = str(emotions.iloc[0]["emotion"]) if not emotions.empty else "Unavailable"

    lines = [
        "# Zambian Green Energy Public Sentiment Report",
        "",
        f"**Generated:** {generated.astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Data source:** {source_label}",
        f"**Coverage:** {_date_range(labeled)}",
        f"**Comments analyzed:** {len(labeled):,}",
        "",
        "## Executive Summary",
        "",
        f"- Dominant sentiment: **{dominant_sentiment}**.",
        f"- Most discussed topic: **{top_topic}**.",
        f"- Leading detected emotion: **{leading_emotion}**.",
        f"- {movement_summary}",
        "",
        "## Sentiment Distribution",
        "",
        *_markdown_table(sentiment, ["sentiment", "comments", "percent"]),
        "",
        "## Dominant Topics",
        "",
        *_markdown_table(
            topics.head(8),
            ["topic", "comment_count", "share_percent", "negative_percent", "leading_emotion"],
        ),
        "",
        "## Emotion Index",
        "",
        *_markdown_table(emotions, ["emotion", "average_score"]),
        "",
        "## Policy and Communication Considerations",
        "",
    ]

    if recommendations.empty:
        lines.append(
            "The selected data does not yet contain enough repeated evidence for a targeted recommendation."
        )
    else:
        for index, row in recommendations.iterrows():
            lines.extend(
                [
                    f"### {index + 1}. {row['topic']} ({row['priority']} priority)",
                    "",
                    f"**Evidence:** {row['evidence']}",
                    "",
                    f"**Recommended response:** {row['recommendation']}",
                    "",
                ]
            )

    lines.extend(
        [
            "## Interpretation Limits",
            "",
            "- Social-media comments are unsolicited public reactions and are not a representative population survey.",
            "- Topic groups are assigned using a transparent Zambia-energy keyword taxonomy and should be reviewed when new language patterns emerge.",
            "- Sentiment and emotion outputs are automated estimates; high-impact decisions should include human review and supporting operational evidence.",
            "- Recommendations are decision-support considerations, not automatic policy decisions.",
            "",
        ]
    )
    return "\n".join(lines)
