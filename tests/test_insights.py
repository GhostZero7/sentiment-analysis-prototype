from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.insights.policy_report import (
    build_policy_recommendations,
    build_stakeholder_report,
)
from src.insights.pdf_report import build_executive_summary_pdf
from src.insights.topic_analyzer import add_topic_labels, assign_topic, summarize_topics
from src.temporal.event_tracker import (
    build_comment_progression,
    build_sentiment_trends,
    build_trend_events,
    describe_comment_progression,
    describe_negative_trend,
    resolve_trend_frequency,
)


def _sample_comments() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "text": "The new tariff is too expensive and families cannot afford it",
                "timestamp": "2026-01-01T08:00:00Z",
                "corrected_label": "negative",
                "is_sarcastic": False,
                "nrc_frustration": 0.4,
                "nrc_anger": 0.2,
                "nrc_hope": 0.0,
            },
            {
                "text": "Another blackout and the load shedding schedule was not followed",
                "timestamp": "2026-01-01T09:00:00Z",
                "corrected_label": "negative",
                "is_sarcastic": True,
                "nrc_frustration": 0.5,
                "nrc_anger": 0.3,
                "nrc_hope": 0.0,
            },
            {
                "text": "The solar project can improve electricity reliability",
                "timestamp": "2026-01-02T10:00:00Z",
                "corrected_label": "positive",
                "is_sarcastic": False,
                "nrc_frustration": 0.0,
                "nrc_anger": 0.0,
                "nrc_hope": 0.5,
            },
            {
                "text": "Please publish regular customer service updates",
                "timestamp": "2026-01-02T11:00:00Z",
                "corrected_label": "neutral",
                "is_sarcastic": False,
                "nrc_frustration": 0.1,
                "nrc_anger": 0.0,
                "nrc_hope": 0.1,
            },
        ]
    )


def test_assign_topic_uses_explainable_energy_taxonomy():
    assert assign_topic("We cannot afford another tariff increase") == "Tariffs and affordability"
    assert assign_topic("The load shedding schedule changed again") == "Load shedding and reliability"
    assert assign_topic("Solar batteries could help Zambia") == "Renewable energy and solar"


def test_topic_summary_counts_each_comment_once():
    labeled = add_topic_labels(_sample_comments())
    summary = summarize_topics(labeled)

    assert int(summary["comment_count"].sum()) == len(labeled)
    assert set(summary["topic"]) == {
        "Tariffs and affordability",
        "Load shedding and reliability",
        "Renewable energy and solar",
        "Customer service and communication",
    }
    tariff = summary[summary["topic"].eq("Tariffs and affordability")].iloc[0]
    assert tariff["negative_percent"] == 100.0


def test_trend_builder_aggregates_sentiment_and_describes_direction():
    trends = build_sentiment_trends(_sample_comments(), frequency="Day")

    assert len(trends) == 2
    assert trends.iloc[0]["comment_count"] == 2
    assert trends.iloc[0]["negative_percent"] == 100.0
    assert trends.iloc[1]["negative_percent"] == 0.0
    assert "decreased by 100.0 percentage points" in describe_negative_trend(trends)


def test_comment_progression_shows_movement_without_multiple_time_periods():
    frame = _sample_comments().copy()
    frame["timestamp"] = "2026-01-01T08:00:00Z"
    progression = build_comment_progression(frame, segments=2)

    assert progression["period"].tolist() == ["Comments 1-2", "Comments 3-4"]
    assert progression["negative_percent"].tolist() == [100.0, 0.0]
    assert "fell by 100.0 points" in describe_comment_progression(progression)


def test_automatic_trend_uses_hours_for_short_discussion_and_explains_change():
    frame = _sample_comments().copy()
    frame["timestamp"] = [
        "2026-01-01T08:00:00Z",
        "2026-01-01T08:30:00Z",
        "2026-01-01T10:00:00Z",
        "2026-01-01T10:30:00Z",
    ]
    frame["topic"] = [
        "Load shedding and reliability",
        "Load shedding and reliability",
        "Renewable energy and solar",
        "Renewable energy and solar",
    ]

    assert resolve_trend_frequency(frame) == "Hour"
    trends = build_sentiment_trends(frame, frequency="Automatic")
    events = build_trend_events(frame, trends, frequency="Automatic")

    assert len(trends) == 2
    assert len(events) == 1
    assert "Negative sentiment fell" in events.iloc[0]["movement"]
    assert "Renewable energy and solar" in events.iloc[0]["evidence"]
    assert events.iloc[0]["representative_comment"]


def test_policy_recommendations_include_auditable_evidence():
    topic_summary = pd.DataFrame(
        [
            {
                "topic": "Tariffs and affordability",
                "comment_count": 20,
                "share_percent": 50.0,
                "negative_percent": 75.0,
                "average_frustration": 0.2,
            },
            {
                "topic": "Renewable energy and solar",
                "comment_count": 20,
                "share_percent": 50.0,
                "negative_percent": 20.0,
                "average_frustration": 0.01,
            },
        ]
    )
    recommendations = build_policy_recommendations(topic_summary, total_comments=40)

    tariff = recommendations[recommendations["topic"].eq("Tariffs and affordability")].iloc[0]
    assert tariff["priority"] == "High"
    assert "20 comments" in tariff["evidence"]
    assert "75.0% negative sentiment" in tariff["evidence"]
    assert "affordability impact analysis" in tariff["recommendation"]


def test_stakeholder_report_contains_findings_recommendations_and_limits():
    report = build_stakeholder_report(
        _sample_comments(),
        source_label="Test dataset",
        generated_at=datetime(2026, 1, 3, tzinfo=timezone.utc),
    )

    assert "# Zambian Green Energy Public Sentiment Report" in report
    assert "**Data source:** Test dataset" in report
    assert "## Dominant Topics" in report
    assert "## Policy and Communication Considerations" in report
    assert "not a representative population survey" in report


def test_stakeholder_report_uses_comment_progression_for_one_period():
    frame = _sample_comments().copy()
    frame["timestamp"] = "2026-01-01T08:00:00Z"
    report = build_stakeholder_report(frame, source_label="Test dataset")

    assert "from the first to last comment group" in report


def test_executive_summary_pdf_is_generated():
    pdf = build_executive_summary_pdf(
        _sample_comments(),
        source_label="https://www.facebook.com/example/posts/123/",
        generated_at=datetime(2026, 1, 3, tzinfo=timezone.utc),
    )

    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 2_000
