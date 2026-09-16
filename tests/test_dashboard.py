from __future__ import annotations

import pandas as pd

from src.dashboard.app import (
    RESPONSIVE_CSS,
    _comment_emotion_table,
    _dashboard_tabs,
    _display_periods,
    _friendly_error_message,
    _readable_trend_table,
    _sentiment_change_explanation,
    _sentiment_change_table,
)
from src.data_collection.apify_client import ApifyFetchError


def test_topic_comment_audit_includes_topic_and_emotion_evidence():
    frame = pd.DataFrame(
        [
            {
                "text": "The load shedding schedule is inconsistent",
                "topic": "Load shedding and reliability",
                "corrected_label": "negative",
                "nrc_trust": 0.0,
                "nrc_hope": 0.0,
                "nrc_frustration": 0.6,
                "nrc_emotion_terms": "inconsistent (anger)",
                "local_emotion_terms": "",
                "positive_emotion_suppressed": True,
            }
        ]
    )

    table = _comment_emotion_table(frame, "corrected_label")

    assert table.iloc[0]["Topic"] == "Load shedding & reliability"
    assert table.iloc[0]["Sentiment"] == "Negative"
    assert table.iloc[0]["Dominant emotion"] == "Frustration"
    assert table.iloc[0]["Frustration %"] == 60.0
    assert table.iloc[0]["Matched emotion words"] == "inconsistent (anger)"


def test_user_facing_errors_hide_technical_details():
    apify_message = _friendly_error_message(
        ApifyFetchError("HTTP 402 with private diagnostic details")
    )
    generic_message = _friendly_error_message(
        RuntimeError("database path and stack details"),
        fallback="The comments could not be analyzed. Please try again.",
    )

    assert apify_message == (
        "Apify tokens are depleted. Please try again after the tokens are renewed."
    )
    assert "402" not in apify_message
    assert generic_message == "The comments could not be analyzed. Please try again."
    assert "database" not in generic_message


def test_common_input_and_storage_errors_are_actionable():
    assert _friendly_error_message(ValueError("raw details")) == (
        "Enter a valid public Facebook post URL and try again."
    )
    assert "administrator" in _friendly_error_message(FileNotFoundError("model.bin"))
    assert "could not be saved" in _friendly_error_message(PermissionError("denied"))


def test_dashboard_css_includes_mobile_layout_rules():
    assert "@media (max-width: 768px)" in RESPONSIVE_CSS
    assert 'data-testid="stHorizontalBlock"' in RESPONSIVE_CSS
    assert "flex-direction: column" in RESPONSIVE_CSS
    assert 'data-baseweb="tab-list"' in RESPONSIVE_CSS
    assert 'data-testid="stFormSubmitButton"' in RESPONSIVE_CSS
    assert "overflow-x: auto" in RESPONSIVE_CSS
    assert "min-height: 2.75rem" in RESPONSIVE_CSS


def test_dashboard_tabs_preserve_the_selected_view_on_button_reruns(monkeypatch):
    captured: dict[str, object] = {}
    expected_tabs = [object() for _ in range(6)]

    def fake_tabs(labels, **kwargs):
        captured["labels"] = labels
        captured.update(kwargs)
        return expected_tabs

    monkeypatch.setattr("src.dashboard.app.st.tabs", fake_tabs)

    assert _dashboard_tabs() == expected_tabs
    assert captured["labels"] == [
        "Analyze URL",
        "Overview",
        "Trends",
        "Topics",
        "Comments",
        "Report",
    ]
    assert captured["key"] == "active_dashboard_tab"
    assert captured["on_change"] == "rerun"


def test_trend_explanation_states_exact_sentiment_changes_in_plain_language():
    trends = pd.DataFrame(
        [
            {
                "period": "Comments 1-8",
                "comment_count": 8,
                "negative_percent": 37.5,
                "neutral_percent": 25.0,
                "positive_percent": 37.5,
            },
            {
                "period": "Comments 41-47",
                "comment_count": 7,
                "negative_percent": 71.4,
                "neutral_percent": 0.0,
                "positive_percent": 28.6,
            },
        ]
    )

    explanation = _sentiment_change_explanation(trends, temporal=False)
    changes = _sentiment_change_table(trends)

    assert "became more negative" in explanation
    assert "37.5% to 71.4%" in explanation
    assert "+33.9 percentage points" in explanation
    assert "neutral sentiment fell by 25.0 points" in explanation
    assert "positive sentiment fell by 8.9 points" in explanation
    assert changes.loc[changes["Sentiment"].eq("Negative"), "Meaning"].iloc[0] == "Increased"


def test_readable_trend_table_replaces_raw_emotion_columns():
    trends = pd.DataFrame(
        [
            {
                "period": "Comments 1-4",
                "comment_count": 4,
                "negative_percent": 50.0,
                "neutral_percent": 25.0,
                "positive_percent": 25.0,
                "nrc_frustration": 0.2,
                "nrc_anger": 0.1,
            },
            {
                "period": "Comments 5-8",
                "comment_count": 4,
                "negative_percent": 25.0,
                "neutral_percent": 25.0,
                "positive_percent": 50.0,
                "nrc_hope": 0.3,
            },
        ]
    )

    readable = _readable_trend_table(trends, temporal=False)

    assert _display_periods(trends, temporal=False) == [
        "Beginning: Comments 1-4",
        "Recent: Comments 5-8",
    ]
    assert "nrc_frustration" not in readable.columns
    assert readable["Leading emotion"].tolist() == ["Frustration", "Hope"]
    assert readable["Leading sentiment"].tolist() == ["Negative (50.0%)", "Positive (50.0%)"]
