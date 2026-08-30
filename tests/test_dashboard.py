from __future__ import annotations

import pandas as pd

from src.dashboard.app import _comment_emotion_table, _friendly_error_message
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
