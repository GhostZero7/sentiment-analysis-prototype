from __future__ import annotations

import pandas as pd

from src.dashboard.app import _comment_emotion_table


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
