from __future__ import annotations

import pandas as pd

from src.dashboard import live_analysis


class _DummyModel:
    def __init__(self, label: str):
        self.label = label

    def predict(self, x_vec):
        return [self.label for _ in range(x_vec.shape[0])]


class _DummyVectorizer:
    def transform(self, texts):
        return pd.DataFrame({"text": list(texts)})


def test_live_analysis_scores_comments_and_dedupes_saved_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(
        live_analysis,
        "predict_texts",
        lambda texts, models_dir: pd.DataFrame(
            {
                "text": texts,
                "naive_bayes_prediction": ["negative"] * len(texts),
                "logistic_regression_prediction": ["negative"] * len(texts),
                "svm_prediction": ["negative"] * len(texts),
            }
        ),
    )
    monkeypatch.setattr(live_analysis, "LIVE_RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(live_analysis, "LIVE_HISTORY_PATH", tmp_path / "history.csv")
    monkeypatch.setattr(live_analysis, "LIVE_TRAINING_CANDIDATES_PATH", tmp_path / "candidates.csv")

    comments = [
        {
            "comment_id": "1",
            "text": "Ba Zesco fyabupuba this power is gone again",
            "timestamp": "2026-01-01T00:00:00Z",
            "source_url": "https://www.facebook.com/test",
        },
        {
            "comment_id": "2",
            "text": "Happy birthday my friend enjoy your day",
            "timestamp": "2026-01-01T00:00:00Z",
            "source_url": "https://www.facebook.com/test",
        },
    ]

    analyzed = live_analysis.analyze_comments_frame(
        comments,
        source_url="https://www.facebook.com/test",
        models_dir=tmp_path,
        batch_id="test-batch",
    )
    assert len(analyzed) == 2
    assert analyzed["is_relevant"].sum() == 1
    assert set(["corrected_label", "nrc_anger", "logistic_regression_prediction"]).issubset(analyzed.columns)

    first = live_analysis.save_live_analysis(analyzed, source_url="https://www.facebook.com/test")
    second = live_analysis.save_live_analysis(analyzed, source_url="https://www.facebook.com/test")

    assert first["history_rows"] == 2
    assert second["history_rows"] == 2
    assert second["training_candidate_rows"] == 1
