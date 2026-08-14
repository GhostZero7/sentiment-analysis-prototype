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
    monkeypatch.setattr(live_analysis, "LIVE_LINKS_PATH", tmp_path / "url_links.csv")

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
        {
            "comment_id": "3",
            "text": (
                "We don't want stories about stabilization because yesterday "
                "we didn't have power from Zesco."
            ),
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
    assert len(analyzed) == 3
    assert analyzed["is_relevant"].sum() == 2
    assert set(["corrected_label", "nrc_anger", "logistic_regression_prediction"]).issubset(analyzed.columns)
    negated_outage = analyzed.loc[analyzed["comment_id"].astype(str).eq("3")].iloc[0]
    assert negated_outage["corrected_label"] == "negative"

    first = live_analysis.save_live_analysis(analyzed, source_url="https://www.facebook.com/test")
    second = live_analysis.save_live_analysis(analyzed, source_url="https://www.facebook.com/test")

    assert first["history_rows"] == 3
    assert second["history_rows"] == 3
    assert second["training_candidate_rows"] == 2


def test_live_analysis_keeps_short_energy_complaint(tmp_path, monkeypatch):
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
    analyzed = live_analysis.analyze_comments_frame(
        [
            {
                "comment_id": "short-1",
                "text": "Which electricity?",
                "timestamp": "2026-01-01T00:00:00Z",
                "source_url": "https://www.facebook.com/test",
            }
        ],
        source_url="https://www.facebook.com/test",
        models_dir=tmp_path,
    )
    assert len(analyzed) == 1
    assert analyzed.iloc[0]["label"] == "negative"
    assert bool(analyzed.iloc[0]["is_relevant"]) is True


def test_link_registry_dedupes_by_source_url(tmp_path, monkeypatch):
    monkeypatch.setattr(live_analysis, "LIVE_LINKS_PATH", tmp_path / "url_links.csv")
    first = live_analysis._save_link_record(
        {
            "source_url": "https://www.facebook.com/test",
            "url_hash": "abc",
            "last_analyzed_at": "2026-01-01T00:00:00+00:00",
            "fetched_rows": 10,
            "analyzed_rows": 8,
            "relevant_rows": 7,
            "raw_fetch_path": "raw-old.csv",
            "single_analysis_path": "analysis-old.csv",
            "history_path": "history.csv",
            "training_candidates_path": "candidates.csv",
        }
    )
    second = live_analysis._save_link_record(
        {
            "source_url": "https://www.facebook.com/test",
            "url_hash": "abc",
            "last_analyzed_at": "2026-01-02T00:00:00+00:00",
            "fetched_rows": 12,
            "analyzed_rows": 9,
            "relevant_rows": 8,
            "raw_fetch_path": "raw-new.csv",
            "single_analysis_path": "analysis-new.csv",
            "history_path": "history.csv",
            "training_candidates_path": "candidates.csv",
        }
    )
    assert len(first) == 1
    assert len(second) == 1
    assert second.iloc[0]["raw_fetch_path"] == "raw-new.csv"


def test_cached_post_is_loaded_by_content_id_and_respects_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(live_analysis, "RAW_DIR", tmp_path)
    monkeypatch.setattr(live_analysis, "LIVE_RAW_DIR", tmp_path / "url_fetches")
    cached_path = tmp_path / "comments_post_123456.csv"
    pd.DataFrame(
        [
            {
                "comment_id": str(index),
                "text": f"Zesco power comment {index}",
                "timestamp": "2026-01-01T00:00:00Z",
                "source_url": "https://www.facebook.com/page/posts/123456/?app=fbl",
            }
            for index in range(125)
        ]
    ).to_csv(cached_path, index=False)

    cached = live_analysis.load_cached_comments(
        "https://m.facebook.com/page/posts/123456/?app=fbl",
        max_comments=500,
    )

    assert cached is not None
    assert cached.path == cached_path
    assert cached.available_rows == 125
    assert len(cached.comments) == 100
    assert all(
        row["source_url"] == "https://www.facebook.com/page/posts/123456/"
        for row in cached.comments
    )


def test_analyze_facebook_url_uses_cache_without_calling_apify(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    cached_path = raw_dir / "comments_post_987654.csv"
    pd.DataFrame(
        [
            {
                "comment_id": "1",
                "text": "Zesco restored power today",
                "timestamp": "2026-01-01T00:00:00Z",
                "source_url": "https://www.facebook.com/page/posts/987654/",
            }
        ]
    ).to_csv(cached_path, index=False)

    monkeypatch.setattr(live_analysis, "RAW_DIR", raw_dir)
    monkeypatch.setattr(live_analysis, "LIVE_RAW_DIR", raw_dir / "url_fetches")
    monkeypatch.setattr(
        live_analysis,
        "fetch_comments",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Apify should not run")),
    )
    monkeypatch.setattr(
        live_analysis,
        "analyze_comments_frame",
        lambda comments, **kwargs: pd.DataFrame(
            {"comment_id": [row["comment_id"] for row in comments], "is_relevant": [True]}
        ),
    )
    monkeypatch.setattr(
        live_analysis,
        "save_live_analysis",
        lambda analyzed, source_url: {
            "single_analysis_path": "analysis.csv",
            "history_path": "history.csv",
            "training_candidates_path": "candidates.csv",
            "history_rows": len(analyzed),
            "training_candidate_rows": len(analyzed),
        },
    )
    saved_records: list[dict[str, object]] = []

    def fake_save_link(record):
        saved_records.append(record)
        return pd.DataFrame([record])

    monkeypatch.setattr(live_analysis, "_save_link_record", fake_save_link)

    analyzed, summary = live_analysis.analyze_facebook_url(
        "https://www.facebook.com/page/posts/987654/?app=fbl",
        models_dir=tmp_path,
        max_comments=100,
    )

    assert len(analyzed) == 1
    assert summary["cache_hit"] is True
    assert summary["collection_source"] == "local_cache"
    assert summary["raw_fetch_path"] == str(cached_path)
    assert summary["fetched_rows"] == 1
    assert saved_records[0]["cache_hit"] is True


def test_link_registry_dedupes_tracking_variants_by_content_id(tmp_path, monkeypatch):
    monkeypatch.setattr(live_analysis, "LIVE_LINKS_PATH", tmp_path / "url_links.csv")
    common = {
        "url_hash": "abc",
        "fetched_rows": 10,
        "analyzed_rows": 8,
        "relevant_rows": 7,
        "raw_fetch_path": "raw.csv",
        "single_analysis_path": "analysis.csv",
        "history_path": "history.csv",
        "training_candidates_path": "candidates.csv",
    }
    live_analysis._save_link_record(
        {
            **common,
            "source_url": "https://www.facebook.com/page/posts/123/?app=fbl",
            "last_analyzed_at": "2026-01-01T00:00:00+00:00",
        }
    )
    second = live_analysis._save_link_record(
        {
            **common,
            "source_url": "https://m.facebook.com/page/posts/123/",
            "last_analyzed_at": "2026-01-02T00:00:00+00:00",
        }
    )

    assert len(second) == 1
    assert second.iloc[0]["content_id"] == "123"
