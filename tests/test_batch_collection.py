from __future__ import annotations

import pandas as pd
import pytest

from scripts.collection import download_batch
from src.data_collection import apify_client


def test_content_id_and_canonical_url_ignore_tracking_query():
    url = "https://m.facebook.com/example/posts/123456/?app=fbl"
    assert download_batch.content_id(url) == "123456"
    assert download_batch.canonical_url(url) == "https://www.facebook.com/example/posts/123456/"


def test_batch_skips_existing_and_enforces_limit(tmp_path, monkeypatch):
    existing = tmp_path / "comments_post_111.csv"
    pd.DataFrame([{"comment_id": "old", "text": "old"}]).to_csv(existing, index=False)

    calls: list[tuple[str, int]] = []

    def fake_fetch(url: str, max_comments: int):
        calls.append((url, max_comments))
        return [
            {
                "comment_id": str(index),
                "text": f"comment {index}",
                "timestamp": "2026-01-01T00:00:00Z",
                "source_url": url,
            }
            for index in range(150)
        ]

    monkeypatch.setattr(download_batch, "fetch_comments", fake_fetch)
    manifest = tmp_path / "manifest.csv"
    results = download_batch.collect_batch(
        [
            "https://www.facebook.com/page/posts/111/?app=fbl",
            "https://www.facebook.com/page/posts/222/?app=fbl",
        ],
        limit=100,
        output_dir=tmp_path,
        manifest_path=manifest,
    )

    assert results["status"].tolist() == ["skipped_existing", "collected"]
    assert calls == [("https://www.facebook.com/page/posts/222/", 100)]
    assert len(pd.read_csv(tmp_path / "comments_post_222.csv")) == 100


def test_batch_rejects_limit_above_100(tmp_path):
    with pytest.raises(ValueError, match="between 1 and 100"):
        download_batch.collect_batch(
            ["https://www.facebook.com/page/posts/222/"],
            limit=101,
            output_dir=tmp_path,
            manifest_path=tmp_path / "manifest.csv",
        )


def test_shared_apify_client_caps_every_request_at_100(monkeypatch):
    captured: dict[str, object] = {}

    class DummyActor:
        def call(self, *, run_input, timeout_secs):
            captured["run_input"] = run_input
            return {"defaultDatasetId": "dataset-1"}

    class DummyDataset:
        def iterate_items(self):
            return [
                {
                    "commentId": "1",
                    "text": "@Zesco Power restored",
                    "date": "2026-01-01",
                    "threadingDepth": 0,
                },
                {
                    "commentId": "2",
                    "text": "John Banda this is a reply",
                    "date": "2026-01-02",
                    "threadingDepth": 1,
                },
            ]

    class DummyClient:
        def __init__(self, api_key):
            captured["api_key"] = api_key

        def actor(self, actor_id):
            return DummyActor()

        def dataset(self, dataset_id):
            return DummyDataset()

    monkeypatch.setattr(
        apify_client,
        "_load_config",
        lambda: apify_client.ApifyConfig(api_key="test-key"),
    )
    monkeypatch.setattr(apify_client, "ApifyClient", DummyClient)

    rows = apify_client.fetch_comments(
        "https://www.facebook.com/page/posts/123/",
        max_comments=5000,
    )

    assert len(rows) == 1
    assert captured["run_input"]["resultsLimit"] == 100
    assert captured["run_input"]["includeNestedComments"] is False
    assert rows[0]["text"] == "Power restored"
    assert rows[0]["timestamp"] == "2026-01-01"


def test_apify_client_rejects_parent_linked_reply_and_strips_structured_mention():
    source_url = "https://www.facebook.com/page/posts/123/"
    assert apify_client._normalise_item(
        {
            "commentId": "reply-1",
            "text": "A reply",
            "parentCommentId": "parent-1",
        },
        source_url,
    ) is None

    row = apify_client._normalise_item(
        {
            "commentId": "comment-1",
            "text": "Zesco Limited please restore power",
            "date": "2026-01-01T10:00:00Z",
            "threadingDepth": 0,
            "mentions": [{"name": "Zesco Limited"}],
        },
        source_url,
    )
    assert row is not None
    assert row["text"] == "please restore power"
    assert row["timestamp"] == "2026-01-01T10:00:00Z"
