from __future__ import annotations

from datetime import timedelta
from enum import Enum
from types import SimpleNamespace

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
            for index in range(350)
        ]

    monkeypatch.setattr(download_batch, "fetch_comments", fake_fetch)
    manifest = tmp_path / "manifest.csv"
    results = download_batch.collect_batch(
        [
            "https://www.facebook.com/page/posts/111/?app=fbl",
            "https://www.facebook.com/page/posts/222/?app=fbl",
        ],
        limit=300,
        output_dir=tmp_path,
        manifest_path=manifest,
    )

    assert results["status"].tolist() == ["skipped_existing", "collected"]
    assert calls == [("https://www.facebook.com/page/posts/222/", 300)]
    assert len(pd.read_csv(tmp_path / "comments_post_222.csv")) == 300


def test_batch_rejects_limit_above_1000(tmp_path):
    with pytest.raises(ValueError, match="between 1 and 1000"):
        download_batch.collect_batch(
            ["https://www.facebook.com/page/posts/222/"],
            limit=1001,
            output_dir=tmp_path,
            manifest_path=tmp_path / "manifest.csv",
        )


def test_shared_apify_client_caps_every_request_at_1000(monkeypatch):
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
                    "postTitle": "Power restoration update",
                    "postCreatedAt": "2025-12-31T18:00:00Z",
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
    assert captured["run_input"]["resultsLimit"] == 1000
    assert captured["run_input"]["includeNestedComments"] is False
    assert rows[0]["text"] == "Power restored"
    assert rows[0]["timestamp"] == "2026-01-01"
    assert rows[0]["post_title"] == "Power restoration update"
    assert rows[0]["post_date"] == "2025-12-31T18:00:00Z"
    assert rows[0]["collected_at"]


def test_apify_client_supports_legacy_actor_call_without_timeout(monkeypatch):
    captured: dict[str, object] = {}

    class LegacyActor:
        def call(self, *, run_input):
            captured["run_input"] = run_input
            return {"defaultDatasetId": "legacy-dataset"}

    class DummyDataset:
        def iterate_items(self):
            return [
                {
                    "commentId": "legacy-1",
                    "text": "Power supply is stable today",
                    "date": "2026-01-01",
                    "threadingDepth": 0,
                }
            ]

    class LegacyClient:
        def __init__(self, api_key):
            captured["api_key"] = api_key

        def actor(self, actor_id):
            captured["actor_id"] = actor_id
            return LegacyActor()

        def dataset(self, dataset_id):
            captured["dataset_id"] = dataset_id
            return DummyDataset()

    monkeypatch.setattr(
        apify_client,
        "_load_config",
        lambda: apify_client.ApifyConfig(api_key="legacy-key"),
    )
    monkeypatch.setattr(apify_client, "ApifyClient", LegacyClient)

    rows = apify_client.fetch_comments(
        "https://www.facebook.com/page/posts/123/",
        max_comments=300,
    )

    assert len(rows) == 1
    assert captured["run_input"]["resultsLimit"] == 300
    assert captured["dataset_id"] == "legacy-dataset"


def test_apify_client_reads_v3_run_objects_and_preserves_timeout(monkeypatch):
    calls = []

    class Status(Enum):
        SUCCEEDED = "SUCCEEDED"

    class ActorV3:
        def call(self, *, run_input, run_timeout):
            calls.append((run_input, run_timeout))
            return SimpleNamespace(default_dataset_id="typed-dataset", status=Status.SUCCEEDED)

    def dataset(dataset_id):
        assert dataset_id == "typed-dataset"
        return SimpleNamespace(iterate_items=lambda: [{"commentId": "1", "text": "Power is back"}])

    monkeypatch.setattr(apify_client, "_load_config", lambda: apify_client.ApifyConfig(api_key="test"))
    monkeypatch.setattr(apify_client, "ApifyClient", lambda key: SimpleNamespace(actor=lambda name: ActorV3(), dataset=dataset))

    rows = apify_client.fetch_comments("https://www.facebook.com/page/posts/123/", max_comments=10)

    assert [row["text"] for row in rows] == ["Power is back"]
    assert len(calls) == 1
    assert calls[0][0]["resultsLimit"] == 10
    assert calls[0][1] == timedelta(seconds=120)


@pytest.mark.parametrize("status", ["FAILED", "TIMED-OUT", "ABORTED"])
@pytest.mark.parametrize("typed_run", [False, True])
def test_unsuccessful_runs_are_not_read_as_success(monkeypatch, status, typed_run):
    if typed_run:
        run = SimpleNamespace(default_dataset_id="partial-dataset", status=status)
    else:
        run = {"defaultDatasetId": "partial-dataset", "status": status}
    actor = SimpleNamespace(call=lambda run_input: run)
    client = SimpleNamespace(actor=lambda name: actor)
    monkeypatch.setattr(apify_client, "_load_config", lambda: apify_client.ApifyConfig(api_key="test"))
    monkeypatch.setattr(apify_client, "ApifyClient", lambda key: client)

    with pytest.raises(apify_client.ApifyFetchError, match=status) as failure:
        apify_client.fetch_comments("https://www.facebook.com/page/posts/123/")

    assert failure.value.reason == ("timeout" if status == "TIMED-OUT" else "collection")


@pytest.mark.parametrize(
    ("status_code", "error_type", "reason"),
    [
        (402, "not-enough-usage-to-run-paid-actor", "billing"),
        (401, "invalid-token", "authentication"),
        (403, "insufficient-permissions", "authentication"),
        (429, "rate-limit-exceeded", "rate_limit"),
        (402, "actor-memory-limit-exceeded", "resource_limit"),
        (403, "concurrent-runs-limit-exceeded", "resource_limit"),
        (500, "internal-server-error", "collection"),
    ],
)
def test_fetch_errors_preserve_actual_failure_category(monkeypatch, status_code, error_type, reason):
    error = RuntimeError("private diagnostic text")
    error.status_code = status_code
    error.type = error_type

    def call(*, run_input):
        raise error

    client = SimpleNamespace(actor=lambda name: SimpleNamespace(call=call))
    monkeypatch.setattr(apify_client, "_load_config", lambda: apify_client.ApifyConfig(api_key="test"))
    monkeypatch.setattr(apify_client, "ApifyClient", lambda key: client)

    with pytest.raises(apify_client.ApifyFetchError) as failure:
        apify_client.fetch_comments("https://www.facebook.com/page/posts/123/")

    assert failure.value.reason == reason
    assert failure.value.__cause__ is error
    assert "private diagnostic text" not in failure.value.user_message


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


def test_apify_client_does_not_invent_missing_comment_time_and_keeps_title():
    row = apify_client._normalise_item(
        {
            "commentId": "comment-2",
            "text": "Public reaction",
            "postTitle": "New solar project announced",
            "postPublishedAt": "2025-12-30T09:00:00Z",
            "threadingDepth": 0,
        },
        "https://www.facebook.com/page/posts/123/",
    )

    assert row is not None
    assert row["timestamp"] == ""
    assert row["post_title"] == "New solar project announced"
    assert row["post_date"] == "2025-12-30T09:00:00Z"
    assert row["collected_at"]
