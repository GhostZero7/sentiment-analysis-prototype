"""Apify client wrapper for fetching Facebook post content and comments."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from apify_client import ApifyClient


FACEBOOK_URL_PATTERN = re.compile(r"^https?://(www\.)?facebook\.com/", re.IGNORECASE)
AT_MENTION_PATTERN = re.compile(r"@[\w.-]+", re.UNICODE)
DEFAULT_ACTOR_ID = "apify/facebook-comments-scraper"
MAX_COMMENTS_PER_URL = 1_000


class ApifyFetchError(RuntimeError):
    """Raised when comment collection from Apify fails."""


@dataclass(slots=True)
class ApifyConfig:
    """Runtime configuration for Apify scraping."""

    api_key: str
    actor_id: str = DEFAULT_ACTOR_ID
    timeout_secs: int = 120
    max_comments: int = MAX_COMMENTS_PER_URL
    comments_mode: str = "ALL"


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def validate_facebook_url(url: str) -> bool:
    """Return True when URL points to facebook.com."""
    if not isinstance(url, str):
        return False
    return bool(FACEBOOK_URL_PATTERN.search(url.strip()))


def _normalize_facebook_url(url: str) -> str:
    """Normalize common Facebook share URLs to canonical form where possible."""
    cleaned = url.strip()
    parsed = urlparse(cleaned)
    host = parsed.netloc.lower()
    if host in {"m.facebook.com", "mbasic.facebook.com"}:
        return cleaned.replace(host, "www.facebook.com")
    return cleaned


def _load_config() -> ApifyConfig:
    api_key = os.getenv("APIFY_API_KEY", "").strip()
    if not api_key:
        raise ApifyFetchError("APIFY_API_KEY is missing. Add it to your environment or .env file.")

    actor_id = os.getenv("APIFY_ACTOR_ID", DEFAULT_ACTOR_ID).strip() or DEFAULT_ACTOR_ID
    timeout_str = os.getenv("APIFY_TIMEOUT_SECS", "120").strip()
    timeout_secs = int(timeout_str) if timeout_str.isdigit() else 120
    max_comments_str = os.getenv("APIFY_MAX_COMMENTS", str(MAX_COMMENTS_PER_URL)).strip()
    configured_max = (
        int(max_comments_str) if max_comments_str.isdigit() else MAX_COMMENTS_PER_URL
    )
    max_comments = min(max(configured_max, 1), MAX_COMMENTS_PER_URL)
    comments_mode = os.getenv("APIFY_COMMENTS_MODE", "ALL").strip().upper() or "ALL"
    if comments_mode not in {"ALL", "NEWEST", "MOST_RELEVANT"}:
        comments_mode = "ALL"
    return ApifyConfig(
        api_key=api_key,
        actor_id=actor_id,
        timeout_secs=timeout_secs,
        max_comments=max_comments,
        comments_mode=comments_mode,
    )


def _has_value(value: Any) -> bool:
    if value is None or value is False:
        return False
    return str(value).strip().lower() not in {"", "0", "false", "none", "null"}


def _is_reply_item(item: dict[str, Any]) -> bool:
    """Detect reply rows across common Facebook scraper schemas."""
    for field in ("threadingDepth", "depth", "replyDepth"):
        value = item.get(field)
        try:
            if value is not None and int(value) > 0:
                return True
        except (TypeError, ValueError):
            if _has_value(value):
                return True

    if any(
        _has_value(item.get(field))
        for field in (
            "replyToCommentId",
            "parentCommentId",
            "parent_comment_id",
            "replyToId",
            "parentId",
            "topCommentId",
        )
    ):
        return True

    if str(item.get("type", "")).strip().lower() == "reply":
        return True
    return str(item.get("isReply", "")).strip().lower() == "true"


def _structured_mention_names(item: dict[str, Any]) -> list[str]:
    mentions = item.get("mentions") or []
    if isinstance(mentions, dict):
        mentions = [mentions]
    if not isinstance(mentions, list):
        return []

    names: list[str] = []
    for mention in mentions:
        if isinstance(mention, str):
            name = mention
        elif isinstance(mention, dict):
            name = next(
                (
                    str(mention[field])
                    for field in ("name", "text", "displayName", "profileName")
                    if _has_value(mention.get(field))
                ),
                "",
            )
        else:
            name = ""
        if name.strip():
            names.append(name.strip())
    return names


def _strip_mentions(text: str, item: dict[str, Any]) -> str:
    cleaned = AT_MENTION_PATTERN.sub(" ", text)
    for name in sorted(_structured_mention_names(item), key=len, reverse=True):
        cleaned = re.sub(
            rf"(?<!\w)@?{re.escape(name)}(?=\s|[:,.!?;-]|$)",
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )
    return re.sub(r"\s+", " ", cleaned).strip(" ,:;-\t\r\n")


def _normalise_item(item: dict[str, Any], source_url: str) -> dict[str, Any] | None:
    """Normalize a top-level actor result to the dashboard's stable schema."""
    if _is_reply_item(item):
        return None

    text = (
        item.get("text")
        or item.get("message")
        or item.get("commentText")
        or item.get("body")
        or ""
    )
    text = _strip_mentions(str(text), item)
    comment_id = (
        item.get("commentId")
        or item.get("id")
        or item.get("fbId")
        or f"generated-{hash((text, item.get('timestamp')))}"
    )
    timestamp = (
        item.get("date")
        or item.get("time")
        or item.get("timestamp")
        or item.get("publishedAt")
        or item.get("createdAt")
        or item.get("created_time")
        or ""
    )
    post_title = (
        item.get("postTitle")
        or item.get("post_title")
        or item.get("storyTitle")
        or ""
    )

    return {
        "comment_id": str(comment_id),
        "text": text,
        "timestamp": str(timestamp),
        "source_url": source_url,
        "post_title": str(post_title).strip(),
        "collected_at": _utc_now_iso(),
    }


def fetch_comments(url: str, max_comments: int | None = None) -> list[dict[str, Any]]:
    """
    Fetch post + comments from an input Facebook URL using Apify.

    Returns a list of dictionaries with:
    - comment_id
    - text
    - timestamp
    - source_url
    - post_title
    - collected_at
    """
    cleaned_url = _normalize_facebook_url((url or "").strip())
    if not validate_facebook_url(cleaned_url):
        raise ValueError("Invalid URL. Please provide a valid facebook.com post/comment URL.")

    config = _load_config()
    client = ApifyClient(config.api_key)
    requested_limit = (
        max_comments if isinstance(max_comments, int) and max_comments > 0 else config.max_comments
    )
    limit = min(requested_limit, MAX_COMMENTS_PER_URL)

    if "facebook-comments-scraper" in config.actor_id:
        view_option_map = {
            "ALL": "RANKED_UNFILTERED",
            "NEWEST": "RECENT_ACTIVITY",
            "MOST_RELEVANT": "RANKED_THREADED",
        }
        actor_input = {
            "startUrls": [{"url": cleaned_url}],
            "resultsLimit": limit,
            "includeNestedComments": False,
            "viewOption": view_option_map.get(config.comments_mode, "RANKED_UNFILTERED"),
        }
    else:
        actor_input = {
            "startUrls": [{"url": cleaned_url}],
            "resultsLimit": limit,
            "includeNestedComments": False,
            "viewOption": "RECENT_ACTIVITY",
        }

    def _run_actor(actor_id: str) -> list[dict[str, Any]]:
        run = client.actor(actor_id).call(run_input=actor_input, timeout_secs=config.timeout_secs)
        dataset_id = run.get("defaultDatasetId")
        if not dataset_id:
            raise ApifyFetchError("Apify run completed without a dataset.")
        return list(client.dataset(dataset_id).iterate_items())

    try:
        items = _run_actor(config.actor_id)
    except Exception as exc:
        # Backward compatibility for outdated actor ids in local .env files.
        if "Actor with this name was not found" in str(exc) and config.actor_id != DEFAULT_ACTOR_ID:
            try:
                items = _run_actor(DEFAULT_ACTOR_ID)
            except Exception as fallback_exc:  # pragma: no cover
                raise ApifyFetchError(
                    "Failed to fetch comments from Apify. "
                    f"Details: {fallback_exc}. "
                    "Check API key validity, actor id, URL visibility (public post), and network."
                ) from fallback_exc
        else:
            raise ApifyFetchError(
                "Failed to fetch comments from Apify. "
                f"Details: {exc}. "
                "Check API key validity, actor id, URL visibility (public post), and network."
            ) from exc

    normalised: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        row = _normalise_item(item, cleaned_url)
        if row is None or not row["text"]:
            continue
        dedupe_key = (row["comment_id"], row["text"])
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        normalised.append(row)
        if len(normalised) >= limit:
            break
    return normalised
