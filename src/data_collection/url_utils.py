"""Canonical Facebook URL helpers shared by collection and cache lookup."""

from __future__ import annotations

import re
from urllib.parse import urlsplit, urlunsplit


FACEBOOK_CONTENT_ID = re.compile(r"/(?:posts|videos)/(\d+)", re.IGNORECASE)
FACEBOOK_HOSTS = {
    "facebook.com",
    "www.facebook.com",
    "m.facebook.com",
    "mbasic.facebook.com",
}


def facebook_content_id(url: str) -> str | None:
    """Return the stable post/video ID when it is present in a Facebook URL."""
    match = FACEBOOK_CONTENT_ID.search(str(url))
    return match.group(1) if match else None


def canonical_facebook_url(url: str) -> str:
    """Normalize Facebook hosts and remove tracking query parameters."""
    parsed = urlsplit(str(url).strip())
    host = parsed.netloc.lower()
    if host in FACEBOOK_HOSTS:
        host = "www.facebook.com"
    path = parsed.path.rstrip("/")
    if path:
        path += "/"
    return urlunsplit((parsed.scheme or "https", host, path, "", ""))
