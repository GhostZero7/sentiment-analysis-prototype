"""Collect a deduplicated batch of Facebook URLs with a strict Apify limit."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_collection.apify_client import (
    MAX_COMMENTS_PER_URL,
    ApifyFetchError,
    fetch_comments,
)
from src.data_collection.url_utils import canonical_facebook_url, facebook_content_id


MAX_BATCH_LIMIT = MAX_COMMENTS_PER_URL
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_MANIFEST = DEFAULT_OUTPUT_DIR / "collection_manifest.csv"


def content_id(url: str) -> str:
    resolved = facebook_content_id(url)
    if not resolved:
        raise ValueError(f"Could not identify a Facebook post/video ID: {url}")
    return resolved


def canonical_url(url: str) -> str:
    return canonical_facebook_url(url)


def existing_content_ids(output_dir: Path) -> set[str]:
    return {
        path.stem.removeprefix("comments_post_")
        for path in output_dir.glob("comments_post_*.csv")
    }


def _save_manifest(rows: list[dict[str, object]], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    current = pd.DataFrame(rows)
    if manifest_path.is_file():
        previous = pd.read_csv(manifest_path)
        current = pd.concat([previous, current], ignore_index=True)
    if not current.empty:
        current = current.drop_duplicates(
            subset=["content_id", "source_url", "status", "collected_at"],
            keep="last",
        )
    current.to_csv(manifest_path, index=False)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def collect_batch(
    urls: list[str],
    *,
    limit: int,
    output_dir: Path,
    manifest_path: Path,
) -> pd.DataFrame:
    if limit < 1 or limit > MAX_BATCH_LIMIT:
        raise ValueError(f"limit must be between 1 and {MAX_BATCH_LIMIT}")

    output_dir.mkdir(parents=True, exist_ok=True)
    known_ids = existing_content_ids(output_dir)
    batch_ids: set[str] = set()
    manifest_rows: list[dict[str, object]] = []

    for raw_url in urls:
        source_url = canonical_url(raw_url)
        post_id = content_id(source_url)
        output_path = output_dir / f"comments_post_{post_id}.csv"
        base_record = {
            "content_id": post_id,
            "source_url": source_url,
            "requested_limit": limit,
            "output_path": _display_path(output_path),
            "collected_at": datetime.now(timezone.utc).isoformat(),
        }

        if post_id in known_ids:
            manifest_rows.append(
                {**base_record, "status": "skipped_existing", "comment_count": 0, "error": ""}
            )
            print(f"SKIP existing {post_id}: {source_url}")
            continue
        if post_id in batch_ids:
            manifest_rows.append(
                {**base_record, "status": "skipped_batch_duplicate", "comment_count": 0, "error": ""}
            )
            print(f"SKIP batch duplicate {post_id}: {source_url}")
            continue

        batch_ids.add(post_id)
        try:
            comments = fetch_comments(source_url, max_comments=limit)
            frame = pd.DataFrame(comments).head(limit)
            if not frame.empty:
                dedupe_columns = [
                    column for column in ("comment_id", "text") if column in frame.columns
                ]
                if dedupe_columns:
                    frame = frame.drop_duplicates(subset=dedupe_columns, keep="first")
            frame.to_csv(output_path, index=False)
            known_ids.add(post_id)
            manifest_rows.append(
                {
                    **base_record,
                    "status": "collected",
                    "comment_count": int(len(frame)),
                    "error": "",
                }
            )
            print(f"COLLECTED {post_id}: {len(frame)} comments")
        except (ValueError, ApifyFetchError) as exc:
            manifest_rows.append(
                {
                    **base_record,
                    "status": "failed",
                    "comment_count": 0,
                    "error": str(exc),
                }
            )
            print(f"FAILED {post_id}: {exc}")

    _save_manifest(manifest_rows, manifest_path)
    return pd.DataFrame(manifest_rows)


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urls-file", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=MAX_BATCH_LIMIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    if not args.urls_file.is_file():
        print(f"URL file not found: {args.urls_file}")
        return 1
    urls = [
        line.strip()
        for line in args.urls_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not urls:
        print("No URLs were found in the input file.")
        return 1

    try:
        results = collect_batch(
            urls,
            limit=args.limit,
            output_dir=args.output_dir,
            manifest_path=args.manifest,
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    failed = int(results["status"].eq("failed").sum()) if not results.empty else 0
    collected = int(results["status"].eq("collected").sum()) if not results.empty else 0
    skipped = int(results["status"].str.startswith("skipped").sum()) if not results.empty else 0
    print(f"Batch complete: {collected} collected, {skipped} skipped, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
