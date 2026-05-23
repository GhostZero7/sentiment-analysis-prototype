"""Download Facebook comments through Apify and save raw CSV."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Ensure project root is importable when running: python scripts/download_data.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_collection.apify_client import ApifyFetchError, fetch_comments

MENTION_PATTERN = re.compile(r"@\w+", re.IGNORECASE)


def _strip_mentions(text: str) -> str:
    if text is None:
        return ""
    cleaned = MENTION_PATTERN.sub(" ", str(text))
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Fetch Facebook comments from Apify and write CSV.")
    parser.add_argument("--url", required=True, help="Facebook post/comment URL")
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of items/comments to fetch (default: 100)",
    )
    parser.add_argument(
        "--output",
        default="data/raw/comments.csv",
        help="Output CSV path (default: data/raw/comments.csv)",
    )
    args = parser.parse_args()

    try:
        comments = fetch_comments(args.url, max_comments=args.limit)
    except (ValueError, ApifyFetchError) as exc:
        print(f"Error: {exc}")
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(comments)
    if "text" in df.columns:
        df["text"] = df["text"].map(_strip_mentions)
    if "author_name" in df.columns:
        df = df.drop(columns=["author_name"])
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} comments to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
