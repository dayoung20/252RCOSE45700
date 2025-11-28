from __future__ import annotations

import argparse
import hashlib
import textwrap
from datetime import datetime
from html import unescape
from pathlib import Path
from typing import Any, Dict, List

import feedparser
import requests
import yaml


def load_config(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Feed config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    feeds = data.get("feeds")
    if not feeds:
        raise ValueError("No feeds configured. Add entries under 'feeds:' in feeds.yaml.")
    return feeds


def slugify(text: str) -> str:
    safe = "".join(ch if ch.isalnum() else "-" for ch in text.lower())
    while "--" in safe:
        safe = safe.replace("--", "-")
    slug = safe.strip("-") or hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    if len(slug) > 80:
        slug = slug[:80]
    return slug


def sanitize_html(raw: str | None) -> str:
    if not raw:
        return ""
    stripped = raw.replace("<br/>", "\n").replace("<br>", "\n")
    return unescape(stripped)


def format_entry(feed_name: str, entry: Dict[str, Any]) -> str:
    title = entry.get("title", "Untitled")
    published = entry.get("published", entry.get("updated", ""))
    link = entry.get("link", "")
    summary = sanitize_html(entry.get("summary"))

    lines = [
        f"# {title}",
        "",
        f"- Feed: {feed_name}",
        f"- Published: {published or 'N/A'}",
        f"- Link: {link or 'N/A'}",
        "",
        "## 내용 요약",
        "",
        textwrap.dedent(summary).strip() or "요약 제공되지 않았습니다.",
    ]
    return "\n".join(lines).strip() + "\n"


def write_entry(output_dir: Path, feed_name: str, entry: Dict[str, Any]) -> Path:
    slug = slugify(entry.get("id") or entry.get("title") or datetime.utcnow().isoformat())
    filename = output_dir / f"{slug}.md"
    if filename.exists():
        return filename
    content = format_entry(feed_name, entry)
    filename.write_text(content, encoding="utf-8")
    return filename


def fetch_feed(feed: Dict[str, Any], output_dir: Path) -> List[Path]:
    response = requests.get(feed["url"], timeout=30)
    response.raise_for_status()
    parsed = feedparser.parse(response.content)
    if parsed.bozo and not getattr(parsed, "entries", None):
        raise RuntimeError(f"Failed to parse feed: {feed['url']} ({parsed.bozo_exception})")

    limit = int(feed.get("limit", 5))
    saved_files = []
    for entry in parsed.entries[:limit]:
        saved = write_entry(output_dir, feed["name"], entry)
        saved_files.append(saved)
    return saved_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch RSS feeds and store as markdown.")
    parser.add_argument(
        "--config",
        default="feeds.yaml",
        type=str,
        help="Path to YAML file listing feeds.",
    )
    parser.add_argument(
        "--output",
        default="data/feeds",
        type=str,
        help="Directory where markdown files will be written.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    feeds = load_config(config_path)
    total = 0
    for feed in feeds:
        files = fetch_feed(feed, output_dir)
        print(f"[OK] {feed['name']}: {len(files)} items saved")
        total += len(files)

    print(f"Done. Generated {total} markdown files in {output_dir}")


if __name__ == "__main__":
    main()

