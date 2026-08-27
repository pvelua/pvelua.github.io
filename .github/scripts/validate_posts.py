#!/usr/bin/env python3
"""Validate automated news digests against _automation/snippet-spec.md.

Runs on every PR. Exits non-zero with a readable report if anything is wrong,
so a malformed digest can never reach master.
"""

import datetime as dt
import pathlib
import re
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
POSTS = ROOT / "_posts"
COVERED = ROOT / "_data" / "covered.yml"

VALID_CATEGORIES = {"ai", "breakthroughs"}
FILENAME_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})-([a-z0-9-]+)\.md$")
SNIPPET_RE = re.compile(r"^### \[(?P<text>[^\]]+)\]\((?P<url>https?://[^)\s]+)\)\s*$", re.M)
ANY_URL_RE = re.compile(r"\((?P<url>https?://[^)\s]+)\)")

errors: list[str] = []
warnings: list[str] = []


def fail(path: pathlib.Path, msg: str) -> None:
    errors.append(f"{path.relative_to(ROOT)}: {msg}")


def split_front_matter(raw: str):
    if not raw.startswith("---"):
        return None, raw
    parts = raw.split("---", 2)
    if len(parts) < 3:
        return None, raw
    try:
        return yaml.safe_load(parts[1]) or {}, parts[2]
    except yaml.YAMLError as exc:
        return {"__yaml_error__": str(exc)}, parts[2]


def load_covered() -> dict[str, dict]:
    if not COVERED.exists():
        errors.append("_data/covered.yml is missing")
        return {}
    data = yaml.safe_load(COVERED.read_text()) or {}
    entries = data.get("entries") or []
    if not isinstance(entries, list):
        errors.append("_data/covered.yml: 'entries' must be a list")
        return {}
    out = {}
    for entry in entries:
        if not isinstance(entry, dict) or "url" not in entry:
            errors.append(f"_data/covered.yml: bad entry {entry!r}")
            continue
        out[entry["url"].rstrip("/")] = entry
    return out


def main() -> int:
    covered = load_covered()
    seen_urls: dict[str, str] = {}

    if not POSTS.exists():
        print("No _posts directory yet - nothing to validate.")
        return 0

    for path in sorted(POSTS.glob("*.md")):
        match = FILENAME_RE.match(path.name)
        if not match:
            fail(path, "filename must be YYYY-MM-DD-slug.md (lowercase, hyphens)")
            continue

        year, month, day, _slug = match.groups()
        raw = path.read_text(encoding="utf-8")
        front, body = split_front_matter(raw)

        if front is None:
            fail(path, "missing YAML front matter")
            continue
        if "__yaml_error__" in front:
            fail(path, f"front matter is not valid YAML: {front['__yaml_error__']}")
            continue

        if "layout" in front:
            warnings.append(f"{path.name}: 'layout' is set by _config.yml; remove it")

        for key in ("title", "date", "categories", "summary", "item_count"):
            if key not in front:
                fail(path, f"front matter is missing required key '{key}'")

        cats = front.get("categories")
        if not isinstance(cats, list) or len(cats) != 1:
            fail(path, "'categories' must be a one-element list, e.g. [ai]")
        elif cats[0] not in VALID_CATEGORIES:
            fail(path, f"unknown category {cats[0]!r}; expected one of {sorted(VALID_CATEGORIES)}")

        date = front.get("date")
        if isinstance(date, dt.datetime):
            date = date.date()
        if isinstance(date, dt.date):
            if (date.year, date.month, date.day) != (int(year), int(month), int(day)):
                fail(path, f"front matter date {date} does not match filename date {year}-{month}-{day}")
        elif date is not None:
            fail(path, "'date' must be an unquoted YYYY-MM-DD value")

        summary = front.get("summary")
        if isinstance(summary, str) and len(summary.split()) > 30:
            warnings.append(f"{path.name}: summary is long ({len(summary.split())} words); aim under 25")

        snippets = SNIPPET_RE.findall(body)
        snippet_urls = [u.rstrip("/") for _t, u in snippets]

        declared = front.get("item_count")
        if isinstance(declared, int) and declared != len(snippets):
            fail(path, f"item_count is {declared} but {len(snippets)} snippets were found")

        if not snippets:
            fail(path, "no snippets found; each must start '### [text](https://url)'")

        all_urls = [u.rstrip("/") for u in ANY_URL_RE.findall(body)]
        extra = [u for u in all_urls if u not in snippet_urls]
        if extra:
            fail(path, f"snippets may only contain the heading link; found extra: {extra[:3]}")

        for url in snippet_urls:
            if snippet_urls.count(url) > 1:
                fail(path, f"duplicate URL within the digest: {url}")
            if url not in covered:
                fail(path, f"{url} is not in _data/covered.yml - the ledger was not updated")
            if url in seen_urls and seen_urls[url] != path.name:
                fail(path, f"{url} was already used in {seen_urls[url]}")
            seen_urls[url] = path.name

    for warning in warnings:
        print(f"warning: {warning}")

    if errors:
        print("\nValidation failed:\n")
        for err in errors:
            print(f"  - {err}")
        return 1

    print(f"OK - {len(seen_urls)} unique source URLs across all digests.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
