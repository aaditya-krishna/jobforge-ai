"""Clean every posting in data/raw/ and insert it into the jobs table.

Idempotent: ON CONFLICT (source, external_id) DO NOTHING, so reloading never duplicates.
Boilerplate is detected per company, so it needs all of a company's postings at once.

    python scripts/load_jobs.py
"""

import json
from collections import defaultdict
from datetime import date

from jobforge.clean import clean_description, find_boilerplate, html_to_lines
from jobforge.config import RAW_DIR
from jobforge.db import connect

MIN_DESCRIPTION_CHARS = 200

INSERT_SQL = """
    INSERT INTO jobs (source, external_id, title, company, location, description, url, posted_at)
    VALUES (%(source)s, %(external_id)s, %(title)s, %(company)s, %(location)s,
            %(description)s, %(url)s, %(posted_at)s)
    ON CONFLICT (source, external_id) DO NOTHING
"""


def load_raw() -> dict[tuple[str, str], list[dict]]:
    """Raw postings grouped by (source, company)."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for path in sorted(RAW_DIR.glob("*.json")):
        for job in json.loads(path.read_text(encoding="utf-8")):
            groups[(job["source"], job["company"])].append(job)
    return groups


def clean_group(jobs: list[dict]) -> tuple[list[dict], int, int]:
    """Clean one company's postings. Returns (rows, words_removed, too_short)."""
    all_lines = [html_to_lines(j["description_html"]) for j in jobs]
    boilerplate = find_boilerplate(all_lines)
    rows, removed, too_short = [], 0, 0
    for job, lines in zip(jobs, all_lines):
        description = clean_description(lines, boilerplate)
        removed += len(" ".join(lines).split()) - len(description.split())
        if len(description) < MIN_DESCRIPTION_CHARS or not job["title"]:
            too_short += 1
            continue
        rows.append({
            "source": job["source"],
            "external_id": job["external_id"],
            "title": job["title"],
            "company": job["company"],
            "location": job["location"],
            "description": description,
            "url": job["url"],
            "posted_at": date.fromisoformat(job["posted_at"]) if job["posted_at"] else None,
        })
    return rows, removed, too_short


def main() -> None:
    groups = load_raw()
    rows, removed, too_short, total_words = [], 0, 0, 0
    for (source, company), jobs in groups.items():
        group_rows, group_removed, group_short = clean_group(jobs)
        rows += group_rows
        removed += group_removed
        too_short += group_short
        total_words += group_removed + sum(len(r["description"].split()) for r in group_rows)

    with connect(vector=False) as conn:
        before = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        with conn.cursor() as cur:
            cur.executemany(INSERT_SQL, rows)
        after = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]

    print(f"Raw postings: {sum(len(j) for j in groups.values())} from {len(groups)} companies")
    print(f"Dropped as too short (<{MIN_DESCRIPTION_CHARS} chars): {too_short}")
    # Greenhouse intro/conclusion divs are already gone at parse time; this counts repeated paragraphs only
    print(f"Repeated paragraphs removed: {removed} words ({removed / max(total_words, 1):.0%} of remaining text)")
    print(f"Inserted: {after - before} new; jobs table now has {after} rows")


if __name__ == "__main__":
    main()
