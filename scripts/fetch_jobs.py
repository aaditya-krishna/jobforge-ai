"""Fetch job postings for every board in data/companies.csv into data/raw/{source}_{board}.json.

Idempotent: each board's file is rewritten from scratch, de-duplicated by external_id,
so rerunning never accumulates duplicates. A failed board keeps its previous file.

    python scripts/fetch_jobs.py                 # all boards
    python scripts/fetch_jobs.py --only stripe   # one board
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import requests

from jobforge.ingest import FETCHERS
from jobforge.ingest.common import dedupe, make_session, save_raw

ROOT = Path(__file__).resolve().parents[1]
COMPANIES_CSV = ROOT / "data" / "companies.csv"
RAW_DIR = ROOT / "data" / "raw"


def load_companies(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def count_unique_on_disk(raw_dir: Path) -> int:
    keys = set()
    for path in raw_dir.glob("*.json"):
        for job in json.loads(path.read_text(encoding="utf-8")):
            keys.add((job["source"], job["external_id"]))
    return len(keys)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", help="fetch a single board by name")
    parser.add_argument("--sleep", type=float, default=1.0, help="seconds between requests (default 1.0)")
    args = parser.parse_args()

    companies = load_companies(COMPANIES_CSV)
    if args.only:
        companies = [c for c in companies if c["board"] == args.only]
        if not companies:
            print(f"No board named {args.only!r} in {COMPANIES_CSV.name}", file=sys.stderr)
            return 1

    session = make_session()
    failures = []
    for i, c in enumerate(companies):
        fetch = FETCHERS.get(c["source"])
        if fetch is None:
            print(f"  skip  {c['board']}: unknown source {c['source']!r}", file=sys.stderr)
            failures.append(c["board"])
            continue
        try:
            jobs = dedupe(fetch(c["board"], c["company"], session))
        except (requests.RequestException, ValueError, KeyError) as e:
            print(f"  FAIL  {c['source']}/{c['board']}: {e}", file=sys.stderr)
            failures.append(c["board"])
        else:
            save_raw(jobs, RAW_DIR / f"{c['source']}_{c['board']}.json")
            print(f"  ok    {c['source']}/{c['board']}: {len(jobs)} postings")
        if i < len(companies) - 1:
            time.sleep(args.sleep)

    print(f"\n{len(companies) - len(failures)}/{len(companies)} boards fetched; "
          f"{count_unique_on_disk(RAW_DIR)} unique postings in {RAW_DIR.relative_to(ROOT)}")
    if failures:
        print(f"Failed: {', '.join(failures)}", file=sys.stderr)
    return 1 if len(failures) == len(companies) else 0


if __name__ == "__main__":
    sys.exit(main())
