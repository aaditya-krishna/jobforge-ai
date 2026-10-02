"""Greenhouse job board API: https://developers.greenhouse.io/job-board.html"""

from datetime import datetime

import requests

from jobforge.ingest.common import JobPosting

API_URL = "https://boards-api.greenhouse.io/v1/boards/{board}/jobs"


def _iso_date(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).date().isoformat()
    except ValueError:
        return None


def parse_greenhouse(payload: dict, company: str) -> list[JobPosting]:
    """Normalize a Greenhouse /jobs?content=true response. Content stays HTML-escaped; cleaning is Phase 2."""
    jobs: list[JobPosting] = []
    for j in payload.get("jobs", []):
        jobs.append({
            "source": "greenhouse",
            "external_id": str(j["id"]),
            "title": (j.get("title") or "").strip(),
            "company": company,
            "location": (j.get("location") or {}).get("name"),
            "description_html": j.get("content") or "",
            "url": j.get("absolute_url"),
            # first_published is when the job went live; updated_at changes on every edit
            "posted_at": _iso_date(j.get("first_published") or j.get("updated_at")),
        })
    return jobs


def fetch_greenhouse(board: str, company: str, session: requests.Session) -> list[JobPosting]:
    resp = session.get(API_URL.format(board=board), params={"content": "true"}, timeout=30)
    resp.raise_for_status()
    return parse_greenhouse(resp.json(), company)
