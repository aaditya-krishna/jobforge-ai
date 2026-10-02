"""Lever postings API: https://github.com/lever/postings-api"""

from datetime import datetime, timezone

import requests

from jobforge.ingest.common import JobPosting

API_URL = "https://api.lever.co/v0/postings/{board}"


def _description_html(p: dict) -> str:
    """Lever splits a posting into description, titled lists (responsibilities, requirements), and a footer.
    Stitch them back together so requirements aren't lost."""
    parts = [p.get("description") or ""]
    for lst in p.get("lists") or []:
        parts.append(f"<h3>{lst.get('text', '')}</h3><ul>{lst.get('content', '')}</ul>")
    parts.append(p.get("additional") or "")
    return "\n".join(part for part in parts if part)


def _iso_date(ms: int | None) -> str | None:
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).date().isoformat()


def parse_lever(payload: list[dict], company: str) -> list[JobPosting]:
    """Normalize a Lever /postings?mode=json response."""
    jobs: list[JobPosting] = []
    for p in payload:
        jobs.append({
            "source": "lever",
            "external_id": str(p["id"]),
            "title": (p.get("text") or "").strip(),
            "company": company,
            "location": (p.get("categories") or {}).get("location"),
            "description_html": _description_html(p),
            "url": p.get("hostedUrl"),
            "posted_at": _iso_date(p.get("createdAt")),
        })
    return jobs


def fetch_lever(board: str, company: str, session: requests.Session) -> list[JobPosting]:
    resp = session.get(API_URL.format(board=board), params={"mode": "json"}, timeout=30)
    resp.raise_for_status()
    return parse_lever(resp.json(), company)
