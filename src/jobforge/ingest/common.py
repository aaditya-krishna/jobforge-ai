"""Shared pieces for job source fetchers: the normalized record shape, HTTP session, and raw file I/O."""

import json
import os
from pathlib import Path
from typing import TypedDict

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

USER_AGENT = "jobforge-ai/0.1 (personal learning project)"


class JobPosting(TypedDict):
    """The one shape every source is normalized into before it touches disk or the database."""

    source: str
    external_id: str
    title: str
    company: str
    location: str | None
    description_html: str
    url: str | None
    posted_at: str | None  # ISO date, YYYY-MM-DD


def make_session() -> requests.Session:
    """Session that retries transient failures (429 / 5xx) with exponential backoff."""
    retry = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
    )
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def dedupe(jobs: list[JobPosting]) -> list[JobPosting]:
    """Keep one posting per (source, external_id), sorted so reruns produce identical files."""
    unique = {(j["source"], j["external_id"]): j for j in jobs}
    return [unique[k] for k in sorted(unique)]


def save_raw(jobs: list[JobPosting], path: Path) -> None:
    """Write atomically: an interrupted run never leaves a half-written file behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(jobs, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)
