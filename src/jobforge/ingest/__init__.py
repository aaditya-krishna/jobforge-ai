"""Job source fetchers. Each takes (board, company, session) and returns normalized JobPosting dicts."""

from jobforge.ingest.greenhouse import fetch_greenhouse
from jobforge.ingest.lever import fetch_lever

FETCHERS = {
    "greenhouse": fetch_greenhouse,
    "lever": fetch_lever,
}
