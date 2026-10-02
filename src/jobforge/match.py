"""Retrieval (pgvector) and re-ranking (semantic similarity + skill coverage)."""

import numpy as np
import psycopg

# HNSW returns at most ef_search rows (default 40), so it must be at least the candidate count
EF_SEARCH = 200
DEFAULT_W_SEM = 0.7

RETRIEVE_SQL = """
    SELECT j.id, j.title, j.company, j.location, j.url, 1 - (e.embedding <=> %(q)s) AS similarity
    FROM job_embeddings e
    JOIN jobs j ON j.id = e.job_id
    WHERE %(location)s::text IS NULL OR j.location ILIKE '%%' || %(location)s || '%%'
    ORDER BY e.embedding <=> %(q)s
    LIMIT %(k)s
"""


def retrieve(conn: psycopg.Connection, query_vec: np.ndarray, k: int = 50,
             location: str | None = None) -> list[dict]:
    """Top-k jobs by cosine similarity, optionally only where location contains `location`.

    `<=>` is cosine distance, so similarity = 1 - distance. With a filter, iterative index scans
    (pgvector 0.8) keep walking the HNSW graph until k rows pass it, instead of returning fewer.
    """
    conn.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(max(EF_SEARCH, k)),))
    conn.execute("SELECT set_config('hnsw.iterative_scan', 'relaxed_order', true)")
    params = {"q": query_vec, "k": k, "location": (location or "").strip() or None}
    rows = conn.execute(RETRIEVE_SQL, params).fetchall()
    cols = ("job_id", "title", "company", "location", "url", "similarity")
    return [dict(zip(cols, r)) for r in rows]


def load_descriptions(conn: psycopg.Connection, job_ids: list[int]) -> dict[int, str]:
    rows = conn.execute("SELECT id, description FROM jobs WHERE id = ANY(%s)", (job_ids,)).fetchall()
    return dict(rows)


def load_job_skills(conn: psycopg.Connection, job_ids: list[int],
                    methods: tuple[str, ...] = ("phrase_match",)) -> dict[int, set[str]]:
    """Skills per job. Coverage uses phrase_match only by default; KeyBERT rows are opt-in."""
    skills: dict[int, set[str]] = {job_id: set() for job_id in job_ids}
    rows = conn.execute(
        "SELECT job_id, skill FROM job_skills WHERE job_id = ANY(%s) AND method = ANY(%s)",
        (job_ids, list(methods)),
    ).fetchall()
    for job_id, skill in rows:
        skills[job_id].add(skill)
    return skills


# --- Pure scoring (no database) -----------------------------------------------------------

def score(semantic: float, resume_skills: set[str], job_skills: set[str], w_sem: float = DEFAULT_W_SEM) -> dict:
    """Hybrid score: w_sem * semantic similarity + (1 - w_sem) * skill coverage.

    Coverage = share of the job's skills the resume has. A job with no extracted skills gets
    coverage 0, so it competes on semantic similarity alone.
    """
    matched = resume_skills & job_skills
    coverage = len(matched) / len(job_skills) if job_skills else 0.0
    return {
        "score": w_sem * semantic + (1 - w_sem) * coverage,
        "semantic": semantic,
        "coverage": coverage,
        "matched": sorted(matched),
        "missing": sorted(job_skills - resume_skills),
    }


def rank(candidates: list[dict], job_skills: dict[int, set[str]], resume_skills: set[str],
         w_sem: float = DEFAULT_W_SEM, top_n: int = 10) -> list[dict]:
    """Re-rank retrieved candidates by the hybrid score. Ties break on job_id for stable output.

    The same role is often posted once per location; those collapse into the best-scoring copy
    so duplicates don't crowd out other jobs. Other copies' locations are kept in `other_locations`.
    """
    scored = [{**c, **score(c["similarity"], resume_skills, job_skills.get(c["job_id"], set()), w_sem)}
              for c in candidates]
    scored.sort(key=lambda r: (-r["score"], r["job_id"]))
    unique: dict[tuple, dict] = {}
    for r in scored:
        key = ((r.get("company") or "").lower(), (r.get("title") or "").lower())
        if key[1] and key in unique:
            if r.get("location"):
                unique[key]["other_locations"].append(r["location"])
            continue
        unique[key if key[1] else ("", str(r["job_id"]))] = {**r, "other_locations": []}
    return list(unique.values())[:top_n]


def match_resume(conn: psycopg.Connection, resume_text: str, extractor, k: int = 50, top_n: int = 10,
                 w_sem: float = DEFAULT_W_SEM, methods: tuple[str, ...] = ("phrase_match",),
                 location: str | None = None) -> tuple[set[str], list[dict]]:
    """Full online flow: embed + extract skills from the resume, retrieve k candidates, re-rank.

    Returns (resume_skills, ranked results).
    """
    from jobforge.embed import embed_long  # imported here so pure scoring tests don't load the model

    resume_skills = extractor.extract(resume_text)
    candidates = retrieve(conn, embed_long(resume_text), k, location)
    skills = load_job_skills(conn, [c["job_id"] for c in candidates], methods)
    return resume_skills, rank(candidates, skills, resume_skills, w_sem, top_n)
