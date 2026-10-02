"""Offline evaluation: candidate pools, relevance labels and precision@k for ranking configs.

Every config re-ranks the same retrieved pool (top-k by semantic similarity), and every job in
the pool is labeled, so no ranked result is ever unlabeled.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import psycopg

from jobforge.config import DATA_DIR
from jobforge.embed import embed_long
from jobforge.match import load_job_skills, rank, retrieve

EVAL_DIR = DATA_DIR / "eval"
RESUME_DIR = EVAL_DIR / "resumes"
LABELS_CSV = EVAL_DIR / "labels.csv"
POOL_K = 50


@dataclass(frozen=True)
class Config:
    name: str
    w_sem: float
    prior: float = 0.0
    keybert: bool = False


CONFIGS = [
    Config("semantic only", 1.0),
    Config("coverage only", 0.0),
    Config("hybrid w=0.5", 0.5),
    Config("hybrid w=0.6", 0.6),
    Config("hybrid w=0.7", 0.7),
    Config("hybrid w=0.8", 0.8),
    Config("hybrid w=0.7, smoothed coverage (prior 3)", 0.7, prior=3.0),
    Config("hybrid w=0.7, smoothed coverage (prior 5)", 0.7, prior=5.0),
    Config("hybrid w=0.6, smoothed coverage (prior 5)", 0.6, prior=5.0),
    Config("hybrid w=0.7, + KeyBERT skills", 0.7, keybert=True),
]


def label_key(company: str, title: str) -> tuple[str, str]:
    """Labels are per role, not per posting: copies of a role in other cities share one label."""
    return (company or "").strip().lower(), " ".join((title or "").lower().split())


def precision_at_k(relevance: list[bool], k: int) -> float:
    """Share of the top k results that are relevant (fewer than k results count as misses)."""
    return sum(relevance[:k]) / k


@dataclass
class Pool:
    resume: str
    resume_skills: set[str]
    candidates: list[dict]
    skills: dict[int, set[str]]  # phrase_match only
    skills_with_keybert: dict[int, set[str]]


def build_pool(conn: psycopg.Connection, name: str, text: str, extractor) -> Pool:
    candidates = retrieve(conn, embed_long(text), POOL_K)
    ids = [c["job_id"] for c in candidates]
    return Pool(name, extractor.extract(text), candidates, load_job_skills(conn, ids),
                load_job_skills(conn, ids, ("phrase_match", "keybert")))


def ranked_keys(pool: Pool, config: Config, top_n: int = 10) -> list[tuple[str, str]]:
    skills = pool.skills_with_keybert if config.keybert else pool.skills
    ranked = rank(pool.candidates, skills, pool.resume_skills, config.w_sem, top_n, config.prior)
    return [label_key(r["company"], r["title"]) for r in ranked]


def load_resumes(resume_dir: Path = RESUME_DIR) -> dict[str, str]:
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted(resume_dir.glob("*.md"))}


def load_labels(path: Path = LABELS_CSV) -> dict[tuple[str, str, str], bool]:
    """(resume, company, title) -> relevant."""
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {(r["resume"], *label_key(r["company"], r["title"])): r["relevant"] == "1"
                for r in csv.DictReader(f)}
