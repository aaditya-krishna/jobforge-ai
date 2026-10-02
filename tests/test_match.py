import pytest

from jobforge.embed import chunk_words
from jobforge.match import rank, score


def test_coverage_and_weighting():
    r = score(0.5, {"Python", "SQL", "Excel"}, {"Python", "SQL", "Kafka", "AWS"}, w_sem=0.7)
    assert r["coverage"] == pytest.approx(0.5)
    assert r["score"] == pytest.approx(0.7 * 0.5 + 0.3 * 0.5)
    assert r["matched"] == ["Python", "SQL"]
    assert r["missing"] == ["AWS", "Kafka"]


def test_job_without_skills_has_zero_coverage():
    r = score(0.8, {"Python"}, set())
    assert r["coverage"] == 0.0
    assert r["score"] == pytest.approx(0.7 * 0.8)
    assert r["matched"] == [] and r["missing"] == []


def test_resume_without_skills():
    r = score(0.6, set(), {"Python", "Go"})
    assert r["coverage"] == 0.0 and r["missing"] == ["Go", "Python"]


@pytest.mark.parametrize("w_sem, expected", [(1.0, 0.4), (0.0, 1.0)])
def test_weight_extremes(w_sem, expected):
    assert score(0.4, {"Python"}, {"Python"}, w_sem=w_sem)["score"] == pytest.approx(expected)


def test_rank_reorders_by_hybrid_score():
    candidates = [
        {"job_id": 1, "similarity": 0.60},  # semantically closest, but no skill overlap
        {"job_id": 2, "similarity": 0.55},
    ]
    skills = {1: {"Kafka", "Scala"}, 2: {"Python", "SQL"}}
    ranked = rank(candidates, skills, {"Python", "SQL"}, w_sem=0.7)
    assert [r["job_id"] for r in ranked] == [2, 1]
    assert [r["job_id"] for r in rank(candidates, skills, {"Python", "SQL"}, w_sem=1.0)] == [1, 2]


def test_rank_truncates_and_breaks_ties_stably():
    candidates = [{"job_id": i, "similarity": 0.5} for i in (3, 1, 2)]
    assert [r["job_id"] for r in rank(candidates, {}, set(), top_n=2)] == [1, 2]


def test_rank_collapses_same_role_posted_in_several_locations():
    candidates = [
        {"job_id": 1, "similarity": 0.50, "company": "Reddit", "title": "Senior ML Engineer", "location": "Remote - US"},
        {"job_id": 2, "similarity": 0.55, "company": "Reddit", "title": "Senior ML Engineer", "location": "Toronto"},
        {"job_id": 3, "similarity": 0.40, "company": "Spotify", "title": "Data Engineer", "location": "NYC"},
    ]
    ranked = rank(candidates, {}, set(), w_sem=1.0)
    assert [r["job_id"] for r in ranked] == [2, 3]  # best-scoring copy kept
    assert ranked[0]["other_locations"] == ["Remote - US"]


def test_chunk_words_overlaps_and_covers_everything():
    words = [f"w{i}" for i in range(400)]
    chunks = chunk_words(" ".join(words), size=180, overlap=40)
    assert all(len(c.split()) <= 180 for c in chunks)
    assert chunks[0].split()[-40:] == chunks[1].split()[:40]
    assert set(" ".join(chunks).split()) == set(words)


def test_chunk_words_short_and_empty():
    assert chunk_words("just a few words") == ["just a few words"]
    assert chunk_words("   ") == []
