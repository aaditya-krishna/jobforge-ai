from pathlib import Path

import psycopg
import pytest
from streamlit.testing.v1 import AppTest

from jobforge.config import get_settings

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
RESUME = (Path(__file__).resolve().parents[1] / "data" / "eval" / "resumes" / "backend_engineer.md").read_text("utf-8")


def db_has_jobs() -> bool:
    try:
        with psycopg.connect(get_settings().database_url, connect_timeout=3) as conn:
            return conn.execute("SELECT EXISTS (SELECT 1 FROM job_embeddings)").fetchone()[0]
    except Exception:
        return False


def test_app_starts_without_errors():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert at.button[0].label == "Find matches"


def test_short_resume_shows_warning():
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.text_area(key="pasted").input("too short").run()
    at.button[0].click().run()
    assert any("too short" in w.value for w in at.warning)


@pytest.mark.skipif(not db_has_jobs(), reason="needs the database with indexed jobs")
def test_pasted_resume_returns_matches():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.text_area(key="pasted").input(RESUME).run()
    at.button[0].click().run()
    assert not at.exception and not at.error
    assert any("Skills found in your resume" in s.value for s in at.subheader)
    assert len(at.metric) >= 5  # one score per result
