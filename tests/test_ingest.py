from jobforge.ingest.common import dedupe
from jobforge.ingest.greenhouse import parse_greenhouse
from jobforge.ingest.lever import parse_lever

GREENHOUSE_PAYLOAD = {
    "jobs": [
        {
            "id": 4012345,
            "title": "  Machine Learning Engineer ",
            "location": {"name": "Boston, MA"},
            "content": "&lt;p&gt;Build models with PyTorch&lt;/p&gt;",
            "absolute_url": "https://boards.greenhouse.io/examplecorp/jobs/4012345",
            "first_published": "2026-09-01T10:00:00-04:00",
            "updated_at": "2026-09-20T10:00:00-04:00",
        },
        {"id": 7, "title": "Recruiter", "location": None, "content": None},
    ]
}

LEVER_PAYLOAD = [
    {
        "id": "abc-123",
        "text": "Data Engineer",
        "categories": {"location": "London", "team": "Data"},
        "description": "<p>About the role</p>",
        "lists": [{"text": "Requirements", "content": "<li>SQL</li><li>Airflow</li>"}],
        "additional": "<p>Equal opportunity employer</p>",
        "hostedUrl": "https://jobs.lever.co/examplecorp/abc-123",
        "createdAt": 1756684800000,  # 2025-09-01T00:00:00Z
    }
]


def test_greenhouse_normalizes_fields():
    job = parse_greenhouse(GREENHOUSE_PAYLOAD, "ExampleCorp")[0]
    assert job == {
        "source": "greenhouse",
        "external_id": "4012345",
        "title": "Machine Learning Engineer",
        "company": "ExampleCorp",
        "location": "Boston, MA",
        "description_html": "&lt;p&gt;Build models with PyTorch&lt;/p&gt;",  # unescaping is Phase 2
        "url": "https://boards.greenhouse.io/examplecorp/jobs/4012345",
        "posted_at": "2026-09-01",  # first_published wins over updated_at
    }


def test_greenhouse_tolerates_missing_fields():
    job = parse_greenhouse(GREENHOUSE_PAYLOAD, "ExampleCorp")[1]
    assert job["location"] is None
    assert job["description_html"] == ""
    assert job["url"] is None
    assert job["posted_at"] is None


def test_greenhouse_empty_payload():
    assert parse_greenhouse({}, "ExampleCorp") == []


def test_lever_keeps_requirements_lists():
    job = parse_lever(LEVER_PAYLOAD, "ExampleCorp")[0]
    html = job["description_html"]
    assert "About the role" in html
    assert "<h3>Requirements</h3>" in html and "<li>Airflow</li>" in html
    assert "Equal opportunity employer" in html


def test_lever_normalizes_fields():
    job = parse_lever(LEVER_PAYLOAD, "ExampleCorp")[0]
    assert job["source"] == "lever"
    assert job["external_id"] == "abc-123"
    assert job["location"] == "London"
    assert job["posted_at"] == "2025-09-01"


def test_dedupe_keeps_one_per_id_in_stable_order():
    a = parse_greenhouse(GREENHOUSE_PAYLOAD, "ExampleCorp")
    result = dedupe(a + list(reversed(a)))
    assert [j["external_id"] for j in result] == ["4012345", "7"]
    assert dedupe(list(reversed(a))) == result
