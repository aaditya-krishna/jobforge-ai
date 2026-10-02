import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from jobforge.config import Settings
from jobforge.explain import explain_matches, make_llm, template_explanation, ungrounded_skills
from jobforge.skills import SkillExtractor, load_vocab

MATCH = {
    "job_id": 7, "title": "Backend Engineer", "company": "ExampleCorp",
    "matched": ["Go", "Kubernetes", "PostgreSQL"], "missing": ["Kafka", "Terraform"],
}
GROUNDED = ("Your Go and Kubernetes experience fits this backend role well, and you already run PostgreSQL. "
            "The main gaps are Kafka and Terraform.")
HALLUCINATED = "Great fit: you know Go, and your Rust and Snowflake work maps directly to this role."


@pytest.fixture(scope="module")
def extractor() -> SkillExtractor:
    return SkillExtractor(load_vocab())


class DictCache:
    def __init__(self):
        self.data = {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value):
        self.data[key] = value


class FailingChatModel(BaseChatModel):
    """Simulates an API timeout."""

    @property
    def _llm_type(self) -> str:
        return "failing"

    def _generate(self, *args, **kwargs):
        raise TimeoutError("simulated timeout")


def run(extractor, llm, cache=None, matches=(MATCH,), **kw):
    return explain_matches(list(matches), {7: "We build backend services."}, "Resume text", extractor,
                           llm=llm, model_name="fake", cache=cache, **kw)


def test_grounded_llm_output_is_used(extractor):
    out = run(extractor, FakeListChatModel(responses=[GROUNDED]))[0]
    assert out["explanation"] == GROUNDED and out["explanation_source"] == "llm"


def test_every_named_skill_is_in_matched_or_missing(extractor):
    for text in (GROUNDED, template_explanation(MATCH)):
        assert ungrounded_skills(text, MATCH, extractor) == set()
    assert ungrounded_skills(HALLUCINATED, MATCH, extractor) == {"Rust", "Snowflake"}


def test_hallucination_is_regenerated_once(extractor):
    llm = FakeListChatModel(responses=[HALLUCINATED, GROUNDED])
    out = run(extractor, llm)[0]
    assert out["explanation"] == GROUNDED and out["explanation_source"] == "llm"


def test_persistent_hallucination_falls_back_to_template(extractor):
    out = run(extractor, FakeListChatModel(responses=[HALLUCINATED, HALLUCINATED]))[0]
    assert out["explanation_source"] == "template"
    assert ungrounded_skills(out["explanation"], MATCH, extractor) == set()


def test_llm_failure_degrades_to_template(extractor):
    out = run(extractor, FailingChatModel())[0]
    assert out["explanation"] == template_explanation(MATCH)
    assert out["explanation_source"] == "template"


def test_no_llm_configured_uses_template(extractor):
    assert run(extractor, None)[0]["explanation_source"] == "template"
    assert make_llm(Settings("postgresql://x", "all-MiniLM-L6-v2", "openai", None)) is None


def test_cache_prevents_second_llm_call(extractor):
    cache, llm = DictCache(), FakeListChatModel(responses=[GROUNDED, "SECOND CALL"])
    first = run(extractor, llm, cache)[0]["explanation"]
    second = run(extractor, llm, cache)[0]["explanation"]
    assert first == second == GROUNDED
    assert len(cache.data) == 1


def test_only_top_n_use_the_llm(extractor):
    matches = [{**MATCH, "job_id": i} for i in range(4)]
    out = run(extractor, FakeListChatModel(responses=[GROUNDED]), matches=matches, llm_top_n=2)
    assert [m["explanation_source"] for m in out] == ["llm", "llm", "template", "template"]


def test_template_grammar_for_one_gap():
    assert "The main gap is Kafka." in template_explanation({**MATCH, "missing": ["Kafka"]})


def test_template_handles_jobs_without_skills():
    text = template_explanation({**MATCH, "matched": [], "missing": []})
    assert "Backend Engineer" in text and "no specific skills" in text
