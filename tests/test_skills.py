import pytest

from jobforge.skills import SkillExtractor, filter_candidates, load_vocab, mentions_company


@pytest.fixture(scope="module")
def extractor() -> SkillExtractor:
    return SkillExtractor(load_vocab())


@pytest.mark.parametrize("text, expected", [
    ("Strong Python and SQL skills", {"Python", "SQL"}),
    ("Experience with Postgres and k8s", {"PostgreSQL", "Kubernetes"}),  # alias normalization
    ("PYTORCH or tensorflow", {"PyTorch", "TensorFlow"}),  # case-insensitive by default
    ("We use C++, C# and .NET", {"C++", "C#", ".NET"}),
    ("Build CI/CD with GitHub Actions", {"CI/CD", "GitHub Actions"}),
    ("Shipping LLMs with RAG", {"large language models", "retrieval-augmented generation"}),
])
def test_finds_and_normalizes_skills(extractor, text, expected):
    assert extractor.extract(text) == expected


def test_longest_match_wins(extractor):
    assert extractor.extract("Mobile apps in React Native") == {"React Native"}
    assert extractor.extract("Ruby on Rails backend") == {"Ruby on Rails"}


@pytest.mark.parametrize("text", [
    "You will excel in a fast-paced environment",  # Excel is case-sensitive
    "We react quickly to customer feedback",
    "Guard rails for safe deployment",
    "An agile team that loves to spark ideas",
    "Plan your workday",
])
def test_common_words_are_not_skills(extractor, text):
    assert extractor.extract(text) == set()


def test_case_sensitive_skills_match_when_capitalized(extractor):
    assert extractor.extract("Build dashboards in Excel and React") == {"Excel", "React", "data visualization"}


@pytest.mark.parametrize("text, expected", [
    ("Backend services in Go", {"Go"}),
    ("Proficient in Go, Rust, or Java", {"Go", "Rust", "Java"}),
    ("Statistics with R", {"R", "statistics"}),
    ("Systems programming in C/C++", {"C", "C++"}),
    ("Go-to-market strategy", {"go-to-market"}),
    ("Go above and beyond for customers", set()),
    ("Our Series C funding round", set()),
    ("Partner with R&D teams", set()),
])
def test_strict_skills_need_context(extractor, text, expected):
    assert extractor.extract(text) == expected


def test_extract_many_matches_extract(extractor):
    texts = ["Python and Go, Rust", "Excel", ""]
    assert extractor.extract_many(texts) == [extractor.extract(t) for t in texts]


def test_vocab_rejects_conflicting_aliases(tmp_path):
    path = tmp_path / "skills.csv"
    path.write_text("skill,aliases,match\nPostgreSQL,Postgres,\nPostGIS,postgres,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="maps to both"):
        load_vocab(path)


def test_keybert_candidates_keep_noun_phrases_outside_the_vocab(extractor):
    phrases = ["financial reporting", "autonomous vehicles", "automate", "role experience",
               "python developer", "postgres tuning", "years 2026", "build agentic"]
    kept = filter_candidates(phrases, extractor.vocab, extractor.nlp)
    assert "financial reporting" in kept and "autonomous vehicles" in kept
    for dropped in ("automate", "role experience", "python developer", "postgres tuning", "years 2026", "build agentic"):
        assert dropped not in kept


@pytest.mark.parametrize("phrase, company, expected", [
    ("databricks anticipates", "Databricks", True),
    ("asanas option", "Asana", True),  # possessive with the apostrophe stripped
    ("scale ai platform", "Scale AI", True),
    ("robotics platform", "Ro", False),  # short names only match whole words
    ("ro pharmacy", "Ro", True),
    ("financial analysis", "Coinbase", False),
])
def test_mentions_company(phrase, company, expected):
    assert mentions_company(phrase, company) is expected
