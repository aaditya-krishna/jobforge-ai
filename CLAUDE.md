# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project overview

JobForge AI is an LLM-powered job matching platform. A user uploads a resume; the system retrieves the most relevant job postings using vector search, re-ranks them with a hybrid of semantic similarity and skill coverage, and uses an LLM to generate grounded explanations of each match (why it fits, what's missing).

The full build plan, concepts, and phase-by-phase steps live in `LEARNING_GUIDE.md`. Treat that file as the source of truth for architecture and scope.

## Current status

- [x] Phase 0: Setup (repo, venv, Postgres + pgvector via Docker)
- [x] Phase 1: Collecting job postings (Greenhouse / Lever APIs, public datasets)
- [x] Phase 2: Cleaning data and database schema
- [x] Phase 3: Skill extraction (spaCy PhraseMatcher + KeyBERT)
- [x] Phase 4: Embeddings and pgvector search
- [x] Phase 5: Matching and ranking
- [ ] Phase 6: LLM explanations (RAG): built and tested with a fake LLM; needs an API key in `.env` for a live run
- [x] Phase 7: Streamlit app
- [ ] Phase 8: Evaluation
- [ ] Phase 9: Packaging and polish

Update these checkboxes as phases are completed. Don't start a later phase until the current one meets its "Done when" criteria in `LEARNING_GUIDE.md`.

## How to work with me

This project is being built to learn, not just to finish. Default to **learning mode**:

- Before writing code for a new concept, briefly explain the concept and the approach, then let me try writing it first when I ask to.
- Work in small steps: one function or one file at a time, not whole phases at once.
- When I share code, review it: point out bugs and explain why, rather than silently rewriting everything.
- When I'm stuck, give a hint first, then a partial solution, then a full one if I still need it.
- After finishing something, suggest how to test it and one interview question I should be able to answer about it.

If I say "build mode," you can write complete implementations directly, but still explain the key decisions.

## Tech stack

- Python 3.11+
- PostgreSQL 16 with the pgvector extension (Docker image `pgvector/pgvector:pg16`)
- psycopg 3 + `pgvector` Python package (`register_vector`)
- spaCy (`en_core_web_sm`) for PhraseMatcher skill extraction
- KeyBERT for keyword extraction
- sentence-transformers, model `all-MiniLM-L6-v2` (384 dimensions)
- LangChain (`langchain-core` plus `langchain-openai` or `langchain-anthropic`)
- pdfplumber for resume parsing
- Streamlit for the UI
- pytest for tests
- Docker Compose for local setup

Do not add new dependencies without asking first. When adding one, pin its version in `requirements.txt`.

## Repository structure

```
data/raw/            untouched downloads (gitignored)
data/processed/      cleaned files (gitignored)
data/skills.csv      skills vocabulary: skill,aliases(|-separated),match(""|case|strict)
data/companies.csv   job boards to fetch: source,board,company
data/eval/           labeled resume-job pairs for evaluation
sql/schema.sql       database schema
src/jobforge/        package code
  config.py          settings from .env
  db.py              connection helpers
  ingest/            job source fetchers
  clean.py           text cleaning
  skills.py          skill extraction
  embed.py           embedding functions
  match.py           retrieval + re-ranking
  explain.py         LLM explanations
  resume.py          resume parsing
app/streamlit_app.py UI
scripts/             runnable one-off scripts (load, index, match, evaluate)
tests/               pytest tests
```

## Common commands

```bash
# environment
py -3.12 -m venv .venv          # project uses Python 3.12
.venv\Scripts\activate          # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
pip install -e .                # makes `import jobforge` work in scripts and tests
python -m spacy download en_core_web_sm

# database (Docker maps host port 5433; a native PostgreSQL 18 owns 5432.
# Use 127.0.0.1, not localhost: localhost resolves to ::1 first and hangs)
docker compose up -d db
python scripts/init_db.py      # applies sql/schema.sql; safe to rerun

# pipeline
python scripts/fetch_jobs.py                # boards listed in data/companies.csv
python scripts/fetch_jobs.py --only stripe  # a single board
python scripts/load_jobs.py                 # clean + insert; strips per-company boilerplate
python scripts/extract_skills.py            # new jobs only; --rebuild after editing skills.csv
python scripts/index_jobs.py                # embeds jobs missing a vector, builds HNSW index
python scripts/search.py "ml engineer recommendations"   # ad-hoc semantic search
python scripts/match_resume.py path/to/resume.pdf          # also .txt / .md; --w-sem, --top-n
python scripts/evaluate.py

# app and tests
streamlit run app/streamlit_app.py
pytest
```

Update this section when scripts are added or renamed.

## Database schema

- `jobs(id, source, external_id, title, company, location, description, url, posted_at, created_at)` with `UNIQUE (source, external_id)`
- `job_skills(job_id, skill, method)` where `method` is `phrase_match` or `keybert`. Matching and coverage use `phrase_match` only; `keybert` rows are noisier free-text keywords kept for vocabulary discovery and ablation.
- `job_embeddings(job_id, model, embedding vector(384))` with an HNSW index using `vector_cosine_ops`

The `vector(384)` dimension must match `EMBEDDING_MODEL`. Changing the model means altering the column and re-embedding every job.

## Conventions

- Configuration comes from environment variables loaded in `config.py`. Never hardcode credentials, model names, or connection strings.
- Every ingestion and load step must be idempotent: use `ON CONFLICT` and stable IDs (`source` + `external_id`).
- Embeddings are always normalized (`normalize_embeddings=True`).
- pgvector's `<=>` returns cosine **distance**; similarity is `1 - distance`.
- Use `nlp.pipe()` and batched `model.encode()` for bulk processing, never per-row loops over large datasets.
- Type hints on public functions; short docstrings explaining purpose, not restating the code.
- Use parameterized SQL queries only. Never build SQL with f-strings.
- Keep functions small and testable. Pure logic (like scoring) stays separate from database and API calls.

## Rules

- **Never commit secrets.** `.env` is gitignored; `.env.example` holds empty placeholders. Warn me if a key appears anywhere in tracked files.
- **Don't scrape sites whose terms forbid it** (e.g. LinkedIn). Use public job-board APIs and datasets.
- **LLM output must stay grounded.** Explanations may only mention skills from the provided `matched` and `missing` lists. Keep the grounding test in `tests/` passing.
- **Only generate LLM explanations for top results and cache them**, to control API cost.
- **Never invent metrics.** README results must come from `scripts/evaluate.py`. If a number hasn't been measured, leave it out.
- **Ask before destructive operations**: dropping tables, deleting data files, or rewriting git history.
- Handle external failures gracefully: API timeouts or LLM errors should degrade the result (e.g. show a match without an explanation), not crash the app.

## Testing expectations

- `tests/test_clean.py`: HTML-to-text cleaning on sample inputs
- `tests/test_skills.py`: known sentences produce the expected skills, including alias normalization
- `tests/test_match.py`: scoring math (coverage, weighting, empty skill sets)
- `tests/test_explain.py`: explanations only mention allowed skills (mock the LLM)

Run `pytest` before considering any phase done.

## Known limitations

- Long job descriptions are truncated by the embedding model's input limit. Postings have a median of ~880 words, while all-MiniLM-L6-v2 reads about the first 200. Boilerplate removal in `clean.py` (Greenhouse intro/conclusion divs plus paragraphs repeated in ≥50% of a company's postings) cuts the median to ~560 words, but most postings are still truncated.
- `load_jobs.py` uses `ON CONFLICT DO NOTHING`, so edits to an already-loaded posting are not picked up, and postings removed from a board are never deleted.
- About half of the fetched postings are non-technical (sales, recruiting, ops), which the tech-focused skills vocabulary covers poorly.
- Seniority isn't modeled: intern and staff postings for the same role family can rank side by side, because neither embeddings nor skill coverage capture level.
- Coverage is noisy for jobs with only a few extracted skills (one matched skill out of one gives 100%).
- `posted_at` comes from the board's first-published date; some evergreen postings date back years, so it is not a reliable freshness signal.
- The skills vocabulary is hand-built and focused on tech roles.
- Evaluation set is small and hand-labeled.

Add to this list as new limitations are discovered, and keep it mirrored in the README.
