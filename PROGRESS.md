# Progress

Summary of the autonomous run through Phases 3–9 (2026-10-02). Decisions and their reasons are in
[DECISIONS.md](DECISIONS.md); measured results are in [data/eval/results.md](data/eval/results.md).

## Status

| Phase | Status | Commit |
|---|---|---|
| 0 Setup | done | `7238e81` |
| 1 Ingestion | done: 9,790 postings from 40 Greenhouse/Lever boards, idempotent | `ef891ac` |
| 2 Cleaning + loading | done: 9,789 jobs in Postgres, boilerplate removal cut median length 884 → 562 words | `caba201` |
| 3 Skill extraction | done: 298-skill vocabulary; 62% of jobs have ≥5 skills (81% of technical roles) | `0b7fb45` |
| 4 Embeddings + search | done: 9,789 vectors, HNSW index verified in `EXPLAIN` | `8995ec5` |
| 5 Matching + ranking | done: chunked resume embedding, hybrid re-rank, duplicate-role collapse | `fe5eb0a` |
| 6 LLM explanations | **built and tested with a fake LLM; not run against a real LLM (no API key)** | `ea87855` |
| 7 Streamlit app | done: verified in a browser with a pasted resume; location filter via pgvector iterative scans | `d1962d4` |
| 8 Evaluation | done, with caveats below: best P@5 0.817 / P@10 0.808 vs 0.767 / 0.733 semantic-only | `2a26a31` |
| 9 Packaging | done: Dockerfile, compose app service, README; quickstart verified from a fresh clone | `52b90f0` |

All 82 tests pass. No secrets in the git history (scanned every commit for key patterns; `.env` was never tracked).

## What changed from the guide's plan, and why

- **Smoothed coverage became the default** (`matched / (job skills + 5)`, w_sem 0.7). The guide's plain
  hybrid did *not* beat semantic-only at P@5 in the evaluation, because jobs with one or two extracted
  skills got 100% coverage. See the results table.
- **KeyBERT phrases are stored but not used for coverage.** In the evaluation they scored exactly the same
  as smoothing with prior 5, so their apparent benefit came from enlarging the denominator, not from better skills.
- **Explanations always exist.** A template built from the skill lists is the fallback for no key, API
  errors and ungrounded LLM output.
- **Resumes are embedded in overlapping chunks and averaged**, because the model only reads ~200 words.

## Open questions for you

1. **Review the evaluation labels.** All 518 labels in `data/eval/labels.csv` were assigned by me under the
   rules in `data/eval/LABELING.md`. The README says so, but the numbers only become meaningful after a
   person checks them, especially the level rules (Staff counts for 4–5 year resumes; Principal doesn't) and
   borderline families (analytics engineer for the analyst resume, DevSecOps for security).
   After relabeling, rerun `python scripts/evaluate.py` and copy the new table into the README.
2. **Real resumes.** The 12 evaluation resumes are synthetic, and written by the same assistant that labeled
   them. Adding your own resume (and friends', with permission) would make the evaluation far more credible.
3. **LLM for Phase 6.** Pick a provider and model, put them with a key in `.env`
   (`LLM_PROVIDER`, `LLM_MODEL`, `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`), then run
   `python scripts/match_resume.py data/eval/resumes/ml_engineer.md` and `python scripts/evaluate.py`; the
   grounding table will then include LLM explanations. Phase 6 is left unchecked in CLAUDE.md until then.
4. **Held-out evaluation.** The default weights were chosen on the same 12 resumes they're reported on.
   With more resumes, split them into tuning and test sets.
5. **Seniority.** Intern and new-grad roles rank alongside senior ones for experienced resumes. A simple
   title-level feature (intern / new grad / senior / staff / manager) is the most likely next ranking gain;
   the labels already encode level, so it can be measured.
6. **README screenshot.** I couldn't save a screenshot file from the browser; add one of the app.
7. **Vocabulary.** `extract_skills.py` prints the most common KeyBERT phrases, which are candidates for
   `data/skills.csv` (e.g. domain terms like "financial infrastructure"). The non-technical half of the
   postings is thinly covered.

## Verification of the README quickstart

Fresh `git clone` of the pushed repo (commit `52b90f0`) into a scratch folder, run as a separate compose
project on ports 5434/8502 so the working database was untouched:

- `docker compose up -d --build`: built and started; the database turned healthy and the app started after it.
  Torch in the image is `2.14.1+cpu` (no CUDA). The image is 3.1 GB, mostly the ML libraries (2.1 GB of
  site-packages).
- The four pipeline scripts via `docker compose run --rm app ...` with `--only discord`: 51 postings
  fetched, loaded, skill-extracted (median 8 skills) and embedded.
- App health endpoint returned `ok`; `match_resume.py` inside the container returned ranked matches.
- The verification containers, volume and image were then removed. The working database still has all
  9,789 jobs and embeddings.

Not verified from scratch: the full 40-board run inside Docker (it was run locally instead; ~15 minutes),
and LLM explanations (no key).
