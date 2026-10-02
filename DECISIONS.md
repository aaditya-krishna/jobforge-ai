# Decisions

One line per decision, with the reason. Newest phase at the bottom.

## Phase 0: Setup
- **Python 3.12** (installed via winget): safest wheel support for spaCy/torch/KeyBERT; 3.13/3.14 were available but riskier.
- **Postgres on host port 5433**: a native PostgreSQL 18 service already owns 5432; separate ports keep the pgvector DB unambiguous.
- **`127.0.0.1` instead of `localhost` in DATABASE_URL**: `localhost` resolves to `::1` first, which Docker doesn't listen on, and the connection hangs.
- **Container bound to 127.0.0.1 only**: the dev password is `jobforge`; it shouldn't be reachable from the network.
- **Added requests, beautifulsoup4, python-dotenv, pinned torch**: the guide's own code uses the first three; pinning torch keeps the transitive install reproducible.

## Phase 1: Ingestion
- **40 Greenhouse/Lever boards, each probed before adding**: public APIs designed for this; no scraping.
- **Pure `parse_*` separate from `fetch_*`**: parsers are tested offline with sample payloads.
- **Rewrite each board's file, deduped and sorted; atomic writes**: reruns produce byte-identical files; a crash can't leave a half-written file.
- **Lever description = description + lists + additional**: the requirements live in `lists`.
- **`posted_at` from Greenhouse `first_published`**: `updated_at` changes on every edit.

## Phase 2: Cleaning and loading
- **Keep all postings, including non-technical**: lets evaluation show that irrelevant roles rank low (user's choice).
- **One line per HTML block**: enables boilerplate detection by repeated lines and keeps LLM excerpts readable.
- **Unescape Greenhouse HTML exactly once, only when escaped**: a remaining `&lt;` is a literal `<` in the text.
- **Boilerplate = Greenhouse intro/conclusion divs + lines of 8+ words in ≥50% of a company's postings (min 5)**: removed 36% of text with no requirement lines lost in a spot check.
- **`ON CONFLICT DO NOTHING`**: the guide's idempotency rule; edits to loaded postings aren't picked up (known limitation).
- **`scripts/init_db.py` instead of `psql`**: works from PowerShell without loading `.env` into the shell.

## Phase 3: Skill extraction
- **`match` column in skills.csv (`case`, `strict`)**: case-sensitive matching stops "excel", "react", "rails", "agile" matching as skills; `strict` (Go, R, C) also needs list context or "in/with/using" before it.
- **Longest overlapping match wins (`filter_spans`)**: "React Native" doesn't also count as "React".
- **Drop a skill equal to the posting's company name**: "Databricks" at Databricks is the employer, not a requirement (1,844 postings affected).
- **Replaced generic "compliance" with GDPR, HIPAA, PCI DSS, FedRAMP, ISO 27001, GRC**: "compliance" matched 17% of postings, mostly generic usage.
- **KeyBERT stored as `method='keybert'`, not used for coverage**: its phrases are noisy and would inflate every job's "missing" list; kept for vocabulary discovery and an ablation in Phase 8.
- **KeyBERT candidates filtered once at corpus level (noun phrases, min_df 10, max_df 0.3, no generic words, not already in vocab), then company names per posting, MMR diversity 0.5**: removes most noise and embeds each candidate once instead of per document.
- **Only the spaCy tokenizer runs for PhraseMatcher**: matching needs no tagger/parser; the full corpus takes ~30s.

## Phase 4: Embeddings and vector search
- **Embed `title + "\n" + description`** (the guide's recommendation): the title alone loses detail, and cleaned descriptions put the role first.
- **Skip jobs already embedded with the current model; upsert on model change**: reruns are free and switching models needs no manual cleanup.
- **Stop if the model's dimension differs from the `vector(384)` column**: fails loudly instead of with an opaque insert error.
- **HNSW index built after loading, default parameters (m=16, ef_construction=64)**: one bulk build is faster than incremental inserts; defaults are fine at ~10k rows.
- **Retrieval sets `hnsw.ef_search` to max(200, k)**: HNSW returns at most ef_search rows (default 40), which would silently cut a 50-candidate retrieval short.
- **`set_config(..., true)` instead of `SET LOCAL`**: `SET` can't take bind parameters; set_config can, and is transaction-scoped.

## Phase 5: Matching and ranking
- **Resume embedded as the normalized mean of overlapping 180-word chunks**: the model reads only ~200 words, and a resume's skills are spread across all of it.
- **Guide's hybrid score as-is: `0.7 * semantic + 0.3 * coverage`, coverage = matched / job skills (0 if the job has none)**: the starting point; weights and a smoothed coverage are compared in Phase 8.
- **Coverage uses `phrase_match` skills only**: KeyBERT phrases would rarely appear in a resume and would just pad every job's missing list.
- **Same company + title collapses to the best-scoring copy, other locations kept**: many roles are posted once per city and would otherwise fill the top 10.
- **Retrieve 50 candidates, return top 10** (guide defaults).
- **12 synthetic resumes in `data/eval/resumes/`, each labeled as fictional**: no real resumes are available; they cover 10 technical and 2 non-technical role families.
- **PDF test builds a minimal PDF in code**: avoids adding a PDF-writing dependency just for tests.

## Phase 6: LLM explanations
- **Every match gets a deterministic template explanation; the LLM replaces it for the top 5 when configured**: the app always explains a match, and works with no API key.
- **Grounding enforced in code: the PhraseMatcher scans LLM output; any skill outside matched/missing (title and company exempt) triggers one regeneration, then the template**: the prompt asks for grounding, but only a check guarantees it.
- **Any LLM error (timeout, auth, rate limit) degrades to the template**: per the CLAUDE.md rule that external failures shouldn't crash the app.
- **Cache table `explanations(resume_hash, job_id, model)`; sha256 of whitespace-normalized resume text**: a resume/job pair is never paid for twice, and changing the model doesn't serve stale text.
- **Prompt gets 1,500-char excerpts plus the skill lists, temperature 0.2, 30s timeout, 1 retry**: the guide's cost advice; low temperature for factual summaries.
- **Provider and model stay config-only (`LLM_PROVIDER`, `LLM_MODEL`), no default model**: no key was provided, so nothing is hardcoded and a live run is left to the user.
- **Continued past Phase 6 without a live LLM run**: the user asked to keep going unless a secret is needed; everything except the live call is built and tested with a fake model, and the gap is recorded in PROGRESS.md.

## Phase 7: Streamlit app
- **Location filter inside the SQL query, with pgvector 0.8 `hnsw.iterative_scan = relaxed_order`**: post-filtering a top-50 could return nothing; iterative scans keep walking the index until 50 rows pass (verified: "Dublin" still yields 50).
- **Minimum score filter applied after ranking**: the score only exists after re-ranking.
- **Models and the LLM client cached with `@st.cache_resource`, loaded on first match**: the page opens instantly; later matches skip model loading.
- **New DB connection per match instead of a pool**: one query burst per click; avoids adding psycopg_pool.
- **Paste-text box besides PDF upload**: handy for users and lets `AppTest` drive the full flow in tests.
- **`DEFAULT_W_SEM` constant shared by the app, CLI and scoring**: Phase 8 tuning changes one line.
- **Seniority mismatch (interns ranking for experienced resumes) logged, not fixed**: needs measurement first; noted as a known limitation.

## Phase 8: Evaluation
- **Claude labeled the 518 pooled roles under written criteria (`data/eval/LABELING.md`), recorded as `labeler=claude`**: no human labels were available and the user asked to continue without stopping; README states the labels are unreviewed.
- **Label every role in each resume's top-50 semantic pool**: every config re-ranks the same fully labeled set, so no ranked result is unjudged and no config is favored by what got labeled.
- **Labels keyed by (resume, company, title), not posting id**: the same role in several cities collapses to one result and should share one judgment.
- **Level rules in the criteria (no interns/new-grad/managers/distinguished for 3–7 year resumes)**: makes the seniority-mismatch limitation visible in the metrics.
- **Added smoothed coverage (`matched / (n + prior)`) and KeyBERT-in-coverage configs to the guide's list**: tests the "1-of-1 = 100%" weakness found in Phase 5.
- **"+ KeyBERT" tied exactly with prior 5 (0.817 / 0.808), so its gain is denominator smoothing, not better skills**: chose explicit smoothing, which is simpler and explainable.
- **New defaults: w_sem = 0.7, prior = 5 (`DEFAULT_W_SEM`, `DEFAULT_PRIOR` in match.py)**: best P@5 and P@10 in `data/eval/results.md`. Caveat: tuned and reported on the same 12 resumes, no held-out set.
- **Displayed coverage stays the raw share; smoothing only affects the score**: keeps the percentage consistent with the matched/missing lists users see.
- **`to_label.csv` gitignored**: regenerable worksheet that contains posting excerpts.
