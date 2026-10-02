# JobForge AI: A Build-It-Yourself Learning Guide

JobForge AI is an LLM-powered job matching platform. You give it a resume, and it finds the job postings that fit best, shows which skills match and which are missing, and uses an LLM to explain each match in plain language.

This guide walks you through building it from zero, one phase at a time. Each phase explains the concepts, gives you the steps and starter code, and tells you exactly when you're done. By the end you'll have built a real retrieval-augmented generation (RAG) system and be able to explain every part of it in an interview.

---

## Table of contents

1. [What you'll learn](#1-what-youll-learn)
2. [How the system works](#2-how-the-system-works)
3. [Prerequisites](#3-prerequisites)
4. [Tech stack and why each piece is there](#4-tech-stack-and-why-each-piece-is-there)
5. [Project structure](#5-project-structure)
6. [Phase 0: Setup](#phase-0-setup)
7. [Phase 1: Collecting job postings](#phase-1-collecting-job-postings)
8. [Phase 2: Cleaning data and designing the database](#phase-2-cleaning-data-and-designing-the-database)
9. [Phase 3: Skill extraction](#phase-3-skill-extraction)
10. [Phase 4: Embeddings and vector search with pgvector](#phase-4-embeddings-and-vector-search-with-pgvector)
11. [Phase 5: Matching and ranking](#phase-5-matching-and-ranking)
12. [Phase 6: LLM explanations with RAG](#phase-6-llm-explanations-with-rag)
13. [Phase 7: The Streamlit app](#phase-7-the-streamlit-app)
14. [Phase 8: Evaluation](#phase-8-evaluation)
15. [Phase 9: Packaging and polish](#phase-9-packaging-and-polish)
16. [Stretch goals](#stretch-goals)
17. [Glossary](#glossary)
18. [Describing the project honestly](#describing-the-project-honestly)

---

## 1. What you'll learn

- **Data engineering:** collecting data from APIs, cleaning messy text, designing a relational schema, loading data into PostgreSQL.
- **NLP:** extracting skills from text with rule-based matching (spaCy) and keyword extraction (KeyBERT).
- **Embeddings and vector search:** turning text into vectors, storing them in PostgreSQL with pgvector, and running similarity search with an index.
- **Retrieval and ranking:** combining semantic similarity with skill overlap into one score.
- **RAG and LLMs:** retrieving the right context and passing it to an LLM to generate grounded explanations.
- **Evaluation:** measuring whether your matches are actually good, using precision@k.
- **Shipping:** a Streamlit UI, Docker Compose, tests, and a clean README.

---

## 2. How the system works

```mermaid
flowchart LR
    A[Job sources<br/>APIs / datasets] --> B[Ingestion<br/>+ cleaning]
    B --> C[(PostgreSQL<br/>jobs table)]
    C --> D[Skill extraction<br/>spaCy + KeyBERT]
    C --> E[Embeddings<br/>sentence-transformers]
    D --> F[(job_skills)]
    E --> G[(job_embeddings<br/>pgvector)]
    R[Resume PDF] --> P[Parse + extract skills<br/>+ embed]
    P --> S[Retrieve top-k jobs<br/>vector search]
    G --> S
    S --> H[Re-rank<br/>semantic + skill overlap]
    F --> H
    H --> L[LLM explanation<br/>RAG]
    L --> U[Streamlit app]
```

There are two flows:

- **Offline (indexing):** collect jobs → clean → store → extract skills → create embeddings → store vectors. You run this once, then again whenever you add new jobs.
- **Online (matching):** a user uploads a resume → parse text → extract skills and embed it → retrieve the most similar jobs → re-rank them → generate an explanation for the top matches → show results.

This split is the core idea behind most RAG systems: do the expensive work ahead of time so each query is fast.

---

## 3. Prerequisites

You should be comfortable with:

- Python basics: functions, classes, virtual environments, `pip`
- Basic SQL: `CREATE TABLE`, `INSERT`, `SELECT`, `JOIN`
- Pandas for loading and cleaning data
- Git and GitHub

You'll also need Docker Desktop (to run PostgreSQL with pgvector) and an API key for an LLM provider (OpenAI or Anthropic). Phases 0 to 5 need no API key and cost nothing.

---

## 4. Tech stack and why each piece is there

| Layer | Tool | Why this one |
|---|---|---|
| Language | Python 3.11+ | The standard for ML and NLP work |
| Database | PostgreSQL 16 + **pgvector** | One database for both regular data and vectors; no separate vector DB to manage |
| Skill extraction | **spaCy** `PhraseMatcher` | Fast, exact matching against a known skills list |
| Keyword extraction | **KeyBERT** | Finds important phrases your skills list doesn't cover |
| Embeddings | **sentence-transformers** (`all-MiniLM-L6-v2`) | Free, runs locally, 384-dimensional vectors, good quality for its size |
| LLM orchestration | **LangChain** | Prompt templates and a clean interface to swap LLM providers |
| LLM | OpenAI or Anthropic API | Generates the match explanations |
| Resume parsing | **pdfplumber** | Pulls text out of PDF resumes |
| UI | **Streamlit** | A working web app in pure Python |
| Packaging | **Docker Compose** | Anyone can run the whole project with one command |
| Testing | **pytest** | Keeps each piece honest as the project grows |

LangChain's APIs change often. Pin exact versions in `requirements.txt` and check the current docs if an import fails.

---

## 5. Project structure

```
jobforge-ai/
├── CLAUDE.md
├── README.md
├── LEARNING_GUIDE.md
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
├── data/
│   ├── raw/                  # untouched downloads (gitignored)
│   ├── processed/            # cleaned files (gitignored)
│   ├── skills.csv            # your skills vocabulary
│   └── eval/                 # labeled resume-job pairs for evaluation
├── sql/
│   └── schema.sql
├── src/jobforge/
│   ├── config.py             # settings loaded from .env
│   ├── db.py                 # database connection helpers
│   ├── ingest/               # Phase 1: fetch job postings
│   ├── clean.py              # Phase 2: text cleaning
│   ├── skills.py             # Phase 3: skill extraction
│   ├── embed.py              # Phase 4: embeddings
│   ├── match.py              # Phase 5: retrieval + ranking
│   ├── explain.py            # Phase 6: LLM explanations
│   └── resume.py             # resume parsing
├── app/
│   └── streamlit_app.py      # Phase 7
├── scripts/                  # one-off runnable scripts (load, index, evaluate)
└── tests/
```

---

## Phase 0: Setup

**Goal:** a repo, a Python environment, and a running PostgreSQL database with pgvector.

**Steps**

1. Create the repo and the folder structure above.
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   ```
3. Create `docker-compose.yml` for the database:
   ```yaml
   services:
     db:
       image: pgvector/pgvector:pg16
       environment:
         POSTGRES_USER: jobforge
         POSTGRES_PASSWORD: jobforge
         POSTGRES_DB: jobforge
       ports:
         - "5432:5432"
       volumes:
         - pgdata:/var/lib/postgresql/data
   volumes:
     pgdata:
   ```
4. Start it with `docker compose up -d`.
5. Create `.env.example` (committed) and `.env` (gitignored):
   ```
   DATABASE_URL=postgresql://jobforge:jobforge@localhost:5432/jobforge
   EMBEDDING_MODEL=all-MiniLM-L6-v2
   LLM_PROVIDER=openai
   LLM_MODEL=your-model-name
   OPENAI_API_KEY=
   ANTHROPIC_API_KEY=
   ```
6. Add `.env`, `.venv/`, and `data/raw/` to `.gitignore`.

**Done when:** `psql postgresql://jobforge:jobforge@localhost:5432/jobforge -c "SELECT 1;"` returns 1, and your first commit is on GitHub.

**Common pitfall:** committing `.env` with an API key. Check `git status` before every early commit.

---

## Phase 1: Collecting job postings

**Goal:** 1,000 or more real job postings saved to `data/raw/`.

**Concepts**

- **APIs vs. scraping.** Many companies publish their openings through public job-board APIs, which return clean JSON. That's easier and more reliable than scraping HTML. Avoid scraping sites like LinkedIn; it breaks their terms of service and the pages change constantly.
- **Idempotent ingestion.** Running your fetch script twice shouldn't create duplicates. Give every posting a stable ID (`source` + the source's own ID).

**Good data sources**

- **Greenhouse job boards:** `https://boards-api.greenhouse.io/v1/boards/{company}/jobs?content=true`
- **Lever job boards:** `https://api.lever.co/v0/postings/{company}?mode=json`
- **Public datasets:** Kaggle has large job-posting datasets you can download as CSV. Good for getting started fast.

Always check each source's terms before using it.

**Steps**

1. Pick 20 to 50 companies whose job boards use Greenhouse or Lever (tech companies often do).
2. Write `src/jobforge/ingest/greenhouse.py` that fetches one company's jobs and returns a list of dicts.
3. Normalize every source into the same shape:
   ```python
   {
       "source": "greenhouse",
       "external_id": "4012345",
       "title": "Machine Learning Engineer",
       "company": "examplecorp",
       "location": "Boston, MA",
       "description_html": "<p>...</p>",
       "url": "https://...",
       "posted_at": "2026-09-01",
   }
   ```
4. Save raw results to `data/raw/{source}_{company}.json`.
5. Add a short `time.sleep()` between requests so you're polite to the API.

**Starter code**

```python
import requests

def fetch_greenhouse(company: str) -> list[dict]:
    url = f"https://boards-api.greenhouse.io/v1/boards/{company}/jobs"
    resp = requests.get(url, params={"content": "true"}, timeout=30)
    resp.raise_for_status()
    jobs = []
    for j in resp.json().get("jobs", []):
        jobs.append({
            "source": "greenhouse",
            "external_id": str(j["id"]),
            "title": j.get("title"),
            "company": company,
            "location": (j.get("location") or {}).get("name"),
            "description_html": j.get("content", ""),
            "url": j.get("absolute_url"),
            "posted_at": j.get("updated_at"),
        })
    return jobs
```

**Done when:** you have at least 1,000 postings saved, and running the script again doesn't duplicate anything.

**Interview questions you should be able to answer**

- Why did you use APIs instead of scraping?
- How do you avoid duplicate postings?

---

## Phase 2: Cleaning data and designing the database

**Goal:** clean job text stored in PostgreSQL with a sensible schema.

**Concepts**

- Job descriptions often arrive as HTML (sometimes HTML-escaped HTML). Convert to plain text, collapse whitespace, and drop boilerplate like equal-opportunity statements if you want.
- **Normalization:** jobs, skills, and embeddings live in separate tables linked by `job_id`. That keeps each table focused and makes it easy to re-run one step (say, re-embedding with a new model) without touching the others.

**Schema (`sql/schema.sql`)**

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS jobs (
    id           SERIAL PRIMARY KEY,
    source       TEXT NOT NULL,
    external_id  TEXT NOT NULL,
    title        TEXT NOT NULL,
    company      TEXT,
    location     TEXT,
    description  TEXT NOT NULL,
    url          TEXT,
    posted_at    DATE,
    created_at   TIMESTAMPTZ DEFAULT now(),
    UNIQUE (source, external_id)
);

CREATE TABLE IF NOT EXISTS job_skills (
    job_id  INT REFERENCES jobs(id) ON DELETE CASCADE,
    skill   TEXT NOT NULL,
    method  TEXT NOT NULL,          -- 'phrase_match' or 'keybert'
    PRIMARY KEY (job_id, skill)
);

CREATE TABLE IF NOT EXISTS job_embeddings (
    job_id     INT PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
    model      TEXT NOT NULL,
    embedding  vector(384) NOT NULL -- must match your model's dimension
);
```

**Steps**

1. Write `clean.py` with a `html_to_text()` function (BeautifulSoup's `get_text(" ")` works well, then collapse whitespace with a regex). Run `html.unescape()` first if the HTML comes escaped.
2. Write `scripts/load_jobs.py` that reads `data/raw/`, cleans each posting, and inserts it.
3. Use `INSERT ... ON CONFLICT (source, external_id) DO NOTHING` so reloading is safe.
4. Drop postings with very short descriptions (under ~200 characters); they're usually broken.

**Done when:** `SELECT COUNT(*) FROM jobs;` shows your postings, and a random sample of descriptions reads as clean text.

**Interview questions**

- Why store skills and embeddings in separate tables?
- What does `ON CONFLICT DO NOTHING` do, and why do you need it?

---

## Phase 3: Skill extraction

**Goal:** a list of skills for every job, stored in `job_skills`, and a function that does the same for a resume.

**Concepts**

- **Rule-based matching** (spaCy `PhraseMatcher`) finds exact phrases from a known list. It's precise: if it says "PyTorch," the text really says PyTorch. But it only finds skills you listed.
- **Keyword extraction** (KeyBERT) uses embeddings to find the phrases most representative of a document. It catches things your list missed, but it's noisier.
- Using both, and tagging which method found each skill, gives you precision plus coverage.
- **Normalization:** "Postgres," "PostgreSQL," and "postgresql" should all become one skill. Keep an alias map.

**Steps**

1. Build `data/skills.csv` with columns `skill,aliases`. Start with 200 to 300 skills you care about (languages, frameworks, cloud, databases, ML concepts). Public taxonomies like O*NET and ESCO can help you expand it later.
2. Build the matcher:
   ```python
   import spacy
   from spacy.matcher import PhraseMatcher

   nlp = spacy.load("en_core_web_sm")   # python -m spacy download en_core_web_sm

   def build_matcher(skill_to_aliases: dict[str, list[str]]):
       matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
       alias_to_skill = {}
       for skill, aliases in skill_to_aliases.items():
           for term in [skill, *aliases]:
               alias_to_skill[term.lower()] = skill
           matcher.add(skill, [nlp.make_doc(t) for t in [skill, *aliases]])
       return matcher, alias_to_skill

   def extract_skills(text: str, matcher, alias_to_skill) -> set[str]:
       doc = nlp(text)
       return {alias_to_skill[doc[s:e].text.lower()] for _, s, e in matcher(doc)}
   ```
3. Add KeyBERT as a second pass:
   ```python
   from keybert import KeyBERT
   kw_model = KeyBERT(model="all-MiniLM-L6-v2")
   keywords = kw_model.extract_keywords(
       text, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=10
   )   # returns [(phrase, score), ...]
   ```
4. Run extraction over all jobs and insert into `job_skills`. For large volumes, use `nlp.pipe(texts, batch_size=64)`, which is much faster than calling `nlp()` in a loop.
5. Write `tests/test_skills.py` with a few sentences where you know the right answer.

**Done when:** most jobs have 5 or more skills, spot checks look right, and your tests pass.

**Common pitfalls**

- Short, ambiguous skills like "R," "Go," or "C" match ordinary words. Handle them with extra rules (e.g. require uppercase for "R") or leave them out at first.
- KeyBERT returns phrases, not necessarily skills. Filter its output before trusting it.

**Interview questions**

- What's the trade-off between rule-based and embedding-based extraction?
- How did you handle skill synonyms?

---

## Phase 4: Embeddings and vector search with pgvector

**Goal:** every job has a vector in `job_embeddings`, and you can find the jobs most similar to any piece of text.

**Concepts**

- An **embedding** is a list of numbers representing the meaning of a text. Texts with similar meanings get vectors that point in similar directions.
- **Cosine similarity** measures how closely two vectors point the same way (1 = identical direction). pgvector's `<=>` operator returns cosine **distance**, which is `1 − similarity`, so smaller means more similar.
- **Normalizing** vectors to length 1 makes cosine similarity equal to the dot product, which is faster and simpler.
- An **HNSW index** makes search fast by approximating the nearest neighbors instead of comparing against every row. It's approximate, which is fine for this use case.
- **What to embed:** job title plus description works well. Embedding the title alone loses too much detail.

**Steps**

1. Write `embed.py`:
   ```python
   from sentence_transformers import SentenceTransformer

   model = SentenceTransformer("all-MiniLM-L6-v2")   # 384 dimensions

   def embed(texts: list[str]):
       return model.encode(texts, batch_size=64, normalize_embeddings=True)
   ```
2. Write `scripts/index_jobs.py` that embeds `title + "\n" + description` for every job without an embedding, then inserts it:
   ```python
   import psycopg
   from pgvector.psycopg import register_vector

   with psycopg.connect(DATABASE_URL) as conn:
       register_vector(conn)
       conn.execute(
           "INSERT INTO job_embeddings (job_id, model, embedding) VALUES (%s, %s, %s) "
           "ON CONFLICT (job_id) DO UPDATE SET embedding = EXCLUDED.embedding, model = EXCLUDED.model",
           (job_id, "all-MiniLM-L6-v2", vector),   # vector is a numpy array
       )
   ```
3. Add the index after loading (building it after the data is in is faster):
   ```sql
   CREATE INDEX IF NOT EXISTS job_embeddings_hnsw
       ON job_embeddings USING hnsw (embedding vector_cosine_ops);
   ```
4. Try a search:
   ```sql
   SELECT j.id, j.title, j.company, 1 - (e.embedding <=> %(q)s) AS similarity
   FROM job_embeddings e
   JOIN jobs j ON j.id = e.job_id
   ORDER BY e.embedding <=> %(q)s
   LIMIT 20;
   ```

**Done when:** searching for "machine learning engineer building recommendation systems" returns clearly relevant jobs.

**Common pitfalls**

- The `vector(384)` column size must match your model. Switch models and you must change the column and re-embed everything.
- Descriptions longer than the model's input limit get truncated silently. For job postings this is usually fine, since the key information tends to be near the top. Note it as a known limitation.

**Interview questions**

- Why pgvector instead of a dedicated vector database?
- What is HNSW, and what does "approximate" nearest neighbor mean?
- Why normalize embeddings?

---

## Phase 5: Matching and ranking

**Goal:** given a resume, return the top 10 jobs ranked by a score that combines meaning and skills.

**Concepts**

- **Two-stage retrieval:** vector search quickly narrows thousands of jobs down to ~50 candidates. A more careful re-ranking step then orders those 50. This pattern is used in most real search and recommendation systems.
- **Why not semantic similarity alone?** It captures overall fit but can miss hard requirements. A resume and a job might sound alike while the job requires a key skill the resume lacks.
- **Skill coverage** = (skills the job wants that the resume has) ÷ (skills the job wants).

**Steps**

1. Write `resume.py` to extract text from a PDF:
   ```python
   import pdfplumber

   def pdf_to_text(path: str) -> str:
       with pdfplumber.open(path) as pdf:
           return "\n".join(page.extract_text() or "" for page in pdf.pages)
   ```
2. In `match.py`:
   - Embed the resume and extract its skills (reuse your Phase 3 and 4 functions).
   - Retrieve the top 50 jobs by vector similarity.
   - Load each candidate's skills from `job_skills`.
   - Compute the final score:
     ```python
     def score(semantic: float, resume_skills: set, job_skills: set,
               w_sem: float = 0.7) -> dict:
         matched = resume_skills & job_skills
         coverage = len(matched) / len(job_skills) if job_skills else 0.0
         return {
             "score": w_sem * semantic + (1 - w_sem) * coverage,
             "semantic": semantic,
             "coverage": coverage,
             "matched": sorted(matched),
             "missing": sorted(job_skills - resume_skills),
         }
     ```
   - Sort by `score` and return the top 10.
3. Start with `w_sem = 0.7`. You'll tune it with real measurements in Phase 8.

**Done when:** running `python scripts/match_resume.py path/to/resume.pdf` prints 10 jobs with scores, matched skills, and missing skills that make sense.

**Interview questions**

- Why two stages instead of scoring every job carefully?
- How did you choose the weights? (Answer honestly: you tuned them on your evaluation set.)

---

## Phase 6: LLM explanations with RAG

**Goal:** for each top match, an LLM writes a short explanation of why it fits and what's missing, grounded only in the actual resume and job data.

**Concepts**

- **RAG (retrieval-augmented generation)** means retrieving relevant information first, then giving it to the LLM as context. The LLM generates from that context instead of from memory. Phases 4 and 5 were the retrieval part; this phase adds generation.
- **Grounding:** the LLM should only mention skills that appear in the data you gave it. Without explicit instructions, LLMs tend to invent plausible-sounding details (hallucination).
- **Cost and speed:** only generate explanations for the top few matches, and cache results so the same resume-job pair isn't explained twice.

**Steps**

1. Write the prompt:
   ```python
   from langchain_core.prompts import ChatPromptTemplate

   prompt = ChatPromptTemplate.from_messages([
       ("system",
        "You explain why a job matches a candidate. Use ONLY the facts provided. "
        "Never mention a skill that isn't listed in the provided data. "
        "Write 3-4 sentences: why it fits, then the most important gaps."),
       ("human",
        "Job title: {title}\nCompany: {company}\n"
        "Job description (excerpt):\n{job_excerpt}\n\n"
        "Candidate skills that match: {matched}\n"
        "Skills the job wants that the candidate lacks: {missing}\n"
        "Resume summary (excerpt):\n{resume_excerpt}"),
   ])
   ```
2. Create the model from settings so you can switch providers:
   ```python
   from langchain_openai import ChatOpenAI          # or: from langchain_anthropic import ChatAnthropic

   llm = ChatOpenAI(model=settings.llm_model, temperature=0.2)
   chain = prompt | llm
   explanation = chain.invoke({...}).content
   ```
3. Cache explanations (a simple `explanations` table keyed by `resume_hash, job_id` works).
4. Handle failures: if the API errors or times out, show the match without an explanation instead of crashing.

**Done when:** explanations read naturally, cite only real skills, and a second run on the same resume uses the cache.

**Common pitfalls**

- Pasting the entire resume and full job description into every call is slow and expensive. Send excerpts plus the structured skill lists.
- Check the explanation against the skill lists. A simple test: every skill named in the output should appear in `matched` or `missing`.

**Interview questions**

- What is RAG, and why use it instead of asking the LLM directly?
- How did you reduce hallucinations?
- How did you control cost?

---

## Phase 7: The Streamlit app

**Goal:** a web app where someone uploads a resume and sees ranked matches with explanations.

**Steps**

1. `app/streamlit_app.py` layout:
   - A file uploader for PDF resumes.
   - The skills extracted from the resume, so users can see what the system understood.
   - Filters: location text, minimum score.
   - A results list: title, company, score, matched skills, missing skills, the explanation, and a link to the posting.
2. Load models once with `@st.cache_resource` so the app doesn't reload them on every interaction.
3. Run it with `streamlit run app/streamlit_app.py`.

**Done when:** a friend can upload their resume and get useful results without your help.

---

## Phase 8: Evaluation

**Goal:** numbers that show your system works, and evidence for every design choice.

This is the phase most portfolio projects skip, and it's the one that makes yours stand out.

**Concepts**

- **Precision@k:** of the top k results, what fraction are actually relevant? Precision@5 = 0.8 means 4 of the top 5 were good matches.
- **Ablation:** turning off one component to measure its contribution. For example, comparing semantic-only ranking with your hybrid score shows whether skill coverage actually helps.

**Steps**

1. Create an evaluation set in `data/eval/`: 10 to 20 resumes (yours, friends' with permission, or anonymized samples), and for each, label 20 retrieved jobs as relevant or not relevant.
2. Write `scripts/evaluate.py` that computes precision@5 and precision@10 for:
   - semantic similarity only (`w_sem = 1.0`)
   - skill coverage only (`w_sem = 0.0`)
   - your hybrid at several weights (0.5, 0.6, 0.7, 0.8)
3. Pick the weight with the best results and record it, along with the table, in your README.
4. Check explanation grounding: for a sample of 50 explanations, what percentage mention only skills from the provided data?

**Done when:** your README has a small results table, and every number in it came from this script.

---

## Phase 9: Packaging and polish

**Goal:** anyone can clone the repo and run everything.

**Steps**

1. Add an app service to `docker-compose.yml` so `docker compose up` starts both the database and Streamlit.
2. Write a `Dockerfile` for the app.
3. Pin every package version in `requirements.txt`.
4. Write the README:
   - What it does, with a screenshot or GIF
   - The architecture diagram
   - Quickstart commands
   - The evaluation results table from Phase 8
   - Known limitations (e.g. truncated long descriptions, a skills list focused on tech roles)
5. Make sure `pytest` passes and there are no secrets anywhere in the git history.

**Done when:** you clone the repo into a fresh folder, follow only the README, and everything works.

---

## Stretch goals

Pick these only after Phase 9 is done:

- **Hybrid search:** combine vector search with PostgreSQL full-text search (`tsvector`) for exact keyword matches.
- **A cross-encoder re-ranker:** a model that scores resume-job pairs directly; more accurate than embeddings, but slower.
- **Resume improvement suggestions:** have the LLM suggest how to present existing experience for a specific job, without inventing experience.
- **Scheduled ingestion:** refresh postings daily with Airflow or a cron job, and remove expired ones.
- **A FastAPI backend:** separate the matching logic into an API so other front-ends can use it.
- **Deployment:** host the app on a cloud service with a managed PostgreSQL database.

---

## Glossary

| Term | Meaning |
|---|---|
| **Embedding** | A vector of numbers representing the meaning of a text |
| **Cosine similarity** | How closely two vectors point the same direction (−1 to 1; higher is more similar) |
| **Cosine distance** | 1 − cosine similarity; what pgvector's `<=>` returns |
| **pgvector** | A PostgreSQL extension that adds a vector column type and similarity search |
| **HNSW** | Hierarchical Navigable Small World; a graph-based index for fast approximate nearest-neighbor search |
| **RAG** | Retrieval-augmented generation: retrieve relevant context, then have an LLM generate from it |
| **Grounding** | Keeping an LLM's output tied to the facts it was given |
| **Hallucination** | When an LLM states something not supported by its input |
| **PhraseMatcher** | spaCy's tool for finding exact phrases from a list in text |
| **KeyBERT** | A library that extracts the most representative keywords from a document using embeddings |
| **Two-stage retrieval** | A fast, broad search followed by a slower, careful re-ranking of the top candidates |
| **Precision@k** | Fraction of the top k results that are relevant |
| **Ablation** | Removing one component to measure how much it contributes |
| **Idempotent** | Running an operation more than once has the same effect as running it once |

---

## Describing the project honestly

Once it's built, describe exactly what you did, using the numbers your evaluation produced. A strong bullet sounds like this template, with your real values filled in:

> Built a RAG-based job matching platform over [N] postings using PostgreSQL + pgvector, spaCy/KeyBERT skill extraction, and an LLM for grounded match explanations; hybrid semantic + skill-coverage ranking improved precision@5 from [X] to [Y] over semantic search alone.

Only include a claim if you can open the code and show it. Interviewers often ask you to walk through one piece in depth, and this guide is designed so you can.
