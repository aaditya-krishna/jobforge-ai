-- Safe to run repeatedly: every statement is IF NOT EXISTS.
-- The HNSW index on job_embeddings is created in Phase 4, after vectors are loaded.

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
    method  TEXT NOT NULL CHECK (method IN ('phrase_match', 'keybert')),
    PRIMARY KEY (job_id, skill)
);

CREATE TABLE IF NOT EXISTS job_embeddings (
    job_id     INT PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
    model      TEXT NOT NULL,
    embedding  vector(384) NOT NULL  -- must match EMBEDDING_MODEL's dimension
);

-- Cache of LLM match explanations, so a resume/job pair is never paid for twice
CREATE TABLE IF NOT EXISTS explanations (
    resume_hash  TEXT NOT NULL,
    job_id       INT REFERENCES jobs(id) ON DELETE CASCADE,
    model        TEXT NOT NULL,
    explanation  TEXT NOT NULL,
    created_at   TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (resume_hash, job_id, model)
);
