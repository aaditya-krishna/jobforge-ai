"""Embed every job that has no vector for the current EMBEDDING_MODEL, then build the HNSW index.

Idempotent: jobs already embedded with this model are skipped; switching models re-embeds via upsert.

    python scripts/index_jobs.py
"""

import time

from jobforge.config import get_settings
from jobforge.db import connect
from jobforge.embed import embed, get_model

BATCH = 1000

TODO_SQL = """
    SELECT j.id, j.title, j.description
    FROM jobs j
    LEFT JOIN job_embeddings e ON e.job_id = j.id AND e.model = %s
    WHERE e.job_id IS NULL
    ORDER BY j.id
"""
UPSERT_SQL = """
    INSERT INTO job_embeddings (job_id, model, embedding) VALUES (%s, %s, %s)
    ON CONFLICT (job_id) DO UPDATE SET model = EXCLUDED.model, embedding = EXCLUDED.embedding
"""
# Built after loading: inserting into an existing HNSW index is much slower than one bulk build
INDEX_SQL = "CREATE INDEX IF NOT EXISTS job_embeddings_hnsw ON job_embeddings USING hnsw (embedding vector_cosine_ops)"
COLUMN_DIM_SQL = """
    SELECT atttypmod FROM pg_attribute
    WHERE attrelid = 'job_embeddings'::regclass AND attname = 'embedding'
"""


def job_text(title: str, description: str) -> str:
    return f"{title}\n{description}"


def main() -> None:
    model_name = get_settings().embedding_model
    with connect() as conn:
        column_dim = conn.execute(COLUMN_DIM_SQL).fetchone()[0]
        model_dim = get_model().get_embedding_dimension()
        if column_dim != model_dim:
            raise SystemExit(f"{model_name} produces {model_dim}-dim vectors but job_embeddings.embedding is "
                             f"vector({column_dim}). Alter the column and re-embed (see CLAUDE.md).")

        todo = conn.execute(TODO_SQL, (model_name,)).fetchall()
        print(f"{len(todo)} jobs to embed with {model_name}")
        t = time.time()
        for i in range(0, len(todo), BATCH):
            batch = todo[i:i + BATCH]
            vectors = embed([job_text(title, desc) for _, title, desc in batch])
            with conn.cursor() as cur:
                cur.executemany(UPSERT_SQL, [(job_id, model_name, v) for (job_id, _, _), v in zip(batch, vectors)])
            conn.commit()  # keep progress if interrupted
            print(f"  {min(i + BATCH, len(todo))}/{len(todo)} ({time.time() - t:.0f}s)")

        t = time.time()
        conn.execute(INDEX_SQL)
        conn.commit()
        total = conn.execute("SELECT COUNT(*) FROM job_embeddings").fetchone()[0]
        print(f"HNSW index ready ({time.time() - t:.0f}s); {total} jobs embedded")


if __name__ == "__main__":
    main()
