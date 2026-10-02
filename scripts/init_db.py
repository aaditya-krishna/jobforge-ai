"""Create the database schema from sql/schema.sql. Safe to rerun.

    python scripts/init_db.py
"""

from jobforge.config import SQL_DIR
from jobforge.db import connect


def main() -> None:
    schema = (SQL_DIR / "schema.sql").read_text(encoding="utf-8")
    with connect(vector=False) as conn:
        conn.execute(schema)  # no parameters, so psycopg allows multiple statements
        tables = conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' ORDER BY table_name"
        ).fetchall()
    print("Tables:", ", ".join(t[0] for t in tables))


if __name__ == "__main__":
    main()
