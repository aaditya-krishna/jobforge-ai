"""Database connection helpers."""

import psycopg
from pgvector.psycopg import register_vector

from jobforge.config import get_settings


def connect(vector: bool = True) -> psycopg.Connection:
    """Open a connection to DATABASE_URL.

    With vector=True, numpy arrays map to pgvector's vector type. That needs the extension to
    exist already, so schema setup connects with vector=False.
    """
    conn = psycopg.connect(get_settings().database_url, connect_timeout=10)
    if vector:
        register_vector(conn)
    return conn
