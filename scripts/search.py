"""Semantic search over indexed jobs from the command line.

    python scripts/search.py "machine learning engineer building recommendation systems"
"""

import argparse

from jobforge.db import connect
from jobforge.embed import embed
from jobforge.match import retrieve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=20)
    args = parser.parse_args()

    with connect() as conn:
        for i, r in enumerate(retrieve(conn, embed([args.query])[0], args.k), 1):
            print(f"{i:2d}. {r['similarity']:.3f}  {r['title']}  ({r['company']}, {r['location']})")


if __name__ == "__main__":
    main()
