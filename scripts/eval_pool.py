"""List every role in each evaluation resume's candidate pool that still needs a relevance label.

Writes data/eval/to_label.csv (resume, company, title, location, snippet). Copy rows into
data/eval/labels.csv with relevant = 1 or 0, then run scripts/evaluate.py.

    python scripts/eval_pool.py
"""

import csv

from jobforge.db import connect
from jobforge.evaluation import EVAL_DIR, build_pool, label_key, load_labels, load_resumes
from jobforge.match import load_descriptions
from jobforge.skills import SkillExtractor, load_vocab


def main() -> None:
    labels, extractor = load_labels(), SkillExtractor(load_vocab())
    rows = []
    with connect() as conn:
        for name, text in load_resumes().items():
            pool = build_pool(conn, name, text, extractor)
            descriptions = load_descriptions(conn, [c["job_id"] for c in pool.candidates])
            seen = set()
            for c in pool.candidates:
                key = label_key(c["company"], c["title"])
                if key in seen or (name, *key) in labels:
                    continue
                seen.add(key)
                snippet = " ".join(descriptions[c["job_id"]].split()[:30])
                rows.append({"resume": name, "company": c["company"], "title": c["title"],
                             "location": c["location"], "snippet": snippet})
    out = EVAL_DIR / "to_label.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["resume", "company", "title", "location", "snippet"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} roles need labels -> {out}")


if __name__ == "__main__":
    main()
