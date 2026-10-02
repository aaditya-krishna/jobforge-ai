"""Match a resume (.pdf, .txt or .md) against indexed jobs and print the top results.

    python scripts/match_resume.py path/to/resume.pdf
    python scripts/match_resume.py data/eval/resumes/backend_engineer.md --w-sem 0.6 --top-n 5
"""

import argparse

from jobforge.db import connect
from jobforge.match import match_resume
from jobforge.resume import load_resume
from jobforge.skills import SkillExtractor, load_vocab


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("resume")
    parser.add_argument("--w-sem", type=float, default=0.7, help="weight of semantic similarity (default 0.7)")
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    text = load_resume(args.resume)
    with connect() as conn:
        resume_skills, results = match_resume(conn, text, SkillExtractor(load_vocab()),
                                              top_n=args.top_n, w_sem=args.w_sem)

    print(f"Resume skills ({len(resume_skills)}): {', '.join(sorted(resume_skills))}\n")
    for i, r in enumerate(results, 1):
        more = f" (+{len(r['other_locations'])} more locations)" if r["other_locations"] else ""
        print(f"{i:2d}. {r['title']} - {r['company']}, {r['location']}{more}")
        print(f"    score {r['score']:.3f} = semantic {r['semantic']:.3f} / coverage {r['coverage']:.0%}")
        print(f"    matched: {', '.join(r['matched']) or '-'}")
        print(f"    missing: {', '.join(r['missing'][:12]) or '-'}{' ...' if len(r['missing']) > 12 else ''}")
        print(f"    {r['url']}")


if __name__ == "__main__":
    main()
