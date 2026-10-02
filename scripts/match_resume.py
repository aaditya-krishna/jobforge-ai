"""Match a resume (.pdf, .txt or .md) against indexed jobs and print the top results.

    python scripts/match_resume.py path/to/resume.pdf
    python scripts/match_resume.py data/eval/resumes/backend_engineer.md --w-sem 0.6 --top-n 5
"""

import argparse
import textwrap

from jobforge.config import get_settings
from jobforge.db import connect
from jobforge.explain import DbCache, explain_matches, make_llm
from jobforge.match import load_descriptions, match_resume
from jobforge.resume import load_resume
from jobforge.skills import SkillExtractor, load_vocab


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("resume")
    parser.add_argument("--w-sem", type=float, default=0.7, help="weight of semantic similarity (default 0.7)")
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    text = load_resume(args.resume)
    settings = get_settings()
    extractor = SkillExtractor(load_vocab())
    llm = make_llm(settings)
    with connect() as conn:
        resume_skills, results = match_resume(conn, text, extractor, top_n=args.top_n, w_sem=args.w_sem)
        descriptions = load_descriptions(conn, [r["job_id"] for r in results])
        results = explain_matches(results, descriptions, text, extractor, llm=llm,
                                  model_name=settings.llm_model or "", cache=DbCache(conn))

    print(f"Resume skills ({len(resume_skills)}): {', '.join(sorted(resume_skills))}")
    print(f"Explanations: {'LLM (' + settings.llm_model + ') for the top 5' if llm else 'template (no LLM configured)'}\n")
    for i, r in enumerate(results, 1):
        more = f" (+{len(r['other_locations'])} more locations)" if r["other_locations"] else ""
        print(f"{i:2d}. {r['title']} - {r['company']}, {r['location']}{more}")
        print(f"    score {r['score']:.3f} = semantic {r['semantic']:.3f} / coverage {r['coverage']:.0%}")
        print(f"    matched: {', '.join(r['matched']) or '-'}")
        print(f"    missing: {', '.join(r['missing'][:12]) or '-'}{' ...' if len(r['missing']) > 12 else ''}")
        print(textwrap.fill(f"[{r['explanation_source']}] {r['explanation']}", 100,
                            initial_indent="    ", subsequent_indent="    "))
        print(f"    {r['url']}")


if __name__ == "__main__":
    main()
