"""Grounded match explanations: an LLM (RAG over the match data) with a deterministic fallback.

Every explanation may only name skills from the match's `matched` and `missing` lists. LLM output
is checked with the PhraseMatcher; ungrounded text is regenerated once, then replaced by the
template. LLM calls happen only for the top results and are cached by (resume hash, job, model).
"""

import hashlib
import logging
from typing import Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from jobforge.config import Settings
from jobforge.skills import SkillExtractor

log = logging.getLogger(__name__)

JOB_EXCERPT_CHARS = 1500
RESUME_EXCERPT_CHARS = 1500

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You explain why a job matches a candidate. Use ONLY the facts provided. "
     "Never mention a skill, tool or technology that isn't in the two skill lists below. "
     "Write 3-4 plain sentences: why it fits, then the most important gaps. No lists, no headings."),
    ("human",
     "Job title: {title}\nCompany: {company}\n"
     "Job description (excerpt):\n{job_excerpt}\n\n"
     "Candidate skills that match: {matched}\n"
     "Skills the job wants that the candidate lacks: {missing}\n\n"
     "Resume (excerpt):\n{resume_excerpt}"),
])
RETRY_NOTE = ("\n\nYour previous answer mentioned skills that are not in the lists. "
              "Mention only skills from the two lists.")


class Cache(Protocol):
    def get(self, key: tuple[str, int, str]) -> str | None: ...
    def set(self, key: tuple[str, int, str], value: str) -> None: ...


def resume_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def _fmt(skills: list[str]) -> str:
    return ", ".join(skills) if skills else "none"


def template_explanation(match: dict) -> str:
    """Deterministic explanation built only from the match data, so it is grounded by construction."""
    matched, missing = match["matched"], match["missing"]
    total = len(matched) + len(missing)
    if not total:
        return (f"This {match['title']} role was matched on the overall similarity of the posting to "
                f"your resume; no specific skills were extracted from the posting.")
    parts = [f"You have {len(matched)} of the {total} skills this posting lists"
             + (f" ({_fmt(matched[:6])})." if matched else ".")]
    if len(missing) == 1:
        parts.append(f"The main gap is {missing[0]}.")
    elif missing:
        parts.append(f"The main gaps are {_fmt(missing[:4])}.")
    else:
        parts.append("It doesn't list any skills you're missing.")
    return " ".join(parts)


def ungrounded_skills(text: str, match: dict, extractor: SkillExtractor) -> set[str]:
    """Skills named in `text` that the match data doesn't provide (the job title and company count as given)."""
    allowed = set(match["matched"]) | set(match["missing"]) | extractor.extract(match.get("title") or "")
    company = (match.get("company") or "").lower()
    return {s for s in extractor.extract(text) if s not in allowed and s.lower() != company}


def make_llm(settings: Settings) -> BaseChatModel | None:
    """Chat model from settings, or None if not configured (explanations then use the template)."""
    if not settings.llm_model:
        return None
    try:
        if settings.llm_provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model=settings.llm_model, temperature=0.2, timeout=30, max_retries=1)
        if settings.llm_provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=settings.llm_model, temperature=0.2, timeout=30, max_retries=1)
    except Exception as e:  # missing API key raises at construction
        log.warning("LLM not available (%s); using template explanations", e)
        return None
    log.warning("Unknown LLM_PROVIDER %r; using template explanations", settings.llm_provider)
    return None


def _llm_explanation(llm: BaseChatModel, match: dict, job_description: str, resume_text: str,
                     extractor: SkillExtractor) -> str | None:
    """One LLM explanation, regenerated once if ungrounded. None on failure or persistent hallucination."""
    inputs = {
        "title": match["title"], "company": match["company"],
        "job_excerpt": job_description[:JOB_EXCERPT_CHARS],
        "matched": _fmt(match["matched"]), "missing": _fmt(match["missing"]),
        "resume_excerpt": resume_text[:RESUME_EXCERPT_CHARS],
    }
    chain = PROMPT | llm
    for attempt in range(2):
        if attempt:
            inputs["resume_excerpt"] = resume_text[:RESUME_EXCERPT_CHARS] + RETRY_NOTE
        try:
            text = chain.invoke(inputs).text.strip()
        except Exception as e:  # timeouts, rate limits, auth errors: degrade, don't crash
            log.warning("LLM call failed for job %s: %s", match.get("job_id"), e)
            return None
        bad = ungrounded_skills(text, match, extractor)
        if text and not bad:
            return text
        log.info("Ungrounded explanation for job %s (mentions %s)", match.get("job_id"), sorted(bad))
    return None


def explain_matches(matches: list[dict], descriptions: dict[int, str], resume_text: str,
                    extractor: SkillExtractor, llm: BaseChatModel | None = None, model_name: str = "",
                    cache: Cache | None = None, llm_top_n: int = 5) -> list[dict]:
    """Attach `explanation` and `explanation_source` ('llm' or 'template') to each match.

    Only the first `llm_top_n` matches use the LLM; cached explanations are reused.
    """
    rhash = resume_hash(resume_text)
    out = []
    for i, m in enumerate(matches):
        text, source = None, "template"
        if llm is not None and i < llm_top_n:
            key = (rhash, m["job_id"], model_name)
            text = cache.get(key) if cache else None
            if text is None:
                text = _llm_explanation(llm, m, descriptions.get(m["job_id"], ""), resume_text, extractor)
                if text and cache:
                    cache.set(key, text)
            source = "llm" if text else "template"
        out.append({**m, "explanation": text or template_explanation(m), "explanation_source": source})
    return out


class DbCache:
    """Explanation cache in the `explanations` table."""

    def __init__(self, conn):
        self.conn = conn

    def get(self, key: tuple[str, int, str]) -> str | None:
        row = self.conn.execute(
            "SELECT explanation FROM explanations WHERE resume_hash = %s AND job_id = %s AND model = %s", key
        ).fetchone()
        return row[0] if row else None

    def set(self, key: tuple[str, int, str], value: str) -> None:
        self.conn.execute(
            "INSERT INTO explanations (resume_hash, job_id, model, explanation) VALUES (%s, %s, %s, %s) "
            "ON CONFLICT (resume_hash, job_id, model) DO UPDATE SET explanation = EXCLUDED.explanation",
            (*key, value),
        )
        self.conn.commit()
