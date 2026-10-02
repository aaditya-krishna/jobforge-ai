"""Skill extraction: spaCy PhraseMatcher against data/skills.csv, plus filtered KeyBERT keywords.

PhraseMatcher results are the source of truth for matching. KeyBERT phrases are stored
separately (method='keybert') because they are noisier; they surface what the vocabulary misses.
"""

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

import spacy
from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Span
from spacy.util import filter_spans

from jobforge.config import DATA_DIR

SKILLS_CSV = DATA_DIR / "skills.csv"

# For `strict` skills (Go, R, C): words that show the token is used as a name, not an English word
LIST_NEIGHBORS = {",", "/", "(", ")", "and", "or"}
INTRO_WORDS = {"in", "with", "using", "like"}
REJECT_NEXT = {"-", "'s", "’s"}


@dataclass
class SkillVocab:
    """Canonical skills, their match mode, and every term (skill name or alias) that maps to them."""

    term_to_skill: dict[str, str] = field(default_factory=dict)  # term (lowercased unless case-sensitive) -> skill
    mode: dict[str, str] = field(default_factory=dict)  # skill -> "", "case" or "strict"


def load_vocab(path: Path = SKILLS_CSV) -> SkillVocab:
    """Read skills.csv (skill,aliases,match). Raises if one term would map to two skills."""
    vocab = SkillVocab()
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            skill, mode = row["skill"].strip(), (row.get("match") or "").strip()
            if mode not in ("", "case", "strict"):
                raise ValueError(f"{skill}: unknown match mode {mode!r}")
            vocab.mode[skill] = mode
            aliases = [a.strip() for a in (row.get("aliases") or "").split("|") if a.strip()]
            for term in [skill, *aliases]:
                key = term if mode else term.lower()
                existing = vocab.term_to_skill.get(key)
                if existing and existing != skill:
                    raise ValueError(f"term {term!r} maps to both {existing!r} and {skill!r}")
                vocab.term_to_skill[key] = skill
    return vocab


class SkillExtractor:
    """Finds vocabulary skills in text. Only the tokenizer runs, so bulk extraction is fast."""

    def __init__(self, vocab: SkillVocab, nlp: Language | None = None):
        self.vocab = vocab
        self.nlp = nlp or spacy.load("en_core_web_sm")
        self.lower = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self.exact = PhraseMatcher(self.nlp.vocab, attr="ORTH")
        for term, skill in vocab.term_to_skill.items():
            matcher = self.exact if vocab.mode[skill] else self.lower
            matcher.add(skill, [self.nlp.make_doc(term)])

    def _strict_ok(self, doc: Doc, span: Span) -> bool:
        """Accept 'Go'/'R'/'C' only where they read as a language name: in a list, or after 'in'/'with'."""
        prev = doc[span.start - 1].text.lower() if span.start > 0 else None
        nxt = doc[span.end].text.lower() if span.end < len(doc) else None
        if nxt in REJECT_NEXT:
            return False
        return prev in LIST_NEIGHBORS or nxt in LIST_NEIGHBORS or prev in INTRO_WORDS

    def _skills_in(self, doc: Doc) -> set[str]:
        spans = []
        for matcher in (self.lower, self.exact):
            for match_id, start, end in matcher(doc):
                span = Span(doc, start, end, label=match_id)
                if self.vocab.mode[span.label_] == "strict" and not self._strict_ok(doc, span):
                    continue
                spans.append(span)
        # Longest match wins: "React Native" doesn't also count as "React"
        return {span.label_ for span in filter_spans(spans)}

    def extract(self, text: str) -> set[str]:
        return self._skills_in(self.nlp.make_doc(text))

    def extract_many(self, texts: list[str], batch_size: int = 256) -> list[set[str]]:
        return [self._skills_in(doc) for doc in self.nlp.tokenizer.pipe(texts, batch_size=batch_size)]


# --- KeyBERT ---------------------------------------------------------------------------------

# Words that are everywhere in job postings but never a skill on their own
GENERIC_WORDS = set("""
ability able apply application based benefits best bachelor build candidate candidates company companies
compensation culture customers degree employee employees engineer engineers engineering equal excellent
experience experienced global great growth help hybrid ideal impact including job jobs join junior lead
leader leaders level location looking manager managers mission new office offices opportunities opportunity
pay people plus position preferred principal range remote required requirements responsibilities role roles
salary senior skill skills staff strong team teams teammates using work working world year years
""".split())
_WORD = re.compile(r"(?u)\b\w\w+\b")  # CountVectorizer's default token pattern


def _normalize(term: str) -> str:
    return " ".join(_WORD.findall(term.lower()))


def filter_candidates(phrases: list[str], vocab: SkillVocab, nlp: Language) -> list[str]:
    """Keep corpus-level KeyBERT candidates that could plausibly be skills not already in the vocabulary."""
    vocab_terms = {n for n in (_normalize(t) for t in vocab.term_to_skill) if len(n) > 2}
    kept = []
    for phrase, doc in zip(phrases, nlp.pipe(phrases, batch_size=1024)):
        words = phrase.split()
        if any(w in GENERIC_WORDS or any(ch.isdigit() for ch in w) for w in words):
            continue
        padded = f" {phrase} "
        if any(f" {t} " in padded for t in vocab_terms):
            continue  # PhraseMatcher already covers it
        # Noun phrases only: adjectives/nouns, ending in a noun ("financial analysis", not "automate")
        if all(t.pos_ in ("NOUN", "PROPN", "ADJ") for t in doc) and doc[-1].pos_ in ("NOUN", "PROPN"):
            kept.append(phrase)
    return kept


def mentions_company(phrase: str, company: str) -> bool:
    """True if the phrase contains the company's name, including possessives like 'asanas'."""
    name = _normalize(company).replace(" ", "")
    if len(name) < 4:  # short names like "Ro" would otherwise match "robotics"
        return name in phrase.split()
    return any(w.startswith(name) for w in phrase.split()) or name in phrase.replace(" ", "")
