"""Turn raw posting HTML into clean plain text and strip per-company boilerplate.

Text is kept as one line per block (paragraph, list item, heading) so that boilerplate can be
detected as repeated lines, and so LLM excerpts in Phase 6 stay readable.
"""

import html
import re
from collections import Counter

from bs4 import BeautifulSoup, NavigableString

BLOCK_TAGS = ["p", "div", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "table", "section", "blockquote"]
# Greenhouse boards put the company-wide intro and legal footer in these wrappers
BOILERPLATE_SELECTORS = "div.content-intro, div.content-conclusion"
_WS = re.compile(r"\s+")


def _looks_escaped(raw: str) -> bool:
    """Greenhouse sends HTML as escaped text (&lt;p&gt;); real HTML has literal tags."""
    return "<" not in raw and "&lt;" in raw


def html_to_lines(raw: str) -> list[str]:
    """Convert posting HTML to a list of non-empty text lines, one per block element."""
    if not raw or not raw.strip():
        return []
    if _looks_escaped(raw):
        raw = html.unescape(raw)  # exactly once: a remaining &lt; is a literal "<" in the text
    soup = BeautifulSoup(raw, "html.parser")

    for tag in soup(["script", "style"]):
        tag.decompose()
    for tag in soup.select(BOILERPLATE_SELECTORS):
        tag.decompose()
    # Newlines in HTML source are just whitespace; flatten them so only block tags create lines.
    # Exact type check skips comments, which get_text() would otherwise start including.
    for s in soup.find_all(string=True):
        if type(s) is NavigableString:
            s.replace_with(_WS.sub(" ", s))
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for li in soup.find_all("li"):
        li.insert(0, "- ")
    for tag in soup.find_all(BLOCK_TAGS):
        tag.insert_before("\n")
        tag.insert_after("\n")

    lines: list[str] = []
    for line in soup.get_text().split("\n"):
        line = _WS.sub(" ", line).strip()  # \s also matches non-breaking spaces
        if not line:
            continue
        if lines and lines[-1] == "-":  # <li><p>text</p></li> splits the bullet from its text
            lines[-1] = f"- {line}"
        else:
            lines.append(line)
    return [line for line in lines if line != "-"]


def _key(line: str) -> str:
    """Compare lines ignoring bullets and case, so '- Foo' and 'foo' count as the same paragraph."""
    return line.removeprefix("- ").lower()


def find_boilerplate(docs: list[list[str]], min_share: float = 0.5, min_docs: int = 5,
                     min_words: int = 8) -> set[str]:
    """Lines that repeat across many postings from the same company.

    A line is boilerplate if it has at least `min_words` words (short headers like
    "Requirements" are kept) and appears in at least `min_share` of the postings,
    and in no fewer than `min_docs` of them (so small companies aren't over-trimmed).
    """
    counts = Counter()
    for lines in docs:
        counts.update({_key(line) for line in lines if len(line.split()) >= min_words})
    threshold = max(min_docs, min_share * len(docs))
    return {key for key, n in counts.items() if n >= threshold}


def clean_description(lines: list[str], boilerplate: set[str] = frozenset()) -> str:
    """Join lines into the final description, dropping boilerplate lines."""
    return "\n".join(line for line in lines if _key(line) not in boilerplate)


def html_to_text(raw: str) -> str:
    """Single-posting convenience: HTML to clean text, without cross-posting boilerplate removal."""
    return clean_description(html_to_lines(raw))
