"""Resume parsing: PDF (or plain text) to text."""

import re
from pathlib import Path
from typing import BinaryIO

import pdfplumber


def pdf_to_text(source: str | Path | BinaryIO) -> str:
    """Extract text from a PDF path or file-like object (e.g. a Streamlit upload)."""
    with pdfplumber.open(source) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    return re.sub(r"[ \t]+", " ", text).strip()


def load_resume(path: str | Path) -> str:
    """Read a resume from .pdf, .txt or .md."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        return pdf_to_text(path)
    return path.read_text(encoding="utf-8").strip()
