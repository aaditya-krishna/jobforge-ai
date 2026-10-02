"""Sentence embeddings. The model is loaded once and shared (KeyBERT reuses it)."""

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from jobforge.config import get_settings


@lru_cache
def get_model() -> SentenceTransformer:
    return SentenceTransformer(get_settings().embedding_model)


def embed(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Normalized embeddings, so cosine similarity equals the dot product."""
    return get_model().encode(texts, batch_size=batch_size, normalize_embeddings=True,
                              show_progress_bar=len(texts) > 1000)


def chunk_words(text: str, size: int = 180, overlap: int = 40) -> list[str]:
    """Split text into overlapping word windows that fit the model's ~256-token input limit."""
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)] if words else []
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, len(words) - overlap, step)]


def embed_long(text: str) -> np.ndarray:
    """One normalized vector for a text longer than the model reads (e.g. a resume): mean of chunk vectors."""
    chunks = chunk_words(text)
    if not chunks:
        raise ValueError("cannot embed empty text")
    mean = embed(chunks).mean(axis=0)
    return mean / np.linalg.norm(mean)
