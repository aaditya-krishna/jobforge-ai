import numpy as np
import pytest

from jobforge.embed import embed


@pytest.fixture(scope="module")
def vectors() -> np.ndarray:
    return embed(["machine learning engineer", "ML engineer training models", "pastry chef"])


def test_shape_matches_schema(vectors):
    assert vectors.shape == (3, 384)  # job_embeddings.embedding is vector(384)


def test_vectors_are_normalized(vectors):
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_similar_texts_are_closer(vectors):
    sims = vectors @ vectors.T  # dot product == cosine similarity for unit vectors
    assert sims[0, 1] > sims[0, 2]
