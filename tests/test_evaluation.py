import pytest

from jobforge.evaluation import label_key, precision_at_k


def test_precision_at_k():
    rel = [True, False, True, True, False, False]
    assert precision_at_k(rel, 5) == pytest.approx(3 / 5)
    assert precision_at_k(rel, 1) == 1.0


def test_precision_counts_missing_results_as_misses():
    assert precision_at_k([True, True], 5) == pytest.approx(2 / 5)
    assert precision_at_k([], 10) == 0.0


def test_label_key_normalizes_case_and_whitespace():
    assert label_key(" Reddit ", "Senior  ML Engineer") == label_key("reddit", "senior ml engineer")
