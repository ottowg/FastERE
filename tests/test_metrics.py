"""Tests for evaluation metrics."""

import numpy as np
import pytest

from fastere.utils.metrics import f1_score, f1_score_simple


class TestF1Score:
    def test_perfect_prediction(self):
        outputs = [
            {"pred": [("a", "b", "c")], "gold": [("a", "b", "c")]},
        ]
        f1, p, r = f1_score(outputs, "pred", "gold")
        assert pytest.approx(f1, abs=1e-4) == 1.0
        assert pytest.approx(p, abs=1e-4) == 1.0
        assert pytest.approx(r, abs=1e-4) == 1.0

    def test_no_predictions(self):
        outputs = [
            {"pred": [], "gold": [("a", "b", "c")]},
        ]
        f1, p, r = f1_score(outputs, "pred", "gold")
        assert f1 < 1e-6
        assert p < 1e-6
        assert r < 1e-6

    def test_no_gold(self):
        outputs = [
            {"pred": [("a", "b", "c")], "gold": []},
        ]
        f1, p, r = f1_score(outputs, "pred", "gold")
        assert f1 < 1e-6
        assert p < 1e-6
        assert r < 1e-6

    def test_partial_match(self):
        outputs = [
            {
                "pred": [("a", "b", "c"), ("x", "y", "z")],
                "gold": [("a", "b", "c"), ("p", "q", "r")],
            },
        ]
        f1, p, r = f1_score(outputs, "pred", "gold")
        assert pytest.approx(p, abs=1e-4) == 0.5
        assert pytest.approx(r, abs=1e-4) == 0.5

    def test_slice(self):
        outputs = [
            {
                "pred": [("a", "b", "c", "extra")],
                "gold": [("a", "b", "c", "different")],
            }
        ]
        # Without slice: tuples differ
        f1_no_slice, _, _ = f1_score(outputs, "pred", "gold")
        assert f1_no_slice < 1e-6
        # With slice=3: first 3 elements match
        f1_slice, _, _ = f1_score(outputs, "pred", "gold", slice=3)
        assert pytest.approx(f1_slice, abs=1e-4) == 1.0

    def test_multi_batch(self):
        outputs = [
            {"pred": [("a", "b", "r1")], "gold": [("a", "b", "r1")]},
            {"pred": [("c", "d", "r2")], "gold": [("c", "d", "r2")]},
        ]
        f1, p, r = f1_score(outputs, "pred", "gold")
        assert pytest.approx(f1, abs=1e-4) == 1.0


class TestF1ScoreSimple:
    def test_perfect(self):
        labels = np.array([0, 1, 2, 1, 0])
        pred = np.array([0, 1, 2, 1, 0])
        f1, p, r = f1_score_simple(labels, pred)
        assert pytest.approx(f1, abs=1e-4) == 1.0

    def test_all_zero(self):
        labels = np.array([0, 0, 0])
        pred = np.array([0, 0, 0])
        f1, p, r = f1_score_simple(labels, pred)
        assert f1 < 1e-6
