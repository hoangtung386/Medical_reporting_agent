"""Tests for utils.metrics module."""

import pytest

from utils.metrics import (
    compute_all_metrics,
    compute_bleu,
    compute_clinical_accuracy,
    compute_rouge,
    evaluate_dataset,
)


class TestComputeBleu:
    """Tests for compute_bleu."""

    def test_identical_text(self):
        ref = ["the liver is normal"]
        hyp = "the liver is normal"
        scores = compute_bleu(ref, hyp)
        assert scores["bleu_1"] > 0.9

    def test_empty_hypothesis(self):
        scores = compute_bleu(["some reference"], "")
        assert scores["bleu_1"] == pytest.approx(0.0, abs=0.01)

    def test_returns_all_ngrams(self):
        scores = compute_bleu(["a b c d e f"], "a b c d e f")
        for n in range(1, 5):
            assert f"bleu_{n}" in scores


class TestComputeRouge:
    """Tests for compute_rouge."""

    def test_identical_text(self):
        scores = compute_rouge(["the liver is normal"], "the liver is normal")
        assert scores.get("rouge1", 0) > 0.9

    def test_returns_expected_keys(self):
        scores = compute_rouge(["hello world"], "hello world")
        assert "rouge1" in scores
        assert "rouge2" in scores
        assert "rougeL" in scores


class TestClinicalAccuracy:
    """Tests for compute_clinical_accuracy."""

    def test_perfect_match(self):
        text = "liver is normal, kidney is normal"
        result = compute_clinical_accuracy(text, text)
        assert result["clinical_f1"] == pytest.approx(1.0)

    def test_no_overlap(self):
        result = compute_clinical_accuracy("liver mass", "hello world")
        assert result["clinical_f1"] == pytest.approx(0.0)

    def test_partial_overlap(self):
        ref = "liver lesion detected. kidney normal."
        hyp = "liver is unremarkable. spleen normal."
        result = compute_clinical_accuracy(ref, hyp)
        assert 0.0 < result["clinical_f1"] < 1.0

    def test_custom_entities(self):
        result = compute_clinical_accuracy(
            "foo bar", "foo baz", clinical_entities=["foo", "bar", "baz"]
        )
        assert result["n_entities_ref"] == 2  # foo, bar
        assert result["n_entities_hyp"] == 2  # foo, baz


class TestComputeAllMetrics:
    """Tests for compute_all_metrics."""

    def test_returns_all_keys(self):
        metrics = compute_all_metrics(
            ["the liver is normal"],
            "the liver is normal",
        )
        assert "bleu_4" in metrics
        assert "meteor" in metrics
        assert "clinical_f1" in metrics


class TestEvaluateDataset:
    """Tests for evaluate_dataset."""

    def test_basic_evaluation(self):
        preds = ["liver normal", "kidney normal"]
        refs = [["liver is normal"], ["kidney is unremarkable"]]
        metrics = evaluate_dataset(preds, refs)
        assert "bleu_1" in metrics

    def test_length_mismatch_raises(self):
        with pytest.raises(AssertionError):
            evaluate_dataset(["a"], [["b"], ["c"]])
