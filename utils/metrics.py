"""Evaluation metrics for medical report generation.

Implements standard NLG metrics (BLEU, ROUGE, METEOR) and a
domain-specific clinical-accuracy metric.
"""

import logging
import warnings
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from nltk.translate.bleu_score import (
        SmoothingFunction,
        sentence_bleu,
    )
    from nltk.translate.meteor_score import meteor_score as _meteor_score
    import nltk

    try:
        nltk.data.find("wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)

    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("nltk not installed. Some metrics unavailable.")

try:
    from rouge_score import rouge_scorer

    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    logger.warning("rouge-score not installed.")


# ------------------------------------------------------------------
# Individual metrics
# ------------------------------------------------------------------


def compute_bleu(
    references: List[str],
    hypothesis: str,
    max_n: int = 4,
) -> Dict[str, float]:
    """Compute BLEU-1 through BLEU-*max_n* scores.

    Args:
        references: Reference report texts.
        hypothesis: Generated report text.
        max_n: Highest n-gram order.

    Returns:
        Dict mapping ``bleu_1`` ... ``bleu_<max_n>`` to scores.
    """
    if not NLTK_AVAILABLE:
        return {f"bleu_{n}": 0.0 for n in range(1, max_n + 1)}

    ref_tokens = [ref.lower().split() for ref in references]
    hyp_tokens = hypothesis.lower().split()
    smooth = SmoothingFunction().method1

    scores: Dict[str, float] = {}
    for n in range(1, max_n + 1):
        weights = tuple([1.0 / n] * n + [0.0] * (4 - n))
        scores[f"bleu_{n}"] = sentence_bleu(
            ref_tokens,
            hyp_tokens,
            weights=weights,
            smoothing_function=smooth,
        )
    return scores


def compute_rouge(
    references: List[str],
    hypothesis: str,
    rouge_types: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute ROUGE scores (F1 measure).

    Args:
        references: Reference texts.
        hypothesis: Generated text.
        rouge_types: Types to compute. Defaults to
            ``["rouge1", "rouge2", "rougeL"]``.

    Returns:
        Dict mapping each ROUGE type to its F1 score.
    """
    if rouge_types is None:
        rouge_types = ["rouge1", "rouge2", "rougeL"]

    if not ROUGE_AVAILABLE:
        return {rt: 0.0 for rt in rouge_types}

    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)

    all_scores: Dict[str, list[float]] = {rt: [] for rt in rouge_types}
    for reference in references:
        scores = scorer.score(reference, hypothesis)
        for rt in rouge_types:
            all_scores[rt].append(scores[rt].fmeasure)

    return {rt: max(vals) for rt, vals in all_scores.items()}


def compute_meteor(
    references: List[str],
    hypothesis: str,
) -> float:
    """Compute METEOR score.

    Args:
        references: Reference texts.
        hypothesis: Generated text.

    Returns:
        METEOR score (float).
    """
    if not NLTK_AVAILABLE:
        return 0.0

    ref_tokens = [ref.lower().split() for ref in references]
    hyp_tokens = hypothesis.lower().split()
    return float(_meteor_score(ref_tokens, hyp_tokens))


def compute_clinical_accuracy(
    reference: str,
    hypothesis: str,
    clinical_entities: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute entity-level clinical accuracy.

    Checks whether important clinical entities (organs, findings) are
    correctly mentioned.

    Args:
        reference: Ground-truth report.
        hypothesis: Generated report.
        clinical_entities: Entity list. Uses a sensible default when
            ``None``.

    Returns:
        Dict with ``clinical_precision``, ``clinical_recall``,
        ``clinical_f1``, and entity counts.
    """
    if clinical_entities is None:
        clinical_entities = [
            "liver", "kidney", "spleen", "pancreas", "lung",
            "normal", "abnormal", "lesion", "mass", "nodule",
            "unremarkable", "enlarged", "small", "calcification",
        ]

    ref_lower = reference.lower()
    hyp_lower = hypothesis.lower()

    ref_entities = {e for e in clinical_entities if e in ref_lower}
    hyp_entities = {e for e in clinical_entities if e in hyp_lower}

    tp = len(ref_entities & hyp_entities)
    fp = len(hyp_entities - ref_entities)
    fn = len(ref_entities - hyp_entities)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "clinical_precision": precision,
        "clinical_recall": recall,
        "clinical_f1": f1,
        "n_entities_ref": len(ref_entities),
        "n_entities_hyp": len(hyp_entities),
    }


# ------------------------------------------------------------------
# Aggregate helpers
# ------------------------------------------------------------------


def compute_all_metrics(
    references: List[str],
    hypothesis: str,
    detailed: bool = True,
) -> Dict[str, Any]:
    """Compute all available metrics for a single example.

    Args:
        references: Reference report(s).
        hypothesis: Generated report.
        detailed: If ``False``, also include a summary score.

    Returns:
        Merged dict of all metric scores.
    """
    metrics: Dict[str, Any] = {}
    metrics.update(compute_bleu(references, hypothesis))
    metrics.update(compute_rouge(references, hypothesis))
    metrics["meteor"] = compute_meteor(references, hypothesis)
    metrics.update(
        compute_clinical_accuracy(references[0], hypothesis)
    )

    if not detailed:
        key_vals = [
            metrics.get("bleu_4", 0),
            metrics.get("rougeL", 0),
            metrics.get("meteor", 0),
            metrics.get("clinical_f1", 0),
        ]
        metrics["summary_score"] = float(np.mean(key_vals))

    return metrics


def evaluate_dataset(
    predictions: List[str],
    references: List[List[str]],
) -> Dict[str, float]:
    """Evaluate on an entire dataset.

    Args:
        predictions: Generated reports.
        references: List of reference-report lists (multiple refs per
            example supported).

    Returns:
        Averaged metrics across the dataset.
    """
    assert len(predictions) == len(references), (
        "predictions and references must have the same length"
    )

    all_metrics: list[Dict[str, Any]] = [
        compute_all_metrics(refs, pred, detailed=True)
        for pred, refs in zip(predictions, references)
    ]

    metric_names = all_metrics[0].keys()
    return {
        name: float(
            np.mean([m[name] for m in all_metrics if name in m])
        )
        for name in metric_names
    }
