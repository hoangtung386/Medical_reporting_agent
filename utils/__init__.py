"""Utility functions for medical report generation."""

from .logging import setup_logger
from .measurements import (
    calculate_dice_score,
    calculate_hu_statistics,
    calculate_volumes,
    format_measurements_for_report,
    get_bounding_boxes,
    measure_lesion_characteristics,
)
from .metrics import (
    compute_all_metrics,
    compute_bleu,
    compute_clinical_accuracy,
    compute_meteor,
    compute_rouge,
    evaluate_dataset,
)
from .rag import (
    CaseRetriever,
    GuidelineRetriever,
    load_guidelines_from_text,
)

__all__ = [
    # Logging
    "setup_logger",
    # Measurements
    "calculate_volumes",
    "get_bounding_boxes",
    "calculate_hu_statistics",
    "measure_lesion_characteristics",
    "calculate_dice_score",
    "format_measurements_for_report",
    # Metrics
    "compute_bleu",
    "compute_rouge",
    "compute_meteor",
    "compute_clinical_accuracy",
    "compute_all_metrics",
    "evaluate_dataset",
    # RAG
    "GuidelineRetriever",
    "CaseRetriever",
    "load_guidelines_from_text",
]
