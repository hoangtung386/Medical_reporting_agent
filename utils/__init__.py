"""
Utility functions for medical report generation.
"""

from .measurements import (
    calculate_volumes,
    get_bounding_boxes,
    calculate_hu_statistics,
    measure_lesion_characteristics,
    calculate_dice_score,
    format_measurements_for_report
)

from .metrics import (
    compute_bleu,
    compute_rouge,
    compute_meteor,
    compute_clinical_accuracy,
    compute_all_metrics,
    evaluate_dataset
)

from .rag import (
    GuidelineRetriever,
    CaseRetriever,
    load_guidelines_from_text
)

__all__ = [
    # Measurements
    'calculate_volumes',
    'get_bounding_boxes',
    'calculate_hu_statistics',
    'measure_lesion_characteristics',
    'calculate_dice_score',
    'format_measurements_for_report',
    
    # Metrics
    'compute_bleu',
    'compute_rouge',
    'compute_meteor',
    'compute_clinical_accuracy',
    'compute_all_metrics',
    'evaluate_dataset',
    
    # RAG
    'GuidelineRetriever',
    'CaseRetriever',
    'load_guidelines_from_text'
]
