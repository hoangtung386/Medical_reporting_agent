"""
Report generation module initialization.
"""

from .medgemma import (
    SegmentationGuidedReportGenerator,
    SegmentationAwareAttention,
    ReportGeneratorTrainer
)

__all__ = [
    'SegmentationGuidedReportGenerator',
    'SegmentationAwareAttention', 
    'ReportGeneratorTrainer'
]
