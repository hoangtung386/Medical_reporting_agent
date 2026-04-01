"""Report generation module."""

from .attention import SegmentationAwareAttention
from .medgemma import SegmentationGuidedReportGenerator
from .trainer import ReportGeneratorTrainer

__all__ = [
    "SegmentationAwareAttention",
    "SegmentationGuidedReportGenerator",
    "ReportGeneratorTrainer",
]
