"""Core models for medical report generation."""

from .generation.medgemma import SegmentationGuidedReportGenerator
from .segmentation.swinunetr import (
    ORGAN_LABELS,
    SegmentationModel,
    SegmentationWrapper,
)

__all__ = [
    "SegmentationModel",
    "SegmentationWrapper",
    "ORGAN_LABELS",
    "SegmentationGuidedReportGenerator",
]
