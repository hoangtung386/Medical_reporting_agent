"""
Core models for medical report generation.
"""

from .segmentation.swinunetr import SegmentationModel
from .generation.medgemma import SegmentationGuidedReportGenerator

__all__ = [
    'SegmentationModel',
    'SegmentationGuidedReportGenerator'
]
