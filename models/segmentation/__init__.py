"""
Segmentation module initialization.
"""

from .swinunetr import SegmentationModel, SegmentationWrapper, ORGAN_LABELS

__all__ = ['SegmentationModel', 'SegmentationWrapper', 'ORGAN_LABELS']
