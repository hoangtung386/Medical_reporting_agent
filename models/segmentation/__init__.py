"""Segmentation module."""

from .swinunetr import ORGAN_LABELS, SegmentationModel, SegmentationWrapper

__all__ = ["SegmentationModel", "SegmentationWrapper", "ORGAN_LABELS"]
