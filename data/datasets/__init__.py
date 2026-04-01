"""Dataset loaders for medical report generation."""

from .abdomen_atlas import AbdomenAtlasDataset, collate_fn

__all__ = ["AbdomenAtlasDataset", "collate_fn"]
