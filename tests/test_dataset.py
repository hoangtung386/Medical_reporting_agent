"""Tests for data.datasets.abdomen_atlas module.

These tests verify the dataset loader logic without requiring the
actual AbdomenAtlas data files to be present.
"""

import pytest
import torch

from data.datasets.abdomen_atlas import collate_fn


class TestCollate:
    """Tests for the custom collate function."""

    def test_collate_basic(self):
        batch = [
            {
                "study_id": "BDMAP_001",
                "report": "liver normal",
                "tumor_info": {"liver": {}},
                "ct_volume": torch.randn(1, 32, 32, 32),
            },
            {
                "study_id": "BDMAP_002",
                "report": "kidney normal",
                "tumor_info": {"kidney": {}},
                "ct_volume": torch.randn(1, 32, 32, 32),
            },
        ]
        result = collate_fn(batch)
        assert len(result["study_id"]) == 2
        assert len(result["report"]) == 2
        assert result["ct_volume"].shape[0] == 2

    def test_collate_variable_sizes(self):
        batch = [
            {
                "study_id": "A",
                "report": "a",
                "tumor_info": {},
                "ct_volume": torch.randn(1, 32, 32, 32),
            },
            {
                "study_id": "B",
                "report": "b",
                "tumor_info": {},
                "ct_volume": torch.randn(1, 64, 64, 64),
            },
        ]
        result = collate_fn(batch)
        # Variable sizes should fall back to list
        assert isinstance(result["ct_volume"], list)

    def test_collate_no_images(self):
        batch = [
            {"study_id": "A", "report": "a", "tumor_info": {}},
            {"study_id": "B", "report": "b", "tumor_info": {}},
        ]
        result = collate_fn(batch)
        assert "ct_volume" not in result
