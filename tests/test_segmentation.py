"""Tests for models.segmentation module (mock mode)."""

import torch

from models.segmentation import SegmentationModel, SegmentationWrapper


class TestSegmentationModel:
    """Tests for SegmentationModel in mock mode."""

    def test_mock_forward_shape(self):
        model = SegmentationModel(img_size=(96, 96, 96))
        x = torch.randn(1, 1, 96, 96, 96)
        output = model(x, return_features=True)
        assert "logits" in output
        assert "masks" in output
        assert "features" in output

    def test_mock_logits_shape(self):
        model = SegmentationModel(out_channels=25)
        x = torch.randn(2, 1, 96, 96, 96)
        output = model(x)
        assert output["logits"].shape[0] == 2
        assert output["logits"].shape[1] == 25

    def test_mock_masks_shape(self):
        model = SegmentationModel()
        x = torch.randn(1, 1, 96, 96, 96)
        output = model(x)
        assert output["masks"].shape == (1, 96, 96, 96)


class TestSegmentationWrapper:
    """Tests for SegmentationWrapper in mock mode."""

    def test_predict_returns_measurements(self):
        model = SegmentationModel()
        wrapper = SegmentationWrapper(model, device="cpu")
        x = torch.randn(1, 1, 96, 96, 96)
        output = wrapper.predict(x)
        assert "measurements" in output
        assert "volumes_mm3" in output["measurements"]
        assert "num_organs_detected" in output["measurements"]

    def test_predict_adds_batch_dim(self):
        model = SegmentationModel()
        wrapper = SegmentationWrapper(model, device="cpu")
        x = torch.randn(1, 96, 96, 96)  # no batch dim
        output = wrapper.predict(x)
        assert "masks" in output
