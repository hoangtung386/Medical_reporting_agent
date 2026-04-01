"""Tests for models.generation module (mock mode)."""

import torch

from models.generation import (
    SegmentationAwareAttention,
    SegmentationGuidedReportGenerator,
)


class TestSegmentationAwareAttention:
    """Tests for the cross-attention layer."""

    def test_output_shape(self):
        attn = SegmentationAwareAttention(
            llm_hidden_size=256,
            seg_feature_size=128,
            num_heads=4,
        )
        llm_states = torch.randn(2, 10, 256)
        seg_features = torch.randn(2, 5, 128)
        output = attn(llm_states, seg_features)
        assert output.shape == (2, 10, 256)

    def test_attention_weights_returned(self):
        attn = SegmentationAwareAttention(
            llm_hidden_size=256,
            seg_feature_size=128,
            num_heads=4,
        )
        llm_states = torch.randn(1, 8, 256)
        seg_features = torch.randn(1, 4, 128)
        output, weights = attn(
            llm_states, seg_features, return_attention_weights=True
        )
        assert output.shape == (1, 8, 256)
        assert weights is not None


class TestReportGeneratorMock:
    """Tests for SegmentationGuidedReportGenerator mock mode."""

    def test_mock_generate(self):
        gen = SegmentationGuidedReportGenerator(device="cpu")
        # In mock mode (no transformers), should return a mock report
        if gen.llm is None:
            seg_features = torch.randn(1, 768, 3, 3, 3)
            result = gen.forward(seg_features, measurements={})
            assert "report" in result
            assert isinstance(result["report"], str)
            assert len(result["report"]) > 0

    def test_encode_segmentation_features(self):
        gen = SegmentationGuidedReportGenerator(device="cpu")
        seg = torch.randn(2, 768, 8, 8, 8)
        seq = gen.encode_segmentation_features(seg)
        assert seq.shape == (2, gen.num_seg_tokens, gen.llm_hidden_size)

    def test_encode_measurements(self):
        gen = SegmentationGuidedReportGenerator(device="cpu")
        measurements = {"volumes_mm3": {0: 100.0, 1: 200.0}}
        emb = gen.encode_measurements(measurements)
        assert emb.shape == (1, gen.llm_hidden_size)
