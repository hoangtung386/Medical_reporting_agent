"""Training utilities for the segmentation-guided report generator."""

import logging
from typing import Any, Dict

import torch

from .medgemma import SegmentationGuidedReportGenerator

logger = logging.getLogger(__name__)


class ReportGeneratorTrainer:
    """Wrapper that handles optimisation for report generation training.

    Args:
        model: The report generator model.
        learning_rate: Learning rate for AdamW.
    """

    def __init__(
        self,
        model: SegmentationGuidedReportGenerator,
        learning_rate: float = 1e-4,
    ) -> None:
        self.model = model
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
        )

    def compute_loss(
        self,
        seg_features: torch.Tensor,
        measurements: Dict[str, Any],
        target_report: str,
    ) -> torch.Tensor:
        """Compute language-modelling loss for a single sample.

        The loss is computed only on the *text* portion of the combined
        embedding sequence (segmentation and measurement prefixes are
        masked with ``-100``).

        Args:
            seg_features: Segmentation encoder features.
            measurements: Quantitative measurements dict.
            target_report: Ground-truth report text.

        Returns:
            Scalar cross-entropy loss.
        """
        if self.model.llm is None:
            return torch.tensor(0.0, requires_grad=True)

        inputs = self.model.tokenizer(
            target_report,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            add_special_tokens=True,
        ).to(self.model.device)

        text_embeds = self.model.llm.get_input_embeddings()(
            inputs["input_ids"]
        )

        # Encode visual features
        seg_sequence = self.model.encode_segmentation_features(seg_features)
        measurement_emb = self.model.encode_measurements(measurements)
        measurement_emb = measurement_emb.unsqueeze(1)

        # Combine: [seg | measurement | text]
        combined_embeds = torch.cat(
            [seg_sequence, measurement_emb, text_embeds],
            dim=1,
        )

        # Labels: ignore prefix tokens with -100
        batch_size = combined_embeds.shape[0]
        prefix_len = seg_sequence.shape[1] + measurement_emb.shape[1]
        prefix_labels = torch.full(
            (batch_size, prefix_len), -100, device=self.model.device
        )
        labels = torch.cat([prefix_labels, inputs["input_ids"]], dim=1)

        outputs = self.model.llm(
            inputs_embeds=combined_embeds,
            labels=labels,
        )
        return outputs.loss
