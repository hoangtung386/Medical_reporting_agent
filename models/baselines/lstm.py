"""LSTM-based report generator baseline.

Simple LSTM decoder that uses only global image features, without any
segmentation guidance. Demonstrates the value of anatomy-aware attention.
"""

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LSTMReportGenerator(nn.Module):
    """Baseline 1: LSTM report generator without segmentation guidance.

    Args:
        image_feature_dim: Dimension after image encoding.
        hidden_dim: LSTM hidden dimension.
        vocab_size: Vocabulary size.
        num_layers: Number of LSTM layers.
    """

    def __init__(
        self,
        image_feature_dim: int = 2048,
        hidden_dim: int = 512,
        vocab_size: int = 10000,
        num_layers: int = 2,
    ) -> None:
        super().__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # Image encoder (simple pooling + projection)
        self.image_encoder = nn.Sequential(
            nn.AdaptiveAvgPool3d((4, 4, 4)),
            nn.Flatten(),
            nn.Linear(768 * 64, image_feature_dim),
            nn.ReLU(),
        )

        # LSTM decoder
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        # Map image features to LSTM initial states
        self.feature_to_hidden = nn.Linear(
            image_feature_dim, hidden_dim * num_layers
        )
        self.feature_to_cell = nn.Linear(
            image_feature_dim, hidden_dim * num_layers
        )

        # Output projection
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        image_features: torch.Tensor,
        target_tokens: Optional[torch.Tensor] = None,
        max_length: int = 100,
    ) -> Dict[str, Any]:
        """Generate report tokens from image features.

        Args:
            image_features: Global features ``[B, C, D, H, W]``.
            target_tokens: Ground-truth tokens for teacher forcing.
            max_length: Maximum generation length.

        Returns:
            Dict with ``logits`` (training) or ``generated_ids`` (inference).
        """
        batch_size = image_features.size(0)
        img_feat = self.image_encoder(image_features)

        h0 = self.feature_to_hidden(img_feat).view(
            self.num_layers, batch_size, -1
        )
        c0 = self.feature_to_cell(img_feat).view(
            self.num_layers, batch_size, -1
        )

        if target_tokens is not None:
            # Teacher forcing
            token_embeds = self.embedding(target_tokens)
            lstm_out, _ = self.lstm(token_embeds, (h0, c0))
            logits = self.fc_out(lstm_out)
            return {"logits": logits}

        # Auto-regressive inference
        generated = []
        device = image_features.device
        input_token = torch.zeros(
            batch_size, 1, dtype=torch.long, device=device
        )
        hidden = (h0, c0)

        for _ in range(max_length):
            token_embed = self.embedding(input_token)
            lstm_out, hidden = self.lstm(token_embed, hidden)
            logits = self.fc_out(lstm_out)
            next_token = torch.argmax(logits, dim=-1)
            generated.append(next_token)
            input_token = next_token

        return {"generated_ids": torch.cat(generated, dim=1)}

    # ------------------------------------------------------------------
    # Convenience methods for evaluation compatibility
    # ------------------------------------------------------------------

    def decode(
        self,
        generated_ids: torch.Tensor,
        tokenizer: Optional[Any] = None,
    ) -> str:
        """Decode generated token IDs to text.

        Args:
            generated_ids: Token IDs ``[B, seq_len]``.
            tokenizer: Optional tokenizer for proper decoding.

        Returns:
            Decoded text string.
        """
        if tokenizer is not None:
            return tokenizer.decode(
                generated_ids[0], skip_special_tokens=True
            )
        return " ".join(str(t.item()) for t in generated_ids[0])

    def generate_report(self, image_features: torch.Tensor) -> str:
        """Generate a report string from image features.

        Args:
            image_features: Input tensor ``[B, C, D, H, W]``.

        Returns:
            Generated report text.
        """
        output = self.forward(image_features)
        return self.decode(output["generated_ids"])
