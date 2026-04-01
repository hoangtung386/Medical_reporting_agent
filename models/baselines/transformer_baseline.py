"""Transformer-based report generator baseline.

Standard Transformer decoder that uses global image features but no
segmentation guidance or anatomy-aware attention.
"""

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TransformerBaseline(nn.Module):
    """Baseline 2: Transformer decoder without anatomy-aware attention.

    Args:
        image_feature_dim: Dimension of image features.
        hidden_dim: Transformer hidden dimension.
        num_heads: Number of attention heads.
        num_layers: Number of decoder layers.
        vocab_size: Vocabulary size.
        max_seq_len: Maximum positional embedding length.
    """

    def __init__(
        self,
        image_feature_dim: int = 768,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        vocab_size: int = 10000,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # Image feature projection
        self.image_projector = nn.Sequential(
            nn.AdaptiveAvgPool3d((4, 4, 4)),
            nn.Flatten(),
            nn.Linear(image_feature_dim * 64, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.to_sequence = nn.Linear(hidden_dim * 4, hidden_dim)

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_seq_len, hidden_dim)
        )

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
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
        device = image_features.device

        img_proj = self.image_projector(image_features)
        memory = self.to_sequence(img_proj).unsqueeze(1)

        if target_tokens is not None:
            token_embeds = self.token_embedding(target_tokens)
            seq_len = token_embeds.size(1)
            token_embeds = token_embeds + self.pos_embedding[:, :seq_len, :]

            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                seq_len, device=device
            )
            output = self.transformer(
                tgt=token_embeds, memory=memory, tgt_mask=tgt_mask
            )
            logits = self.fc_out(output)
            return {"logits": logits}

        # Auto-regressive inference
        generated = []
        input_tokens = torch.zeros(
            batch_size, 1, dtype=torch.long, device=device
        )

        for step in range(max_length):
            token_embeds = self.token_embedding(input_tokens)
            token_embeds = (
                token_embeds + self.pos_embedding[:, : step + 1, :]
            )
            output = self.transformer(tgt=token_embeds, memory=memory)
            logits = self.fc_out(output[:, -1:, :])
            next_token = torch.argmax(logits, dim=-1)
            generated.append(next_token)
            input_tokens = torch.cat([input_tokens, next_token], dim=1)

        return {"generated_ids": torch.cat(generated, dim=1)}

    def decode(
        self,
        generated_ids: torch.Tensor,
        tokenizer: Optional[Any] = None,
    ) -> str:
        """Decode generated token IDs to text."""
        if tokenizer is not None:
            return tokenizer.decode(
                generated_ids[0], skip_special_tokens=True
            )
        return " ".join(str(t.item()) for t in generated_ids[0])

    def generate_report(self, image_features: torch.Tensor) -> str:
        """Generate a report string from image features."""
        output = self.forward(image_features)
        return self.decode(output["generated_ids"])
