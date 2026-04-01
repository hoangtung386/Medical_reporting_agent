"""Simple CNN + LSTM baseline.

Most basic approach: 3D CNN feature extractor followed by LSTM decoder.
No pre-training, no segmentation. Pure end-to-end from scratch.
"""

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SimpleCNNLSTM(nn.Module):
    """Baseline 3: minimal 3D-CNN encoder with LSTM decoder.

    Args:
        vocab_size: Vocabulary size.
        hidden_dim: LSTM and embedding dimension.
    """

    def __init__(
        self,
        vocab_size: int = 10000,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        # Very simple 3D CNN
        self.cnn = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
        )

        # LSTM decoder
        self.fc_init = nn.Linear(64, hidden_dim)
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        ct_volume: torch.Tensor,
        max_length: int = 100,
    ) -> Dict[str, Any]:
        """Generate report tokens from raw CT volume.

        Args:
            ct_volume: Input volume ``[B, 1, D, H, W]``.
            max_length: Maximum generation length.

        Returns:
            Dict with ``generated_ids`` of shape ``[B, max_length]``.
        """
        batch_size = ct_volume.size(0)
        device = ct_volume.device

        features = self.cnn(ct_volume)
        h0 = self.fc_init(features).unsqueeze(0)
        c0 = torch.zeros_like(h0)

        input_token = torch.zeros(
            batch_size, 1, dtype=torch.long, device=device
        )
        hidden = (h0, c0)
        generated = []

        for _ in range(max_length):
            embed = self.embedding(input_token)
            out, hidden = self.lstm(embed, hidden)
            logits = self.fc_out(out)
            next_token = torch.argmax(logits, dim=-1)
            generated.append(next_token)
            input_token = next_token

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

    def generate_report(self, ct_volume: torch.Tensor) -> str:
        """Generate a report string from raw CT volume."""
        output = self.forward(ct_volume)
        return self.decode(output["generated_ids"])
