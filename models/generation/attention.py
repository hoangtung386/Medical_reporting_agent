"""Segmentation-aware cross-attention mechanism.

This module implements the novel cross-attention layer that allows
the language model to attend to specific anatomical regions when
generating report descriptions.
"""

from __future__ import annotations


import torch
import torch.nn as nn


class SegmentationAwareAttention(nn.Module):
    """Cross-attention between LLM hidden states and segmentation features.

    Allows the model to "look at" specific anatomical regions when
    generating descriptions (e.g., attending to liver region when
    writing "liver: unremarkable").

    Args:
        llm_hidden_size: Hidden dimension of the language model.
        seg_feature_size: Dimension of segmentation features.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        llm_hidden_size: int = 2048,
        seg_feature_size: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = llm_hidden_size // num_heads

        # Project segmentation features to LLM dimension
        self.seg_projector = nn.Linear(seg_feature_size, llm_hidden_size)

        # Multi-head cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=llm_hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(llm_hidden_size)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(llm_hidden_size, llm_hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(llm_hidden_size * 4, llm_hidden_size),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(llm_hidden_size)

    def forward(
        self,
        llm_hidden_states: torch.Tensor,
        seg_features: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Apply cross-attention from LLM states to segmentation features.

        Args:
            llm_hidden_states: Shape ``[batch, seq_len, llm_hidden_size]``.
            seg_features: Shape ``[batch, num_regions, seg_feature_size]``.
            return_attention_weights: Whether to return attention map.

        Returns:
            Conditioned hidden states, and optionally attention weights.
        """
        # Project segmentation features to LLM space
        seg_projected = self.seg_projector(seg_features)

        # Cross-attention: Query from LLM, Key/Value from segmentation
        attn_output, attn_weights = self.cross_attn(
            query=llm_hidden_states,
            key=seg_projected,
            value=seg_projected,
            need_weights=return_attention_weights,
        )

        # Residual connection + norm
        hidden_states = self.layer_norm(llm_hidden_states + attn_output)

        # Feed-forward network
        ffn_output = self.ffn(hidden_states)
        output = self.ffn_norm(hidden_states + ffn_output)

        if return_attention_weights:
            return output, attn_weights
        return output
