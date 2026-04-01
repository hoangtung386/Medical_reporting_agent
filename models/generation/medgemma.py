"""Segmentation-guided medical report generator.

Core novelty: uses 3D segmentation features to guide report generation
through a cross-attention mechanism that grounds language in anatomy.

Based on MedGemma-2B with LoRA fine-tuning and custom attention layers.
"""

from __future__ import annotations


import logging
import math
import warnings
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .attention import SegmentationAwareAttention

logger = logging.getLogger(__name__)

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning(
        "transformers not installed. Running in mock mode. "
        "Install with: pip install transformers"
    )


class SegmentationGuidedReportGenerator(nn.Module):
    """Medical report generator guided by 3D segmentation features.

    Architecture:
        1. Base LLM: MedGemma-2B (medical domain pre-trained)
        2. Segmentation feature encoder: projects 3D features to sequence
        3. Cross-attention layers: fuse segmentation info into generation
        4. Optional RAG retrieval for clinical guidelines

    Training phases:
        - Phase 1: Frozen LLM, train projection layers only
        - Phase 2: LoRA fine-tuning of LLM layers

    Args:
        model_name: HuggingFace model identifier.
        seg_feature_size: Dimension of segmentation encoder features.
        use_lora: Whether to apply LoRA adapters.
        lora_r: LoRA rank.
        lora_alpha: LoRA scaling factor.
        num_cross_attn_layers: Number of cross-attention layers.
        pool_size: Spatial pooling target for segmentation features.
        num_seg_tokens: Number of tokens in the segmentation sequence.
        device: Target device.
    """

    def __init__(
        self,
        model_name: str = "google/medgemma-2b",
        seg_feature_size: int = 768,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        num_cross_attn_layers: int = 3,
        pool_size: tuple[int, int, int] = (4, 4, 4),
        num_seg_tokens: int = 8,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        super().__init__()

        self.device = device
        self.seg_feature_size = seg_feature_size
        self.pool_size = pool_size
        self.num_seg_tokens = num_seg_tokens
        pool_elements = math.prod(pool_size)

        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Running in mock mode (transformers unavailable).")
            self.llm = None
            self.tokenizer = None
            self.llm_hidden_size = 2048
        else:
            logger.info("Loading %s...", model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=(
                    torch.float16 if device == "cuda" else torch.float32
                ),
                device_map=device,
            )
            self.llm_hidden_size = self.llm.config.hidden_size

            if use_lora:
                self._apply_lora(lora_r, lora_alpha)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        # --- Segmentation feature processing ---
        self.seg_feature_pooling = nn.AdaptiveAvgPool3d(pool_size)
        self.seg_feature_flatten = nn.Flatten()
        self.seg_to_sequence = nn.Linear(
            seg_feature_size * pool_elements,
            self.llm_hidden_size * num_seg_tokens,
        )

        # --- Cross-attention layers (novel component) ---
        self.cross_attention_layers = nn.ModuleList(
            [
                SegmentationAwareAttention(
                    llm_hidden_size=self.llm_hidden_size,
                    seg_feature_size=self.llm_hidden_size,
                )
                for _ in range(num_cross_attn_layers)
            ]
        )

        # --- Measurement conditioning ---
        self.measurement_encoder = nn.Sequential(
            nn.Linear(25, 128),
            nn.ReLU(),
            nn.Linear(128, self.llm_hidden_size),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_lora(self, r: int, alpha: int) -> None:
        """Apply LoRA for parameter-efficient fine-tuning."""
        try:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=r,
                lora_alpha=alpha,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.llm = get_peft_model(self.llm, lora_config)
            logger.info("Applied LoRA (r=%d, alpha=%d)", r, alpha)
        except ImportError:
            logger.warning(
                "peft not installed. Running without LoRA. "
                "Install with: pip install peft"
            )

    # ------------------------------------------------------------------
    # Feature encoding
    # ------------------------------------------------------------------

    def encode_segmentation_features(
        self,
        seg_features: torch.Tensor,
    ) -> torch.Tensor:
        """Convert 3D segmentation features to a token sequence.

        Args:
            seg_features: Shape ``[batch, channels, D, H, W]``.

        Returns:
            Feature sequence of shape ``[batch, num_seg_tokens, hidden]``.
        """
        pooled = self.seg_feature_pooling(seg_features)
        flattened = self.seg_feature_flatten(pooled)
        seq = self.seg_to_sequence(flattened)
        return seq.reshape(
            seq.size(0), self.num_seg_tokens, self.llm_hidden_size
        )

    def encode_measurements(
        self,
        measurements: Dict[str, Any],
    ) -> torch.Tensor:
        """Encode quantitative measurements as continuous embeddings.

        Args:
            measurements: Dict with ``volumes_mm3`` for each organ.

        Returns:
            Measurement embedding of shape ``[1, hidden_size]``.
        """
        volumes = measurements.get("volumes_mm3", {})
        volume_tensor = torch.zeros(25, device=self.device)

        for organ_id, volume in volumes.items():
            if isinstance(organ_id, int) and organ_id < 25:
                volume_tensor[organ_id] = volume

        # Log-scale normalization
        volume_tensor = torch.log1p(volume_tensor)
        return self.measurement_encoder(volume_tensor.unsqueeze(0))

    # ------------------------------------------------------------------
    # Forward / generation
    # ------------------------------------------------------------------

    def forward(
        self,
        seg_features: torch.Tensor,
        measurements: Dict[str, Any],
        prompt: Optional[str] = None,
        max_length: int = 512,
        return_attention: bool = False,
    ) -> Dict[str, Any]:
        """Generate a medical report conditioned on segmentation.

        Args:
            seg_features: Encoder features from the segmentation model.
            measurements: Volume / bounding-box measurements.
            prompt: Optional text prompt.
            max_length: Maximum report length in tokens.
            return_attention: Whether to return attention weights.

        Returns:
            Dict with ``report`` string and optional ``attention_maps``.
        """
        if self.llm is None:
            return {
                "report": (
                    "MOCK REPORT: Liver unremarkable. "
                    "Lungs clear. No acute findings."
                ),
                "attention_maps": None,
            }

        # 1. Encode segmentation features
        seg_sequence = self.encode_segmentation_features(seg_features)

        # 2. Encode measurements
        measurement_emb = self.encode_measurements(measurements)

        # 3. Prepare text prompt
        if prompt is None:
            prompt = "Generate a radiology report based on the CT scan:\n"

        inputs = self.tokenizer(
            prompt, return_tensors="pt", padding=True
        ).to(self.device)

        # 4. Get LLM embeddings for text tokens
        input_embeds = self.llm.get_input_embeddings()(inputs["input_ids"])

        # 5. Prepend segmentation + measurement embeddings
        combined_embeds = torch.cat(
            [seg_sequence, measurement_emb.unsqueeze(1), input_embeds],
            dim=1,
        )

        # 6. Custom generation loop with cross-attention at each step.
        #    This is the core novelty: at every decoding step the LLM
        #    hidden states are conditioned on segmentation features via
        #    the cross-attention layers before predicting the next token.
        generated_ids: list[int] = []
        attention_maps: list[torch.Tensor] = []
        current_embeds = combined_embeds

        for _ in range(max_length):
            llm_outputs = self.llm(
                inputs_embeds=current_embeds,
                output_hidden_states=True,
            )
            hidden_states = llm_outputs.hidden_states[-1]

            # Apply cross-attention layers (novel component)
            for cross_attn in self.cross_attention_layers:
                if return_attention:
                    hidden_states, attn_w = cross_attn(
                        hidden_states,
                        seg_sequence,
                        return_attention_weights=True,
                    )
                    attention_maps.append(attn_w)
                else:
                    hidden_states = cross_attn(
                        hidden_states, seg_sequence
                    )

            # Predict next token from cross-attention-conditioned states
            next_logits = self.llm.lm_head(hidden_states[:, -1:, :])
            next_id = int(torch.argmax(next_logits, dim=-1).item())

            if next_id == self.tokenizer.eos_token_id:
                break

            generated_ids.append(next_id)

            # Prepare embedding for the next step
            next_emb = self.llm.get_input_embeddings()(
                torch.tensor([[next_id]], device=self.device)
            )
            current_embeds = next_emb  # only feed the new token

        report = self.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        )

        return {
            "report": report,
            "attention_maps": attention_maps if return_attention else None,
        }

    def generate_report(
        self,
        seg_output: Dict[str, Any],
        clinical_indication: Optional[str] = None,
        rag_context: Optional[str] = None,
    ) -> str:
        """High-level interface for report generation.

        Args:
            seg_output: Output from ``SegmentationModel.forward()``.
            clinical_indication: Optional clinical context.
            rag_context: Optional retrieved guidelines.

        Returns:
            Generated radiology report text.
        """
        seg_features = seg_output["features"]["bottleneck"]
        measurements = seg_output.get("measurements", {})

        # Build prompt
        prompt_parts: list[str] = []
        if clinical_indication:
            prompt_parts.append(
                f"Clinical indication: {clinical_indication}"
            )
        if rag_context:
            prompt_parts.append(f"Relevant guidelines: {rag_context}")
        prompt_parts.append("\nGenerate radiology report:\n")
        prompt = "\n".join(prompt_parts)

        output = self.forward(
            seg_features=seg_features,
            measurements=measurements,
            prompt=prompt,
        )
        return output["report"]
