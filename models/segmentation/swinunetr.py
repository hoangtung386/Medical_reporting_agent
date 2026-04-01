"""3D segmentation model using SwinUNETR architecture.

Outputs both segmentation masks and multi-scale encoder features
for downstream report generation.
"""

from __future__ import annotations


import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from utils.measurements import calculate_volumes, get_bounding_boxes

logger = logging.getLogger(__name__)

try:
    from monai.networks.nets import SwinUNETR

    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False
    logger.warning(
        "MONAI not installed. Running in mock mode. "
        "Install with: pip install monai"
    )


class SegmentationModel(nn.Module):
    """3D medical image segmentation with SwinUNETR.

    Architecture:
        - Encoder: Swin Transformer backbone (multi-scale features)
        - Decoder: UNet-style decoder with skip connections
        - Outputs: segmentation masks + intermediate features

    Args:
        img_size: Input volume size ``(D, H, W)``.
        in_channels: Number of input channels (1 for CT).
        out_channels: Number of segmentation classes.
        feature_size: Base feature dimension.
        use_checkpoint: Use gradient checkpointing to save memory.
        pretrained_path: Optional path to SuPreM pre-trained weights.
    """

    def __init__(
        self,
        img_size: Tuple[int, int, int] = (96, 96, 96),
        in_channels: int = 1,
        out_channels: int = 25,
        feature_size: int = 48,
        use_checkpoint: bool = True,
        pretrained_path: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.img_size = img_size
        self.out_channels = out_channels

        if not MONAI_AVAILABLE:
            logger.warning("Running in mock mode (MONAI unavailable).")
            self.model = None
        else:
            self.model = SwinUNETR(
                img_size=img_size,
                in_channels=in_channels,
                out_channels=out_channels,
                feature_size=feature_size,
                use_checkpoint=use_checkpoint,
            )
            if pretrained_path:
                self._load_pretrained(pretrained_path)

    def _load_pretrained(self, path: str) -> None:
        """Load SuPreM pre-trained weights."""
        try:
            checkpoint = torch.load(path, map_location="cpu")
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            self.model.load_state_dict(checkpoint, strict=False)
            logger.info("Loaded pre-trained weights from %s", path)
        except Exception as exc:
            logger.warning("Could not load pre-trained weights: %s", exc)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = True,
    ) -> Dict[str, Any]:
        """Forward pass with optional feature extraction.

        Args:
            x: Input tensor ``[B, C, D, H, W]``.
            return_features: Whether to capture encoder hidden states.

        Returns:
            Dict with ``logits``, ``masks``, ``features``, and
            ``attention_maps``.
        """
        if self.model is None:
            return self._mock_forward(x)

        features: Optional[Dict[str, Any]] = None

        if return_features:
            hidden_states: list[torch.Tensor] = []

            def _hook_fn(
                module: nn.Module,
                input: Any,
                output: Any,
            ) -> None:
                hidden_states.append(output)

            hooks = [
                layer.register_forward_hook(_hook_fn)
                for layer in self.model.swinViT.layers
            ]

            logits = self.model(x)

            for hook in hooks:
                hook.remove()

            features = {
                "encoder_hidden_states": hidden_states,
                "bottleneck": (
                    hidden_states[-1] if hidden_states else None
                ),
            }
        else:
            logits = self.model(x)

        masks = torch.argmax(logits, dim=1)

        return {
            "logits": logits,
            "masks": masks,
            "features": features,
            "attention_maps": None,
        }

    def _mock_forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """Produce random outputs when MONAI is unavailable."""
        b, _c, d, h, w = x.shape
        return {
            "logits": torch.randn(b, self.out_channels, d, h, w),
            "masks": torch.randint(0, self.out_channels, (b, d, h, w)),
            "features": {
                "encoder_hidden_states": torch.randn(
                    b, 768, d // 8, h // 8, w // 8
                ),
                "bottleneck": torch.randn(
                    b, 768, d // 32, h // 32, w // 32
                ),
            },
            "attention_maps": None,
        }

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        loss_type: str = "dice_ce",
    ) -> torch.Tensor:
        """Compute segmentation loss.

        Args:
            predictions: Logits ``[B, C, D, H, W]``.
            targets: Ground-truth masks ``[B, D, H, W]``.
            loss_type: One of ``dice``, ``ce``, or ``dice_ce``.

        Returns:
            Scalar loss tensor.

        Raises:
            ValueError: If *loss_type* is not recognised.
        """
        if loss_type == "dice_ce":
            from monai.losses import DiceCELoss

            loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
            return loss_fn(predictions, targets)
        if loss_type == "dice":
            from monai.losses import DiceLoss

            loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
            return loss_fn(predictions, targets)
        if loss_type == "ce":
            loss_fn = nn.CrossEntropyLoss()
            return loss_fn(predictions, targets)
        raise ValueError(f"Unknown loss type: {loss_type}")


class SegmentationWrapper:
    """High-level wrapper for segmentation inference.

    Handles pre-processing, post-processing, and measurement
    calculation.

    Args:
        model: A ``SegmentationModel`` instance.
        device: Target device string.
    """

    def __init__(
        self,
        model: SegmentationModel,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def predict(
        self,
        ct_volume: torch.Tensor,
        return_features: bool = True,
    ) -> Dict[str, Any]:
        """Run inference on a single CT volume.

        Args:
            ct_volume: Input ``[1, C, D, H, W]`` or ``[C, D, H, W]``.
            return_features: Return encoder features for report generation.

        Returns:
            Dict with masks, features, and measurements.
        """
        if ct_volume.ndim == 4:
            ct_volume = ct_volume.unsqueeze(0)

        ct_volume = ct_volume.to(self.device)

        with torch.no_grad():
            outputs = self.model(ct_volume, return_features=return_features)

        outputs["masks"] = outputs["masks"].cpu()
        outputs["logits"] = outputs["logits"].cpu()

        outputs["measurements"] = self._calculate_measurements(
            outputs["masks"]
        )
        return outputs

    @staticmethod
    def _calculate_measurements(
        masks: torch.Tensor,
    ) -> Dict[str, Any]:
        """Calculate volume and bounding box for each organ."""
        masks_np = masks.squeeze(0).numpy()
        volumes = calculate_volumes(masks_np, spacing=(1.0, 1.0, 1.0))
        bboxes = get_bounding_boxes(masks_np)

        return {
            "volumes_mm3": volumes,
            "bounding_boxes": bboxes,
            "num_organs_detected": len(volumes),
        }


# Organ label mapping for AbdomenAtlas.
# Labels 17-19 are tumour classes intentionally placed after the 16
# standard organ labels per the AbdomenAtlas annotation protocol.
ORGAN_LABELS: Dict[int, str] = {
    1: "spleen",
    2: "right_kidney",
    3: "left_kidney",
    4: "gallbladder",
    5: "liver",
    6: "stomach",
    7: "aorta",
    8: "pancreas",
    9: "right_adrenal_gland",
    10: "left_adrenal_gland",
    11: "duodenum",
    12: "bladder",
    13: "prostate",
    14: "left_lung",
    15: "right_lung",
    16: "colon",
    17: "liver_tumor",
    18: "kidney_tumor",
    19: "pancreas_tumor",
    20: "hepatic_vessel",
    21: "bone",
    22: "esophagus",
    23: "trachea",
    24: "thyroid",
    25: "muscle",
}
