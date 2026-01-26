"""
3D Segmentation Model using SwinUNETR architecture.
Combines Agent 1 (Vision Encoder) + Agent 2 (Segmentation Specialist).

This module outputs both segmentation masks AND multi-scale features
for downstream report generation.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple

try:
    from monai.networks.nets import SwinUNETR
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False
    print("Warning: MONAI not installed. Install with: pip install monai")


class SegmentationModel(nn.Module):
    """
    3D Medical Image Segmentation with SwinUNETR.
    
    Architecture:
        - Encoder: Swin Transformer backbone (extracts multi-scale features)
        - Decoder: UNet-style decoder with skip connections
        - Outputs: Segmentation masks + intermediate features for report generation
    
    Args:
        img_size: Input image size (D, H, W)
        in_channels: Number of input channels (1 for CT, 4 for MRI)
        out_channels: Number of segmentation classes (25 for AbdomenAtlas)
        feature_size: Base feature dimension
        use_checkpoint: Use gradient checkpointing to save memory
    """
    
    def __init__(
        self,
        img_size: Tuple[int, int, int] = (96, 96, 96),
        in_channels: int = 1,
        out_channels: int = 25,
        feature_size: int = 48,
        use_checkpoint: bool = True,
        pretrained_path: Optional[str] = None
    ):
        super().__init__()
        
        self.img_size = img_size
        self.out_channels = out_channels
        
        if not MONAI_AVAILABLE:
            print("WARNING: Running in mock mode. Install MONAI for real inference.")
            self.model = None
        else:
            self.model = SwinUNETR(
                img_size=img_size,
                in_channels=in_channels,
                out_channels=out_channels,
                feature_size=feature_size,
                use_checkpoint=use_checkpoint,
            )
            
            # Load pretrained weights (SuPreM)
            if pretrained_path:
                self._load_pretrained(pretrained_path)
    
    def _load_pretrained(self, path: str):
        """Load SuPreM pretrained weights"""
        try:
            checkpoint = torch.load(path, map_location='cpu')
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            self.model.load_state_dict(checkpoint, strict=False)
            print(f"[SegmentationModel] Loaded pretrained weights from {path}")
        except Exception as e:
            print(f"[SegmentationModel] Warning: Could not load pretrained weights: {e}")
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_features: bool = True
    ) -> Dict[str, Any]:
        """
        Forward pass with optional feature extraction.
        
        Args:
            x: Input tensor [B, C, D, H, W]
            return_features: If True, return intermediate features for report generation
        
        Returns:
            Dictionary containing:
                - 'logits': Segmentation logits [B, num_classes, D, H, W]
                - 'masks': Predicted segmentation masks [B, num_classes, D, H, W]
                - 'features': Multi-scale encoder features (if return_features=True)
                - 'attention_maps': Attention maps for visualization
        """
        
        if self.model is None:
            # Mock output for testing without MONAI
            B, C, D, H, W = x.shape
            return {
                'logits': torch.randn(B, self.out_channels, D, H, W),
                'masks': torch.randint(0, self.out_channels, (B, D, H, W)),
                'features': {
                    'encoder_hidden_states': torch.randn(B, 768, D//8, H//8, W//8),
                    'bottleneck': torch.randn(B, 768, D//32, H//32, W//32)
                },
                'attention_maps': None
            }
        
        # Real forward pass
        if return_features:
            # Extract intermediate features from encoder
            hidden_states_out = []
            
            def hook_fn(module, input, output):
                hidden_states_out.append(output)
            
            # Register hooks to capture encoder outputs
            hooks = []
            for i, layer in enumerate(self.model.swinViT.layers):
                hook = layer.register_forward_hook(hook_fn)
                hooks.append(hook)
            
            logits = self.model(x)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            # Organize features
            features = {
                'encoder_hidden_states': hidden_states_out,
                'bottleneck': hidden_states_out[-1] if hidden_states_out else None
            }
        else:
            logits = self.model(x)
            features = None
        
        # Get predicted masks
        masks = torch.argmax(logits, dim=1)
        
        return {
            'logits': logits,
            'masks': masks,
            'features': features,
            'attention_maps': None  # TODO: Extract from Swin attention
        }
    
    def compute_loss(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor,
        loss_type: str = 'dice_ce'
    ) -> torch.Tensor:
        """
        Compute segmentation loss.
        
        Args:
            predictions: Predicted logits [B, C, D, H, W]
            targets: Ground truth masks [B, D, H, W]
            loss_type: 'dice', 'ce', or 'dice_ce' (combined)
        """
        if loss_type == 'dice_ce':
            from monai.losses import DiceCELoss
            loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
            return loss_fn(predictions, targets)
        elif loss_type == 'dice':
            from monai.losses import DiceLoss
            loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
            return loss_fn(predictions, targets)
        elif loss_type == 'ce':
            loss_fn = nn.CrossEntropyLoss()
            return loss_fn(predictions, targets)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")


class SegmentationWrapper:
    """
    High-level wrapper for easy inference.
    Handles preprocessing, postprocessing, and metric calculation.
    """
    
    def __init__(
        self,
        model: SegmentationModel,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def predict(
        self, 
        ct_volume: torch.Tensor,
        return_features: bool = True
    ) -> Dict[str, Any]:
        """
        Run inference on a single CT volume.
        
        Args:
            ct_volume: Input CT [1, C, D, H, W] or [C, D, H, W]
            return_features: Return encoder features for report generation
        
        Returns:
            Dictionary with masks, features, and measurements
        """
        # Ensure batch dimension
        if ct_volume.ndim == 4:
            ct_volume = ct_volume.unsqueeze(0)
        
        ct_volume = ct_volume.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(ct_volume, return_features=return_features)
        
        # Move to CPU for postprocessing
        outputs['masks'] = outputs['masks'].cpu()
        outputs['logits'] = outputs['logits'].cpu()
        
        # Calculate measurements
        measurements = self._calculate_measurements(outputs['masks'])
        outputs['measurements'] = measurements
        
        return outputs
    
    def _calculate_measurements(self, masks: torch.Tensor) -> Dict[str, Any]:
        """
        Calculate volume and bounding box for each organ.
        This replaces Agent 6 (Measurement Quantifier).
        """
        from utils.measurements import calculate_volumes, get_bounding_boxes
        
        # Convert to numpy for easier processing
        masks_np = masks.squeeze(0).numpy()
        
        # Calculate volumes (deterministic, no AI needed)
        volumes = calculate_volumes(masks_np, spacing=(1.0, 1.0, 1.0))
        bboxes = get_bounding_boxes(masks_np)
        
        return {
            'volumes_mm3': volumes,
            'bounding_boxes': bboxes,
            'num_organs_detected': len(volumes)
        }


# Organ label mapping for AbdomenAtlas
ORGAN_LABELS = {
    1: 'spleen',
    2: 'right_kidney',
    3: 'left_kidney',
    4: 'gallbladder',
    5: 'liver',
    6: 'stomach',
    7: 'aorta',
    8: 'pancreas',
    9: 'right_adrenal_gland',
    10: 'left_adrenal_gland',
    11: 'duodenum',
    12: 'bladder',
    13: 'prostate',
    14: 'left_lung',
    15: 'right_lung',
    16: 'colon',
    17: 'liver_tumor',
    18: 'kidney_tumor',
    19: 'pancreas_tumor',
    20: 'hepatic_vessel',
    21: 'bone',
    22: 'esophagus',
    23: 'trachea',
    24: 'thyroid',
    25: 'muscle'
}
