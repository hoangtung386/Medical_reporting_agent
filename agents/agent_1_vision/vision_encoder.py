"""
Agent 1: 3D Vision Encoder (RadFM/M3D-LaMed style).
Extracts global visual features from volumetric data.
"""
from ..base import BaseAgent
import numpy as np
from typing import Any
try:
    import torch
    from monai.networks.nets import SwinTransformer
except ImportError:
    torch = None
    SwinTransformer = None

class VisionEncoderAgent(BaseAgent):
    """
    Agent 1: 3D Vision Encoder (RadFM/M3D-LaMed style).
    Extracts global visual features from volumetric data.
    """
    def __init__(self, device="cpu"):
        super().__init__(name="Agent 1: 3D Vision Encoder")
        # Determine device safely
        if torch and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
            
        if SwinTransformer:
            print(f"[{self.name}] Initializing SwinTransformer3D on {self.device}...")
            self.model = SwinTransformer(
                in_channels=1,
                embed_dim=128,
                window_size=(7, 7, 7),
                patch_size=(2, 2, 2),
                depths=[2, 2, 18, 2],
                num_heads=[4, 8, 16, 32],
                attn_drop_rate=0.0,
                drop_path_rate=0.0,
                spatial_dims=3,
            ).to(self.device)
        else:
            print(f"[{self.name}] WARNING: MONAI/Torch not installed. Using mock encoder.")
            self.model = None

    def forward(self, ct_volume: np.ndarray) -> Any:
        """
        Args:
            ct_volume: 3D numpy array [D, H, W] or [C, D, H, W]
        Returns:
            features: Extracted features.
        """
        print(f"[{self.name}] Processing volume shape: {ct_volume.shape}")
        
        if self.model and torch:
            # Real Forward Pass
            if isinstance(ct_volume, np.ndarray):
                tensor_vol = torch.from_numpy(ct_volume).float()
            else:
                tensor_vol = ct_volume.float()
                
            if tensor_vol.ndim == 3:
                 tensor_vol = tensor_vol.unsqueeze(0).unsqueeze(0) 
            elif tensor_vol.ndim == 4:
                 tensor_vol = tensor_vol.unsqueeze(0) 
                 
            tensor_vol = tensor_vol.to(self.device)
            
            with torch.no_grad():
                features_list = self.model(tensor_vol)
                final_features = features_list[-1]
            return final_features
        else:
            # Mock Fallback
            return np.random.rand(1, 512, 20, 20, 20)
