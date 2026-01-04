"""
Agent 1: 3D Vision Encoder (SuPreM/SwinUNETR).
Extracts global visual features from volumetric data using a pre-trained SwinUNETR backbone.
"""
from ..base import BaseAgent
import numpy as np
from typing import Any
import os

try:
    import torch
    from monai.networks.nets import SwinUNETR
except ImportError:
    torch = None
    SwinUNETR = None

class VisionEncoderAgent(BaseAgent):
    """
    Agent 1: 3D Vision Encoder.
    Uses a pre-trained SwinUNETR (SuPreM) model to extract features from 3D CT volumes.
    """
    def __init__(self, device="cpu"):
        super().__init__(name="Agent 1: 3D Vision Encoder")
        
        # Determine device
        if torch and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = device

        if SwinUNETR:
            print(f"[{self.name}] Initializing SwinUNETR on {self.device}...")
            # Configuration matches user request:
            # img_size=(96, 96, 96), in_channels=1, out_channels=25, feature_size=48
            self.model = SwinUNETR(
                img_size=(96, 96, 96),
                in_channels=1,
                out_channels=25,
                feature_size=48
            ).to(self.device)
            
            # Load pre-trained weights
            weights_path = os.path.join(os.path.dirname(__file__), '../agent_2_segmentation/pretrained_weights/supervised_suprem_swinunetr_2100.pth')
            if os.path.exists(weights_path):
                print(f"[{self.name}] Loading weights from {weights_path}...")
                try:
                    checkpoint = torch.load(weights_path, map_location=self.device)
                    # Handle if checkpoint is a dict with 'state_dict' key or the state_dict itself
                    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
                    self.model.load_state_dict(state_dict)
                    print(f"[{self.name}] Weights loaded successfully.")
                except Exception as e:
                    print(f"[{self.name}] ERROR loading weights: {e}")
            else:
                print(f"[{self.name}] WARNING: Weights file not found at {weights_path}")
            
            self.model.eval()
        else:
            print(f"[{self.name}] WARNING: MONAI/Torch not installed. Using mock encoder.")
            self.model = None

    def forward(self, ct_volume: np.ndarray) -> Any:
        """
        Extract features from the input volume.
        
        Args:
            ct_volume: 3D numpy array [D, H, W] or [C, D, H, W]
        Returns:
            features: List of multi-scale features or raw embeddings from the encoder.
        """
        # print(f"[{self.name}] Processing volume shape: {ct_volume.shape}")
        
        if self.model and torch:
            # Prepare Input Tensor
            if isinstance(ct_volume, np.ndarray):
                tensor_vol = torch.from_numpy(ct_volume).float()
            else:
                tensor_vol = ct_volume.float()
                
            # Ensure shape is [B, C, D, H, W]
            if tensor_vol.ndim == 3: # [D, H, W]
                 tensor_vol = tensor_vol.unsqueeze(0).unsqueeze(0) 
            elif tensor_vol.ndim == 4: # [C, D, H, W] or [B, D, H, W] - Assume C if channel-first is standard, but usually single volume is input.
                 # If first dim is small (e.g. 1), likely C.
                 tensor_vol = tensor_vol.unsqueeze(0) 
                 
            tensor_vol = tensor_vol.to(self.device)
            
            with torch.no_grad():
                # User requested: "Output multi-scale features; dùng model.encoder để lấy raw embeddings"
                # SwinUNETR's encoder is 'swinViT'. 
                # calling self.model(x) returns segmentation logits. 
                # calling self.model.swinViT(x_in) returns hidden_states_out (list of features)
                
                # Check if we should use the encoder directly for features
                features = self.model.swinViT(tensor_vol, normalize=True) 
                
                # features is a list of outputs from different stages of Swin Transformer
                # If subsequent agents expect a single embedding, we might need to process this.
                # For now, returning the raw multi-scale features as requested.
                return features
        else:
            # Mock Fallback
            return [np.random.rand(1, 48 * (2**i), 96 // (2**(i+1)), 96 // (2**(i+1)), 96 // (2**(i+1))) for i in range(4)]
