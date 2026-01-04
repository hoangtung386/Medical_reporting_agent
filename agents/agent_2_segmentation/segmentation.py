"""
Agent 2: Segmentation Specialist (SuPreM + SAM-Med3D).
"""
from ..base import BaseAgent
from typing import Dict, Tuple, Any, Optional
import numpy as np
import os
import sys

# Ensure we can import from the cloned repos if needed
# (Optional: user can install them as packages, but simple path addition helps if just cloned)
current_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(current_dir, "repos", "SAM-Med3D"))

try:
    import torch
    import torch.nn.functional as F
    import SimpleITK as sitk
    from monai.networks.nets import SwinUNETR
except ImportError:
    torch = None
    sitk = None
    SwinUNETR = None

class SegmentationSpecialistAgent(BaseAgent):
    """
    Agent 2: Segmentation Specialist (Hybrid: SuPreM + SAM-Med3D).
    
    Strategy:
    1. Primary: SuPreM (SwinUNETR) for fast, automatic segmentation of 25 organs.
    2. Refinement: SAM-Med3D for challenging organs or low-confidence predictions.
    """
    def __init__(self, device="cuda"):
        super().__init__(name="Agent 2: Segmentation Specialist")
        
        # Determine device
        if torch and torch.cuda.is_available() and device == "cuda":
            self.device = "cuda"
        else:
            self.device = "cpu"
            
        print(f"[{self.name}] Initializing on {self.device}...")

        self.primary_model = None
        self.refinement_model = None
        
        # Paths
        self.weights_dir = os.path.join(os.path.dirname(__file__), "pretrained_weights")
        self.repos_dir = os.path.join(os.path.dirname(__file__), "repos")
        
        # Configuration
        self.confidence_threshold = 0.85
        self.challenging_organs = [
            'pancreas', 'duodenum', 'celiac_trunk', 
            'hepatic_vessel', 'portal_vein_and_splenic_vein'
        ]

        if SwinUNETR:
            self.load_models()
        else:
            print(f"[{self.name}] WARNING: Dependencies (monai, torch) not found. Running in mock mode.")

    def load_models(self):
        """Load SuPreM and optionally SAM-Med3D."""
        # 1. Load SuPreM
        try:
            print(f"[{self.name}] Loading SuPreM (AbdomenAtlas 1.1)...")
            self.primary_model = SwinUNETR(
                img_size=(96, 96, 96),
                in_channels=1,
                out_channels=25,
                feature_size=48,
                use_checkpoint=True
            ).to(self.device)
            
            suprem_path = os.path.join(self.weights_dir, "supervised_suprem_swinunetr_2100.pth")
            if os.path.exists(suprem_path):
                checkpoint = torch.load(suprem_path, map_location=self.device)
                state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
                self.primary_model.load_state_dict(state_dict)
                self.primary_model.eval()
                print(f"[{self.name}] SuPreM loaded successfully.")
            else:
                print(f"[{self.name}] WARNING: SuPreM weights not found at {suprem_path}")
                self.primary_model = None
        except Exception as e:
            print(f"[{self.name}] Error loading SuPreM: {e}")
            self.primary_model = None

        # 2. Load SAM-Med3D (Lazy load or load here)
        # For now, we'll keep it as a placeholder until the user sets it up specifically
        # as it requires the specific SAM-Med3D codebase.
        self.refinement_model = None 

    def load_sam_med3d_turbo(self):
        """
        Load SAM-Med3D-turbo.
        Requires SAM-Med3D to be installed or in python path.
        """
        try:
            # This is pseudo-code/dependent on SAM-Med3D library structure
            # from sam_med3d import SAMMed3D 
            # self.refinement_model = SAMMed3D(checkpoint=..., device=self.device)
            print(f"[{self.name}] SAM-Med3D loading not yet fully implemented (requires library installation).")
        except Exception as e:
            print(f"[{self.name}] Error loading SAM-Med3D: {e}")

    def preprocess_volume(self, volume_np: np.ndarray) -> torch.Tensor:
        """Standardize preprocessing: Clip and Normalize."""
        # Window [-125, 275] for soft tissues/abdomen usually
        # But SuPreM paper might use different ranges. Defaulting to standard abdomen window.
        vol_clipped = np.clip(volume_np, -125, 275)
        # Normalize 0-1
        vol_norm = (vol_clipped - vol_clipped.min()) / (vol_clipped.max() - vol_clipped.min() + 1e-8)
        
        # To Tensor [B, C, D, H, W]
        # Assuming input is [D, H, W]
        tensor = torch.from_numpy(vol_norm).float()
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.ndim == 4:
            tensor = tensor.unsqueeze(0)
            
        return tensor.to(self.device)

    def segment_with_suprem(self, tensor_vol: torch.Tensor) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        """Run primary segmentation."""
        if self.primary_model is None:
            return {}, {}
            
        with torch.no_grad():
            # Sliding window inference might be needed for large volumes, 
            # but for simplicity/snippet, we assume resize or patch-based.
            # SuPreM takes 96x96x96 patches usually. 
            # Ideally usage: monai.inferers.sliding_window_inference
            from monai.inferers import sliding_window_inference
            
            output_logits = sliding_window_inference(
                inputs=tensor_vol,
                roi_size=(96, 96, 96),
                sw_batch_size=4,
                predictor=self.primary_model,
                overlap=0.5
            )
            
            probs = torch.softmax(output_logits, dim=1)
            pred_masks = torch.argmax(probs, dim=1) # [B, D, H, W]
            
            masks = {}
            confidences = {}
            
            # Mapping from SuPreM paper/repo
            organ_labels = {
                1: 'aorta', 2: 'gall_bladder', 3: 'kidney_left',
                4: 'kidney_right', 5: 'liver', 6: 'pancreas',
                7: 'postcava', 8: 'spleen', 9: 'stomach',
                10: 'adrenal_gland_left', 11: 'adrenal_gland_right',
                12: 'bladder', 13: 'celiac_trunk', 14: 'colon',
                15: 'duodenum', 16: 'esophagus', 17: 'femur_left',
                18: 'femur_right', 19: 'hepatic_vessel', 20: 'intestine',
                21: 'lung_left', 22: 'lung_right', 
                23: 'portal_vein_and_splenic_vein',
                24: 'prostate', 25: 'rectum'
            }
            
            batch_idx = 0
            for label_id, organ_name in organ_labels.items():
                organ_mask = (pred_masks[batch_idx] == label_id).cpu().numpy()
                
                # Mean probability of the predicted class
                organ_probs = probs[batch_idx, label_id, ...].cpu().numpy()
                if organ_mask.any():
                    confidence = float(organ_probs[organ_mask].mean())
                    masks[organ_name] = organ_mask
                    confidences[organ_name] = confidence
                else:
                    confidences[organ_name] = 0.0
                    
            return masks, confidences

    def refine_with_sam(self, volume_tensor: torch.Tensor, coarse_mask: np.ndarray, organ_name: str) -> np.ndarray:
        """
        Refine mask with SAM-Med3D-turbo.
        (Placeholder for logic).
        """
        # If model not satisfied, return coarse
        if self.refinement_model is None:
            return coarse_mask
            
        # Logic to generate prompts and predict
        return coarse_mask

    def forward(self, ct_volume: np.ndarray, use_refinement: bool = True) -> Dict[str, Any]:
        """
        Main pipeline.
        
        Args:
            ct_volume: Numpy array [D, H, W]
            use_refinement: Whether to use SAM for challenging/low-conf organs.
        """
        print(f"[{self.name}] Starting segmentation...")
        
        if self.primary_model is None and torch is not None:
            # Try reloading if missing (e.g. first run)
            self.load_models()
        
        if self.primary_model is None:
            print(f"[{self.name}] Model not loaded. Returning mock data.")
            return {
                "masks": {"liver": np.zeros_like(ct_volume)}, 
                "metadata": {}, 
                "status": "Failed (Mocked)"
            }

        # Preprocess
        tensor_vol = self.preprocess_volume(ct_volume)
        
        # Step 1: SuPreM
        masks, confidences = self.segment_with_suprem(tensor_vol)
        
        final_masks = {}
        metadata = {}
        
        # Step 2: Refinement decision
        for organ, mask in masks.items():
            confidence = confidences.get(organ, 0.0)
            needs_refinement = (
                use_refinement and (
                    confidence < self.confidence_threshold or
                    organ in self.challenging_organs
                )
            )
            
            if needs_refinement and self.refinement_model:
                print(f"[{self.name}] Refining {organ} (conf={confidence:.2f})...")
                refined_mask = self.refine_with_sam(tensor_vol, mask, organ)
                final_masks[organ] = refined_mask
                metadata[organ] = {'confidence': confidence, 'refined': True}
            else:
                final_masks[organ] = mask
                metadata[organ] = {'confidence': confidence, 'refined': False}
                
        return {
            "masks": final_masks,
            "metadata": metadata,
            "status": "Success"
        }

        