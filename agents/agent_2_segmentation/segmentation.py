"""
Agent 2: Segmentation Specialist (SuPreM + SAM-Med3D).
"""
from ..base import BaseAgent
from typing import Dict, Tuple, Any, Optional, Union, List
import numpy as np
import os
import sys

try:
    import torch
    import torch.nn.functional as F
    import SimpleITK as sitk
    from monai.networks.nets import SwinUNETR
    Tensor = torch.Tensor
except ImportError:
    torch = None
    sitk = None
    SwinUNETR = None
    Tensor = Any

class SegmentationSpecialistAgent(BaseAgent):
    """
    Hybrid pipeline: SuPreM (primary) + SAM-Med3D-turbo (refinement)
    """
    
    def __init__(self, device="cuda"):
        # Keep BaseAgent inheritance for system compatibility
        super().__init__(name="Agent 2: Segmentation Specialist")
        
        self.device = device if (torch and torch.cuda.is_available()) else "cpu"
        
        # Paths (System specific)
        self.weights_dir = os.path.join(os.path.dirname(__file__), "pretrained_weights")
        
        # Confidence threshold
        self.confidence_threshold = 0.85
        
        # Challenging organs (lower baseline performance)
        self.challenging_organs = [
            'pancreas', 'duodenum', 'celiac_trunk', 
            'hepatic_vessel', 'portal_vein_and_splenic_vein'
        ]

        if SwinUNETR:
            # Primary: Load SuPreM (Swin UNETR)
            print("Loading SuPreM (AbdomenAtlas 1.1)...")
            self.primary_model = self.load_suprem_model()
            
            # Secondary: Load SAM-Med3D-turbo (optional)
            print("Loading SAM-Med3D-turbo...")
            self.refinement_model = self.load_sam_med3d_turbo()
        else:
             print(f"[{self.name}] WARNING: Dependencies (monai, torch) not found. Running in mock mode.")
             self.primary_model = None
             self.refinement_model = None

    def load_suprem_model(self):
        """
        Load SuPreM pretrained weights
        Download from: https://github.com/MrGiovanni/SuPreM
        """
        try:
            model = SwinUNETR(
                img_size=(96, 96, 96),
                in_channels=1,
                out_channels=25,  # 25 organs
                feature_size=48,
                use_checkpoint=True
            ).to(self.device)
            
            # Load pretrained weights
            weight_path = os.path.join(self.weights_dir, "supervised_suprem_swinunetr_2100.pth")
            if os.path.exists(weight_path):
                checkpoint = torch.load(weight_path, map_location=self.device)
                
                # Handle possible dict structure
                state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
                model.load_state_dict(state_dict)
                model.eval()
                print(f"[{self.name}] SuPreM loaded successfully.")
                return model
            else:
                print(f"[{self.name}] WARNING: SuPreM weights not found at {weight_path}")
                return None
        except Exception as e:
            print(f"[{self.name}] Error loading SuPreM: {e}")
            return None
    
    def load_sam_med3d_turbo(self):
        """
        Load SAM-Med3D-turbo
        Download from: https://huggingface.co/blueyo0/SAM-Med3D
        """
        # This is pseudo-code - adapt based on SAM-Med3D API
        # from sam_med3d import SAMMed3D
        
        # model = SAMMed3D(
        #     checkpoint="ckpt/sam_med3d_turbo.pth",
        #     device=self.device
        # )
        
        # return model
        return None # Placeholder until user installs SAM-Med3D lib
    
    def segment_with_suprem(
        self, 
        volume: Tensor
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        """
        Primary segmentation với SuPreM
        
        Returns:
            masks: Dict {organ: mask}
            confidences: Dict {organ: confidence_score}
        """
        if self.primary_model is None:
            return {}, {}

        with torch.no_grad():
            # NOTE: For real implementation with large volumes, we need Sliding Window Inference.
            # But following user's structure:
            # inputs = volume if volume.ndim == 5 else volume.unsqueeze(0)
            
            # To avoid shape errors with direct model call on mismatched size, 
            # we check size or use monai sliding window if available.
            # Providing simple fallback for this snippet context.
            
            if volume.shape[-1] != 96: # If input is not 96x96x96
                 from monai.inferers import sliding_window_inference
                 logits = sliding_window_inference(
                    inputs=volume, 
                    roi_size=(96, 96, 96), 
                    sw_batch_size=4, 
                    predictor=self.primary_model,
                    overlap=0.5
                )
            else:
                logits = self.primary_model(volume)  # [B, 25, D, H, W]
            
            # Softmax + argmax
            probs = torch.softmax(logits, dim=1)
            pred_masks = torch.argmax(probs, dim=1)  # [B, D, H, W]
            
            # Convert to per-organ masks
            masks = {}
            confidences = {}
            
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
                # Extract mask for this organ
                organ_mask = (pred_masks[batch_idx] == label_id).cpu().numpy()
                
                # Calculate confidence (mean probability for this organ)
                organ_probs = probs[batch_idx, label_id, ...].cpu().numpy()
                confidence = float(organ_probs[organ_mask].mean()) if organ_mask.any() else 0.0
                
                if organ_mask.any():  # Only save if organ detected
                    masks[organ_name] = organ_mask
                    confidences[organ_name] = confidence
        
        return masks, confidences
    
    def refine_with_sam_med3d(
        self,
        volume: Tensor,
        coarse_mask: np.ndarray,
        organ_name: str
    ) -> np.ndarray:
        """
        Refine mask với SAM-Med3D-turbo
        Use coarse mask as prompt
        """
        if self.refinement_model is None:
            return coarse_mask

        # Generate prompts from coarse mask
        # Strategy: Use centroid + boundary points
        coords = np.argwhere(coarse_mask > 0.5)
        
        if len(coords) == 0:
            return coarse_mask
        
        # Centroid
        centroid = coords.mean(axis=0).astype(int)
        
        # Boundary points (optional: add more points for complex shapes)
        # For now, just use centroid + bbox
        bbox_min = coords.min(axis=0)
        bbox_max = coords.max(axis=0)
        
        # SAM-Med3D inference
        # (Pseudo-code - adapt based on actual API)
        # refined_mask = self.refinement_model.predict(
        #     image=volume,
        #     point_prompts=[centroid],
        #     box_prompt=[bbox_min, bbox_max]
        # )
        
        return coarse_mask # return defined_mask when implemented
    
    def preprocess(self, volume: np.ndarray) -> np.ndarray:
        """Standardize preprocessing"""
        # Window
        volume = np.clip(volume, -125, 275)
        
        # Normalize
        volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
        
        return volume
    
    def calculate_measurements(
        self, 
        masks: Dict[str, np.ndarray],
        spacing=(1.0, 1.0, 1.0)
    ) -> Dict[str, Dict]:
        """
        Deterministic measurements
        """
        metrics = {}
        voxel_vol = spacing[0] * spacing[1] * spacing[2]
        
        for name, mask in masks.items():
            vol_mm3 = float(np.sum(mask > 0) * voxel_vol)
            metrics[name] = {"volume_mm3": vol_mm3, "volume_ml": vol_mm3 / 1000.0}
            
        return metrics

    def forward(
        self,
        ct_input: Union[str, np.ndarray],
        vision_features: Any = None, # kept for signature compatibility with main.py
        use_refinement: bool = True
    ) -> Dict[str, Any]:
        """
        Main pipeline
        """
        print(f"🔄 Processing Input...")
        
        # Handle Input: Path vs Numpy
        if isinstance(ct_input, str):
            if sitk:
                img = sitk.ReadImage(ct_input)
                volume = sitk.GetArrayFromImage(img)
            else:
                print("SimpleITK not installed, cannot read file path.")
                return {}
        else:
             volume = ct_input

        # Mock fallback if models not loaded
        if self.primary_model is None:
             print("MODELS NOT LOADED. RETURNING MOCK.")
             return {
                "masks": {"liver": np.zeros_like(volume)},
                "metrics": self.calculate_measurements({"liver": np.zeros_like(volume)}),
                "status": "Mocked"
             }

        # Preprocess
        volume = self.preprocess(volume)
        volume_tensor = torch.from_numpy(volume).float()
        
        if volume_tensor.ndim == 3:
             volume_tensor = volume_tensor.unsqueeze(0).unsqueeze(0)
        elif volume_tensor.ndim == 4:
             volume_tensor = volume_tensor.unsqueeze(0)
             
        volume_tensor = volume_tensor.to(self.device)
        
        # Step 1: Primary segmentation
        print("  → Step 1: Primary segmentation (SuPreM)...")
        masks, confidences = self.segment_with_suprem(volume_tensor)
        
        # Step 2: Optional refinement
        final_masks = {}
        metadata = {}
        
        for organ, mask in masks.items():
            confidence = confidences[organ]
            needs_refinement = (
                use_refinement and (
                    confidence < self.confidence_threshold or
                    organ in self.challenging_organs
                )
            )
            
            if needs_refinement and self.refinement_model:
                print(f"  → Step 2: Refining {organ} (confidence={confidence:.2f})...")
                refined_mask = self.refine_with_sam_med3d(
                    volume_tensor, mask, organ
                )
                final_masks[organ] = refined_mask
                
                metadata[organ] = {
                    'confidence': confidence,
                    'refined': True,
                    'method': 'SuPreM + SAM-Med3D-turbo'
                }
            else:
                final_masks[organ] = mask
                metadata[organ] = {
                    'confidence': confidence,
                    'refined': False,
                    'method': 'SuPreM only'
                }
        
        print(f"✅ Segmented {len(final_masks)} organs")
        
        # Calculate measurements
        # Assuming isotropic 1x1x1 spacing if passing raw numpy, or use metadata if available.
        measurements = self.calculate_measurements(final_masks)
        
        # Return Dict compatible with main.py
        return {
            "masks": final_masks,
            "metrics": measurements, 
            "metadata": metadata
        }


        