import numpy as np
from typing import Dict, Any, Tuple

def calculate_metrics(masks: Dict[str, np.ndarray], spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Dict[str, Any]:
    """
    Calculate volumetric metrics from segmentation masks.
    
    Args:
        masks: Dictionary of organ_name -> binary mask (numpy array)
        spacing: Voxel spacing (x, y, z) in mm
        
    Returns:
        Dictionary containing volumes and bounding boxes
    """
    voxel_vol = np.prod(spacing)
    metrics = {
        
        "volumes": {},
        "bboxes": {}
    }
    
    for name, mask in masks.items():
        # Calculate Volume in mm3 -> cm3 (ml)
        vol_mm3 = np.sum(mask) * voxel_vol
        vol_cm3 = vol_mm3 / 1000.0
        metrics["volumes"][name] = round(vol_cm3, 2)
        
        # Calculate Bounding Box
        coords = np.argwhere(mask)
        if len(coords) > 0:
            min_coords = coords.min(axis=0).tolist()
            max_coords = coords.max(axis=0).tolist()
            metrics["bboxes"][name] = (min_coords, max_coords)
        else:
            metrics["bboxes"][name] = None
            
    return metrics
