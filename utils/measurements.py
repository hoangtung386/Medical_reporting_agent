"""
Measurement utilities for medical images.

This module replaces Agent 6 (Measurement Quantifier).
All functions are DETERMINISTIC - no AI/LLM needed.

Functions:
    - calculate_volumes: Compute organ/lesion volumes from segmentation masks
    - get_bounding_boxes: Extract 3D bounding boxes
    - calculate_hu_statistics: Compute Hounsfield Unit statistics
    - measure_lesion_characteristics: Measure lesion-specific features
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from scipy import ndimage


def calculate_volumes(
    masks: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    organ_labels: Optional[Dict[int, str]] = None
) -> Dict[str, float]:
    """
    Calculate volume for each organ/structure in segmentation mask.
    
    Args:
        masks: Segmentation mask [D, H, W] with integer labels
        spacing: Voxel spacing in mm (depth, height, width)
        organ_labels: Optional mapping from label ID to organ name
    
    Returns:
        Dictionary mapping organ name to volume in mm³
    """
    volumes = {}
    voxel_volume = np.prod(spacing)  # Volume of single voxel in mm³
    
    unique_labels = np.unique(masks)
    unique_labels = unique_labels[unique_labels > 0]  # Exclude background
    
    for label in unique_labels:
        # Count voxels for this organ
        voxel_count = np.sum(masks == label)
        volume_mm3 = voxel_count * voxel_volume
        
        # Get organ name
        if organ_labels and label in organ_labels:
            organ_name = organ_labels[label]
        else:
            organ_name = f"label_{int(label)}"
        
        volumes[organ_name] = float(volume_mm3)
    
    return volumes


def get_bounding_boxes(
    masks: np.ndarray,
    organ_labels: Optional[Dict[int, str]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Extract 3D bounding boxes for each segmented structure.
    
    Args:
        masks: Segmentation mask [D, H, W]
        organ_labels: Optional mapping from label ID to organ name
    
    Returns:
        Dictionary with bounding box coordinates and dimensions
    """
    bboxes = {}
    unique_labels = np.unique(masks)
    unique_labels = unique_labels[unique_labels > 0]
    
    for label in unique_labels:
        # Find coordinates of this organ
        coords = np.where(masks == label)
        
        if len(coords[0]) == 0:
            continue
        
        # Get bounding box
        bbox = {
            'min': (int(coords[0].min()), int(coords[1].min()), int(coords[2].min())),
            'max': (int(coords[0].max()), int(coords[1].max()), int(coords[2].max())),
        }
        
        # Calculate dimensions
        bbox['dimensions_mm'] = tuple(
            bbox['max'][i] - bbox['min'][i] + 1 
            for i in range(3)
        )
        
        # Get organ name
        if organ_labels and label in organ_labels:
            organ_name = organ_labels[label]
        else:
            organ_name = f"label_{int(label)}"
        
        bboxes[organ_name] = bbox
    
    return bboxes


def calculate_hu_statistics(
    ct_volume: np.ndarray,
    mask: np.ndarray,
    percentiles: List[int] = [5, 25, 50, 75, 95]
) -> Dict[str, float]:
    """
    Calculate Hounsfield Unit (HU) statistics for a region.
    
    Args:
        ct_volume: CT scan [D, H, W] with HU values
        mask: Binary mask [D, H, W] for region of interest
        percentiles: Percentiles to compute
    
    Returns:
        Dictionary with HU statistics (mean, std, percentiles)
    """
    # Extract HU values in masked region
    hu_values = ct_volume[mask > 0]
    
    if len(hu_values) == 0:
        return {}
    
    stats = {
        'mean_hu': float(np.mean(hu_values)),
        'std_hu': float(np.std(hu_values)),
        'min_hu': float(np.min(hu_values)),
        'max_hu': float(np.max(hu_values)),
    }
    
    # Add percentiles
    for p in percentiles:
        stats[f'p{p}_hu'] = float(np.percentile(hu_values, p))
    
    return stats


def measure_lesion_characteristics(
    mask: np.ndarray,
    ct_volume: Optional[np.ndarray] = None,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> Dict[str, Any]:
    """
    Measure detailed characteristics of a lesion/nodule.
    
    Useful for lung nodules, tumors, etc.
    
    Args:
        mask: Binary mask of lesion [D, H, W]
        ct_volume: Optional CT data for HU analysis
        spacing: Voxel spacing
    
    Returns:
        Dictionary with lesion measurements
    """
    measurements = {}
    
    # Volume
    voxel_count = np.sum(mask > 0)
    voxel_volume = np.prod(spacing)
    measurements['volume_mm3'] = float(voxel_count * voxel_volume)
    measurements['volume_cm3'] = measurements['volume_mm3'] / 1000.0
    
    # Bounding box dimensions
    coords = np.where(mask > 0)
    if len(coords[0]) > 0:
        bbox_dims = [
            (coords[i].max() - coords[i].min() + 1) * spacing[i]
            for i in range(3)
        ]
        measurements['max_diameter_mm'] = float(max(bbox_dims))
        measurements['dimensions_mm'] = tuple(float(d) for d in bbox_dims)
        
        # Centroid
        centroid = [float(np.mean(coords[i])) for i in range(3)]
        measurements['centroid'] = tuple(centroid)
    
    # Sphericity (how round is the lesion?)
    if measurements['volume_mm3'] > 0:
        # Surface area approximation
        surface_voxels = ndimage.binary_erosion(mask) != mask
        surface_area = np.sum(surface_voxels) * spacing[0] * spacing[1]
        
        # Sphericity: ratio of sphere surface area to actual surface area
        sphere_radius = (3 * measurements['volume_mm3'] / (4 * np.pi)) ** (1/3)
        sphere_surface = 4 * np.pi * sphere_radius ** 2
        
        if surface_area > 0:
            measurements['sphericity'] = float(sphere_surface / surface_area)
        else:
            measurements['sphericity'] = 0.0
    
    # HU statistics if CT volume provided
    if ct_volume is not None:
        hu_stats = calculate_hu_statistics(ct_volume, mask)
        measurements['hu_statistics'] = hu_stats
        
        # Classify by HU (rough categorization)
        mean_hu = hu_stats.get('mean_hu', 0)
        if mean_hu < -500:
            measurements['density_type'] = 'air'
        elif mean_hu < -100:
            measurements['density_type'] = 'fat'
        elif mean_hu < 10:
            measurements['density_type'] = 'water'
        elif mean_hu < 50:
            measurements['density_type'] = 'soft_tissue'
        elif mean_hu < 400:
            measurements['density_type'] = 'bone'
        else:
            measurements['density_type'] = 'metal'
    
    return measurements


def calculate_dice_score(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    smooth: float = 1e-5
) -> float:
    """
    Calculate Dice similarity coefficient for evaluation.
    
    Args:
        pred_mask: Predicted segmentation [D, H, W]
        gt_mask: Ground truth segmentation [D, H, W]
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Dice score (0 to 1)
    """
    pred_flat = (pred_mask > 0).flatten()
    gt_flat = (gt_mask > 0).flatten()
    
    intersection = np.sum(pred_flat & gt_flat)
    union = np.sum(pred_flat) + np.sum(gt_flat)
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return float(dice)


def format_measurements_for_report(
    measurements: Dict[str, Any]
) -> str:
    """
    Format measurements into human-readable text for report.
    
    Args:
        measurements: Dictionary from calculate_volumes() or measure_lesion()
    
    Returns:
        Formatted string suitable for radiology report
    """
    lines = []
    
    # Volumes
    if 'volumes_mm3' in measurements:
        lines.append("Organ Volumes:")
        for organ, volume in measurements['volumes_mm3'].items():
            volume_cm3 = volume / 1000.0
            lines.append(f"  - {organ}: {volume_cm3:.1f} cm³")
    
    # Lesion characteristics
    if 'volume_cm3' in measurements:
        lines.append(f"Lesion volume: {measurements['volume_cm3']:.2f} cm³")
    
    if 'max_diameter_mm' in measurements:
        lines.append(f"Maximum diameter: {measurements['max_diameter_mm']:.1f} mm")
    
    if 'hu_statistics' in measurements:
        hu = measurements['hu_statistics']
        lines.append(f"Attenuation: {hu['mean_hu']:.1f} ± {hu['std_hu']:.1f} HU")
    
    if 'density_type' in measurements:
        lines.append(f"Density: {measurements['density_type']}")
    
    return "\n".join(lines)


# Example usage
if __name__ == "__main__":
    # Mock data
    fake_mask = np.zeros((100, 100, 100), dtype=np.uint8)
    fake_mask[30:50, 30:50, 30:50] = 1  # Liver
    fake_mask[60:70, 60:70, 60:70] = 2  # Kidney
    
    organ_map = {1: 'liver', 2: 'kidney'}
    
    # Calculate volumes
    volumes = calculate_volumes(fake_mask, spacing=(1.5, 1.0, 1.0), organ_labels=organ_map)
    print("Volumes:", volumes)
    
    # Get bounding boxes
    bboxes = get_bounding_boxes(fake_mask, organ_labels=organ_map)
    print("Bounding boxes:", bboxes)
    
    # Lesion characteristics
    lesion_mask = (fake_mask == 1).astype(np.uint8)
    lesion_info = measure_lesion_characteristics(lesion_mask, spacing=(1.5, 1.0, 1.0))
    print("Lesion info:", lesion_info)
