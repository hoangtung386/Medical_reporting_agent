"""Deterministic measurement utilities for medical images.

All functions are purely mathematical -- no AI / LLM involved.

Functions:
    calculate_volumes: Organ / lesion volumes from segmentation masks.
    get_bounding_boxes: 3D bounding-box extraction.
    calculate_hu_statistics: Hounsfield-unit statistics.
    measure_lesion_characteristics: Detailed lesion metrics.
    calculate_dice_score: Dice similarity coefficient.
    format_measurements_for_report: Human-readable formatting.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


def calculate_volumes(
    masks: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    organ_labels: Optional[Dict[int, str]] = None,
) -> Dict[str, float]:
    """Calculate volume for each structure in a segmentation mask.

    Args:
        masks: Integer label map ``[D, H, W]``.
        spacing: Voxel spacing in mm ``(depth, height, width)``.
        organ_labels: Optional mapping from label ID to organ name.

    Returns:
        Mapping of organ name to volume in mm^3.
    """
    volumes: Dict[str, float] = {}
    voxel_volume = float(np.prod(spacing))

    unique_labels = np.unique(masks)
    unique_labels = unique_labels[unique_labels > 0]

    for label in unique_labels:
        voxel_count = int(np.sum(masks == label))
        volume_mm3 = voxel_count * voxel_volume

        if organ_labels and label in organ_labels:
            organ_name = organ_labels[label]
        else:
            organ_name = f"label_{int(label)}"

        volumes[organ_name] = float(volume_mm3)

    return volumes


def get_bounding_boxes(
    masks: np.ndarray,
    organ_labels: Optional[Dict[int, str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Extract 3D bounding boxes for each segmented structure.

    Args:
        masks: Integer label map ``[D, H, W]``.
        organ_labels: Optional mapping from label ID to organ name.

    Returns:
        Mapping of organ name to bounding-box dict with ``min``,
        ``max``, and ``dimensions_mm`` keys.
    """
    bboxes: Dict[str, Dict[str, Any]] = {}
    unique_labels = np.unique(masks)
    unique_labels = unique_labels[unique_labels > 0]

    for label in unique_labels:
        coords = np.where(masks == label)
        if len(coords[0]) == 0:
            continue

        bbox = {
            "min": (
                int(coords[0].min()),
                int(coords[1].min()),
                int(coords[2].min()),
            ),
            "max": (
                int(coords[0].max()),
                int(coords[1].max()),
                int(coords[2].max()),
            ),
        }
        bbox["dimensions_mm"] = tuple(
            bbox["max"][i] - bbox["min"][i] + 1 for i in range(3)
        )

        if organ_labels and label in organ_labels:
            organ_name = organ_labels[label]
        else:
            organ_name = f"label_{int(label)}"

        bboxes[organ_name] = bbox

    return bboxes


def calculate_hu_statistics(
    ct_volume: np.ndarray,
    mask: np.ndarray,
    percentiles: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Calculate Hounsfield-unit statistics for a masked region.

    Args:
        ct_volume: CT scan ``[D, H, W]`` with HU values.
        mask: Binary mask ``[D, H, W]``.
        percentiles: Percentiles to compute. Defaults to
            ``[5, 25, 50, 75, 95]``.

    Returns:
        Dict with ``mean_hu``, ``std_hu``, ``min_hu``, ``max_hu``,
        and requested percentiles.
    """
    if percentiles is None:
        percentiles = [5, 25, 50, 75, 95]

    hu_values = ct_volume[mask > 0]
    if len(hu_values) == 0:
        return {}

    stats: Dict[str, float] = {
        "mean_hu": float(np.mean(hu_values)),
        "std_hu": float(np.std(hu_values)),
        "min_hu": float(np.min(hu_values)),
        "max_hu": float(np.max(hu_values)),
    }
    for p in percentiles:
        stats[f"p{p}_hu"] = float(np.percentile(hu_values, p))

    return stats


def measure_lesion_characteristics(
    mask: np.ndarray,
    ct_volume: Optional[np.ndarray] = None,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Dict[str, Any]:
    """Measure detailed characteristics of a lesion or nodule.

    Args:
        mask: Binary lesion mask ``[D, H, W]``.
        ct_volume: Optional CT data for HU analysis.
        spacing: Voxel spacing in mm.

    Returns:
        Dict with volume, diameter, sphericity, and optional HU info.
    """
    measurements: Dict[str, Any] = {}

    voxel_count = int(np.sum(mask > 0))
    voxel_volume = float(np.prod(spacing))
    measurements["volume_mm3"] = float(voxel_count * voxel_volume)
    measurements["volume_cm3"] = measurements["volume_mm3"] / 1000.0

    coords = np.where(mask > 0)
    if len(coords[0]) > 0:
        bbox_dims = [
            (coords[i].max() - coords[i].min() + 1) * spacing[i]
            for i in range(3)
        ]
        measurements["max_diameter_mm"] = float(max(bbox_dims))
        measurements["dimensions_mm"] = tuple(
            float(d) for d in bbox_dims
        )
        measurements["centroid"] = tuple(
            float(np.mean(coords[i])) for i in range(3)
        )

    # Sphericity
    if measurements["volume_mm3"] > 0:
        surface_voxels = ndimage.binary_erosion(mask) != mask
        surface_area = float(np.sum(surface_voxels)) * spacing[0] * spacing[1]
        sphere_radius = (
            3 * measurements["volume_mm3"] / (4 * np.pi)
        ) ** (1 / 3)
        sphere_surface = 4 * np.pi * sphere_radius**2

        measurements["sphericity"] = (
            float(sphere_surface / surface_area) if surface_area > 0 else 0.0
        )

    # HU statistics
    if ct_volume is not None:
        hu_stats = calculate_hu_statistics(ct_volume, mask)
        measurements["hu_statistics"] = hu_stats

        mean_hu = hu_stats.get("mean_hu", 0)
        if mean_hu < -500:
            measurements["density_type"] = "air"
        elif mean_hu < -100:
            measurements["density_type"] = "fat"
        elif mean_hu < 10:
            measurements["density_type"] = "water"
        elif mean_hu < 50:
            measurements["density_type"] = "soft_tissue"
        elif mean_hu < 400:
            measurements["density_type"] = "bone"
        else:
            measurements["density_type"] = "metal"

    return measurements


def calculate_dice_score(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    smooth: float = 1e-5,
) -> float:
    """Calculate Dice similarity coefficient.

    Args:
        pred_mask: Predicted segmentation ``[D, H, W]``.
        gt_mask: Ground-truth segmentation ``[D, H, W]``.
        smooth: Smoothing factor to avoid division by zero.

    Returns:
        Dice score in ``[0, 1]``.
    """
    pred_flat = (pred_mask > 0).flatten()
    gt_flat = (gt_mask > 0).flatten()

    intersection = np.sum(pred_flat & gt_flat)
    union = np.sum(pred_flat) + np.sum(gt_flat)

    return float((2.0 * intersection + smooth) / (union + smooth))


def format_measurements_for_report(
    measurements: Dict[str, Any],
) -> str:
    """Format measurements into human-readable report text.

    Args:
        measurements: Dict from ``calculate_volumes`` or
            ``measure_lesion_characteristics``.

    Returns:
        Formatted string suitable for a radiology report.
    """
    lines: list[str] = []

    if "volumes_mm3" in measurements:
        lines.append("Organ Volumes:")
        for organ, volume in measurements["volumes_mm3"].items():
            volume_cm3 = volume / 1000.0
            lines.append(f"  - {organ}: {volume_cm3:.1f} cm3")

    if "volume_cm3" in measurements:
        lines.append(
            f"Lesion volume: {measurements['volume_cm3']:.2f} cm3"
        )
    if "max_diameter_mm" in measurements:
        lines.append(
            f"Maximum diameter: {measurements['max_diameter_mm']:.1f} mm"
        )
    if "hu_statistics" in measurements:
        hu = measurements["hu_statistics"]
        lines.append(
            f"Attenuation: {hu['mean_hu']:.1f} +/- {hu['std_hu']:.1f} HU"
        )
    if "density_type" in measurements:
        lines.append(f"Density: {measurements['density_type']}")

    return "\n".join(lines)
