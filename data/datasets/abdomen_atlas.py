"""AbdomenAtlas 3.0 Mini dataset loader.

Dataset structure::

    data/
        AbdomenAtlas3.0MiniWithMeta.csv   (9 262 samples)
        image_only/{BDMAP_ID}/ct.nii.gz
        mask_only/{BDMAP_ID}/segmentations/*.nii.gz

Supports four report types: *structured*, *narrative*,
*fusion_structured*, and *fusion_narrative*.
"""

import logging
import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

try:
    import nibabel as nib

    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    logger.warning("nibabel not installed. Install: pip install nibabel")


# Map short names to CSV column headers.
_REPORT_COLUMNS: Dict[str, str] = {
    "structured": "structured report",
    "narrative": "narrative report",
    "fusion_structured": "fusion structured report",
    "fusion_narrative": "fusion narrative report",
}


class AbdomenAtlasDataset(Dataset):
    """PyTorch dataset for AbdomenAtlas 3.0 Mini.

    Args:
        csv_path: Path to the metadata CSV.
        data_dir: Base directory containing ``image_only/`` and
            ``mask_only/``.
        report_type: One of ``structured``, ``narrative``,
            ``fusion_structured``, or ``fusion_narrative``.
        load_images: Whether to load CT volumes.
        load_masks: Whether to load segmentation masks.
        transform: Optional callable transform.
        subset_ids: Optional list of BDMAP IDs to restrict to.
    """

    def __init__(
        self,
        csv_path: str = (
            "dataset_Abdomen_Atlas_3.0_mini_small/"
            "data/AbdomenAtlas3.0MiniWithMeta.csv"
        ),
        data_dir: str = "dataset_Abdomen_Atlas_3.0_mini_small/data",
        report_type: str = "narrative",
        load_images: bool = True,
        load_masks: bool = False,
        transform: Optional[Callable] = None,
        subset_ids: Optional[List[str]] = None,
    ) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.report_type = report_type
        self.load_images = load_images
        self.load_masks = load_masks
        self.transform = transform

        logger.info("Loading metadata from %s ...", csv_path)
        self.df = pd.read_csv(csv_path)

        if subset_ids:
            self.df = self.df[self.df["BDMAP ID"].isin(subset_ids)]
            logger.info("Using subset: %d samples", len(subset_ids))

        logger.info("Loaded %d samples", len(self.df))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return a single sample."""
        row = self.df.iloc[idx]
        bdmap_id: str = row["BDMAP ID"]

        item: Dict[str, Any] = {
            "study_id": bdmap_id,
            "age": row.get("age"),
            "sex": row.get("sex"),
            "contrast": row.get("contrast"),
        }

        # Report text
        col = _REPORT_COLUMNS.get(self.report_type, "narrative report")
        item["report"] = str(row.get(col, ""))

        # Tumour metadata
        item["tumor_info"] = {
            "liver": {
                "volume_cm3": row.get("liver volume (cm^3)", 0),
                "lesion_count": row.get(
                    "number of liver lesion instances", 0
                ),
                "largest_diameter_cm": row.get(
                    "largest liver lesion diameter (cm)", 0
                ),
                "largest_location": row.get(
                    "largest liver lesion location (1-8)"
                ),
            },
            "kidney": {
                "volume_cm3": row.get("kidney volume (cm^3)", 0),
                "lesion_count": row.get(
                    "number of kidney lesion instances", 0
                ),
                "largest_diameter_cm": row.get(
                    "largest kidney lesion diameter (cm)", 0
                ),
            },
            "pancreas": {
                "volume_cm3": row.get("pancreas volume (cm^3)", 0),
                "lesion_count": row.get(
                    "number of pancreatic lesion instances", 0
                ),
                "largest_diameter_cm": row.get(
                    "largest pancreatic lesion diameter (cm)", 0
                ),
            },
        }

        if self.load_images:
            ct_path = os.path.join(
                self.data_dir, "image_only", bdmap_id, "ct.nii.gz"
            )
            if os.path.exists(ct_path):
                item["ct_volume"] = _load_nifti(ct_path)
            else:
                logger.warning("CT not found for %s", bdmap_id)
                item["ct_volume"] = None

        if self.load_masks:
            mask_dir = os.path.join(
                self.data_dir, "mask_only", bdmap_id, "segmentations"
            )
            if os.path.exists(mask_dir):
                item["masks"] = _load_masks(mask_dir)
            else:
                logger.warning("Masks not found for %s", bdmap_id)
                item["masks"] = None

        if self.transform:
            item = self.transform(item)

        return item


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _load_nifti(path: str) -> torch.Tensor:
    """Load a NIfTI file and return a float tensor ``[1, D, H, W]``."""
    if not NIBABEL_AVAILABLE:
        raise ImportError("nibabel required. Install: pip install nibabel")
    nii = nib.load(path)
    data = nii.get_fdata()
    return torch.from_numpy(data).float().unsqueeze(0)


def _load_masks(mask_dir: str) -> Dict[str, torch.Tensor]:
    """Load all segmentation masks from *mask_dir*."""
    masks: Dict[str, torch.Tensor] = {}
    for mask_file in os.listdir(mask_dir):
        if not mask_file.endswith(".nii.gz"):
            continue
        mask_path = os.path.join(mask_dir, mask_file)
        mask_name = mask_file.replace(".nii.gz", "")
        try:
            masks[mask_name] = _load_nifti(mask_path)
        except Exception as exc:
            logger.warning("Failed to load %s: %s", mask_file, exc)
    return masks


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate for ``DataLoader``.

    Stacks CT volumes when shapes match; falls back to a list
    otherwise.
    """
    collated: Dict[str, Any] = {
        "study_id": [item["study_id"] for item in batch],
        "report": [item["report"] for item in batch],
        "tumor_info": [item["tumor_info"] for item in batch],
    }

    if "ct_volume" in batch[0] and batch[0]["ct_volume"] is not None:
        volumes = [item["ct_volume"] for item in batch]
        shapes = {v.shape for v in volumes}
        if len(shapes) == 1:
            collated["ct_volume"] = torch.stack(volumes)
        else:
            logger.debug(
                "Variable CT sizes in batch; returning as list."
            )
            collated["ct_volume"] = volumes

    return collated
