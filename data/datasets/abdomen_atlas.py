"""
Updated AbdomenAtlas Dataset Loader based on real data structure.

Dataset structure:
- CSV: AbdomenAtlas3.0MiniWithMeta.csv (9,262 samples)
- Images: data/image_only/{BDMAP_ID}/ct.nii.gz
- Masks: data/mask_only/{BDMAP_ID}/segmentations/*.nii.gz
- Reports: 4 types (structured, narrative, fusion structured, fusion narrative)
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
import warnings

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    warnings.warn("nibabel not installed. Install: pip install nibabel")


class AbdomenAtlasDataset(Dataset):
    """
    PyTorch Dataset for AbdomenAtlas3.0Mini (UPDATED for real structure).
    
    Args:
        csv_path: Path to AbdomenAtlas3.0MiniWithMeta.csv
        data_dir: Base directory containing image_only/ and mask_only/
        report_type: 'structured', 'narrative', 'fusion_structured', or 'fusion_narrative'
        load_images: If True, load CT volumes; else only metadata
        load_masks: If True, load segmentation masks
        transform: Optional transformation
        subset_ids: Optional list of BDMAP_IDs to use (for small samples)
    """
    
    def __init__(
        self,
        csv_path: str = "dataset_Abdomen_Atlas_3.0_mini_small/data/AbdomenAtlas3.0MiniWithMeta.csv",
        data_dir: str = "dataset_Abdomen_Atlas_3.0_mini_small/data",
        report_type: str = 'narrative',
        load_images: bool = True,
        load_masks: bool = False,
        transform: Optional[callable] = None,
        subset_ids: Optional[List[str]] = None
    ):
        super().__init__()
        
        self.data_dir = data_dir
        self.report_type = report_type
        self.load_images = load_images
        self.load_masks = load_masks
        self.transform = transform
        
        # Load metadata CSV
        print(f"Loading metadata from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        
        # Filter subset if provided
        if subset_ids:
            self.df = self.df[self.df['BDMAP ID'].isin(subset_ids)]
            print(f"Using subset: {len(subset_ids)} samples")
        
        print(f"Loaded {len(self.df)} samples")
        
        # Report column mapping
        self.report_columns = {
            'structured': 'structured report',
            'narrative': 'narrative report',
            'fusion_structured': 'fusion structured report',
            'fusion_narrative': 'fusion narrative report'
        }
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample."""
        row = self.df.iloc[idx]
        
        # Basic metadata
        bdmap_id = row['BDMAP ID']
        
        item = {
            'study_id': bdmap_id,
            'age': row.get('age', None),
            'sex': row.get('sex', None),
            'contrast': row.get('contrast', None),
        }
        
        # Extract report
        report_col = self.report_columns.get(self.report_type, 'narrative report')
        item['report'] = str(row.get(report_col, ''))
        
        # Tumor metadata from CSV
        item['tumor_info'] = {
            'liver': {
                'volume_cm3': row.get('liver volume (cm^3)', 0),
                'lesion_count': row.get('number of liver lesion instances', 0),
                'largest_diameter_cm': row.get('largest liver lesion diameter (cm)', 0),
                'largest_location': row.get('largest liver lesion location (1-8)', None),
            },
            'kidney': {
                'volume_cm3': row.get('kidney volume (cm^3)', 0),
                'lesion_count': row.get('number of kidney lesion instances', 0),
                'largest_diameter_cm': row.get('largest kidney lesion diameter (cm)', 0),
            },
            'pancreas': {
                'volume_cm3': row.get('pancreas volume (cm^3)', 0),
                'lesion_count': row.get('number of pancreatic lesion instances', 0),
                'largest_diameter_cm': row.get('largest pancreatic lesion diameter (cm)', 0),
            },
        }
        
        # Load CT image if requested
        if self.load_images:
            ct_path = os.path.join(self.data_dir, 'image_only', bdmap_id, 'ct.nii.gz')
            if os.path.exists(ct_path):
                ct_volume = self._load_nifti(ct_path)
                item['ct_volume'] = ct_volume
            else:
                warnings.warn(f"CT not found for {bdmap_id}")
                item['ct_volume'] = None
        
        # Load segmentation masks if requested
        if self.load_masks:
            mask_dir = os.path.join(self.data_dir, 'mask_only', bdmap_id, 'segmentations')
            if os.path.exists(mask_dir):
                masks = self._load_masks(mask_dir)
                item['masks'] = masks
            else:
                warnings.warn(f"Masks not found for {bdmap_id}")
                item['masks'] = None
        
        # Apply transform
        if self.transform:
            item = self.transform(item)
        
        return item
    
    def _load_nifti(self, path: str) -> torch.Tensor:
        """Load NIfTI file and convert to tensor."""
        if not NIBABEL_AVAILABLE:
            raise ImportError("nibabel required. Install: pip install nibabel")
        
        nii = nib.load(path)
        data = nii.get_fdata()
        
        # Convert to tensor and add channel dimension
        tensor = torch.from_numpy(data).float().unsqueeze(0)  # [1, D, H, W]
        
        return tensor
    
    def _load_masks(self, mask_dir: str) -> Dict[str, torch.Tensor]:
        """Load all segmentation masks from directory."""
        masks = {}
        
        if not os.path.exists(mask_dir):
            return masks
        
        for mask_file in os.listdir(mask_dir):
            if mask_file.endswith('.nii.gz'):
                mask_path = os.path.join(mask_dir, mask_file)
                mask_name = mask_file.replace('.nii.gz', '')
                
                try:
                    mask_data = self._load_nifti(mask_path)
                    masks[mask_name] = mask_data
                except Exception as e:
                    warnings.warn(f"Failed to load {mask_file}: {e}")
        
        return masks
    
    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """Custom collate function for DataLoader."""
        # Separate fields
        collated = {
            'study_id': [item['study_id'] for item in batch],
            'report': [item['report'] for item in batch],
            'tumor_info': [item['tumor_info'] for item in batch],
        }
        
        # Stack tensors if present
        if 'ct_volume' in batch[0] and batch[0]['ct_volume'] is not None:
            # Handle variable sizes - pad or crop as needed
            # For now, just stack (assumes same size)
            try:
                collated['ct_volume'] = torch.stack([item['ct_volume'] for item in batch])
            except:
                # If sizes differ, return as list
                collated['ct_volume'] = [item['ct_volume'] for item in batch]
        
        return collated


# Quick test
if __name__ == "__main__":
    print("Testing updated AbdomenAtlas dataset loader...")
    
    # Test with first 3 samples only
    dataset = AbdomenAtlasDataset(
        load_images=False,  # Skip images for quick test
        load_masks=False,
        subset_ids=['BDMAP_00000001', 'BDMAP_00000002', 'BDMAP_00000003']
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Get first sample
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Study ID: {sample['study_id']}")
    print(f"Age: {sample['age']}, Sex: {sample['sex']}")
    print(f"Report length: {len(sample['report'])} chars")
    print(f"Tumor info: {sample['tumor_info']['liver']}")
    
    print("\n✅ Dataset loader working!")
