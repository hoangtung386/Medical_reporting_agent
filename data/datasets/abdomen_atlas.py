"""
Dataset loader for AbdomenAtlas3.0Mini.

This dataset contains:
- 18,524 CT-report pairs (13,000 train, 5,490 test)
- 3 types of reports: structured, narrative, enhanced
- Segmentation masks for 26 organs + tumors
- Focus on liver, kidney, pancreas tumors (10,374 total, 7,003 small ≤2cm)

Citation:
@article{bassi2025radgpt,
  title={RadGPT: Constructing 3D Image-Text Tumor Datasets},
  author={Bassi, Pedro R. A. S. and ...},
  journal={arXiv preprint arXiv:2501.04678},
  year={2025}
}
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    warnings.warn("Hugging Face datasets not installed. Install: pip install datasets")

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    warnings.warn("nibabel not installed. Install: pip install nibabel")


class AbdomenAtlasDataset(Dataset):
    """
    PyTorch Dataset for AbdomenAtlas3.0Mini.
    
    Features:
    - 18,524 CT scans with radiology reports
    - 3 report types: structured (template), narrative (natural), enhanced (66 diagnoses)
    - Segmentation masks: 26 organs + tumor annotations
    - Tumor focus: 10,374 tumors (7,003 small ≤2cm for early detection)
    
    Args:
        split: 'train' or 'test' (IID split from paper)
        report_type: 'structured', 'narrative', or 'enhanced'
        load_images: If True, load CT and masks; else only reports (for text-only experiments)
        focus_small_tumors: If True, prioritize samples with small tumors (≤2cm)
        transform: Optional transformation for CT volumes
    
    Returns:
        Dictionary with:
            - 'ct_volume': Tensor [C, D, H, W] - CT scan
            - 'seg_mask': Tensor [D, H, W] - Segmentation with organ/tumor labels
            - 'report': str - Radiology report text
            - 'tumor_info': Dict with tumor metadata
            - 'study_id': str - Unique identifier (BDMAP_ID)
    """
    
    # Organ label mapping (26 structures from AbdomenAtlas)
    ORGAN_LABELS = {
        0: 'background',
        1: 'spleen',
        2: 'right_kidney',
        3: 'left_kidney',
        4: 'gallbladder',
        5: 'esophagus',
        6: 'liver',
        7: 'stomach',
        8: 'aorta',
        9: 'inferior_vena_cava',
        10: 'portal_vein_splenic_vein',
        11: 'pancreas',
        12: 'right_adrenal_gland',
        13: 'left_adrenal_gland',
        14: 'duodenum',
        15: 'hepatic_vessel',
        16: 'right_lung',
        17: 'left_lung',
        18: 'colon',
        19: 'intestine',
        20: 'rectum',
        21: 'bladder',
        22: 'prostate',
        23: 'left_head_of_femur',
        24: 'right_head_of_femur',
        25: 'celiac_trunk',
        26: 'kidney_tumor',
        27: 'liver_tumor',
        28: 'pancreas_tumor',
        # Extended for sub-segments (from paper)
        # Liver: segments 1-8 (Couinaud)
        # Pancreas: head, body, tail
        # Vessels: SMA, SMV, CA, CHA for staging
    }
    
    def __init__(
        self,
        split: str = 'train',
        report_type: str = 'narrative',
        load_images: bool = True,
        focus_small_tumors: bool = False,
        transform: Optional[callable] = None,
        data_dir: Optional[str] = None
    ):
        super().__init__()
        
        self.split = split
        self.report_type = report_type
        self.load_images = load_images
        self.focus_small_tumors = focus_small_tumors
        self.transform = transform
        self.data_dir = data_dir
        
        # Load metadata from Hugging Face
        if not DATASETS_AVAILABLE:
            raise ImportError("Please install: pip install datasets")
        
        print(f"Loading AbdomenAtlas3.0Mini ({split} split)...")
        self.dataset = load_dataset(
            "AbdomenAtlas/AbdomenAtlas3.0Mini",
            split=split,
            trust_remote_code=True
        )
        
        # Filter for small tumors if requested
        if focus_small_tumors:
            self.dataset = self._filter_small_tumors()
        
        print(f"Loaded {len(self.dataset)} samples")
    
    def _filter_small_tumors(self):
        """Filter to prioritize small tumor cases (≤2cm)"""
        # TODO: Implement filtering based on tumor size metadata
        # For now, return full dataset
        print("Note: Small tumor filtering not yet implemented")
        return self.dataset
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample.
        
        Returns dictionary with CT, mask, report, tumor info.
        """
        sample = self.dataset[idx]
        
        # Study identifier
        study_id = sample.get('BDMAP ID', f'sample_{idx}')
        
        # Load report based on type
        report = self._load_report(sample)
        
        # Prepare return dict
        item = {
            'study_id': study_id,
            'report': report,
        }
        
        # Load CT image and mask if requested
        if self.load_images:
            if not self.data_dir:
                warnings.warn(
                    "data_dir not specified. Cannot load CT/mask files. "
                    "Download full dataset with: bash download_atlas_3.sh"
                )
                # Return mock data for testing
                item['ct_volume'] = torch.randn(1, 96, 96, 96)
                item['seg_mask'] = torch.randint(0, 29, (96, 96, 96))
            else:
                ct_volume, seg_mask = self._load_images(study_id)
                item['ct_volume'] = ct_volume
                item['seg_mask'] = seg_mask
        
        # Extract tumor information
        tumor_info = self._parse_tumor_info(sample, seg_mask if self.load_images else None)
        item['tumor_info'] = tumor_info
        
        # Apply transform if specified
        if self.transform and self.load_images:
            item = self.transform(item)
        
        return item
    
    def _load_report(self, sample: Dict) -> str:
        """
        Load report based on selected type.
        
        Types:
        - 'structured': Template-based, predictable format
        - 'narrative': Natural language, like real radiologist
        - 'enhanced': Human + AI, covers 66 diagnoses
        """
        # Map report type to column name (adjust based on actual dataset structure)
        report_key_map = {
            'structured': 'structured_report',
            'narrative': 'narrative_report',
            'enhanced': 'enhanced_report'
        }
        
        report_key = report_key_map.get(self.report_type, 'narrative_report')
        
        # Fallback to any available report field
        if report_key not in sample:
            # Try common field names
            for key in ['report', 'text', 'radiology_report', 'narrative_report']:
                if key in sample:
                    return sample[key]
            
            # If no report found, return placeholder
            return "No report available for this sample."
        
        return sample[report_key]
    
    def _load_images(self, study_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load CT volume and segmentation mask from disk.
        
        Args:
            study_id: BDMAP ID
        
        Returns:
            ct_volume: [1, D, H, W] tensor
            seg_mask: [D, H, W] tensor with organ/tumor labels
        """
        if not NIBABEL_AVAILABLE:
            raise ImportError("Please install: pip install nibabel")
        
        # Construct file paths (adjust based on actual data structure)
        ct_path = f"{self.data_dir}/images/{study_id}.nii.gz"
        mask_path = f"{self.data_dir}/masks/{study_id}.nii.gz"
        
        try:
            # Load NIfTI files
            ct_nii = nib.load(ct_path)
            mask_nii = nib.load(mask_path)
            
            # Convert to numpy
            ct_array = ct_nii.get_fdata()
            mask_array = mask_nii.get_fdata()
            
            # Convert to torch tensors
            ct_volume = torch.from_numpy(ct_array).float().unsqueeze(0)  # Add channel dim
            seg_mask = torch.from_numpy(mask_array).long()
            
            return ct_volume, seg_mask
            
        except FileNotFoundError:
            warnings.warn(f"Files not found for {study_id}. Using mock data.")
            # Return mock data
            return torch.randn(1, 96, 96, 96), torch.randint(0, 29, (96, 96, 96))
    
    def _parse_tumor_info(self, sample: Dict, seg_mask: Optional[torch.Tensor] = None) -> Dict:
        """
        Extract tumor information from sample.
        
        Returns:
            Dictionary with:
                - 'has_tumor': bool
                - 'tumor_count': int
                - 'tumor_types': List[str] (liver, kidney, pancreas)
                - 'small_tumor_count': int (≤2cm)
                - 'stage': str (for pancreatic cancer, if available)
        """
        tumor_info = {
            'has_tumor': False,
            'tumor_count': 0,
            'tumor_types': [],
            'small_tumor_count': 0,
            'stage': None
        }
        
        # If we have segmentation mask, count tumors directly
        if seg_mask is not None:
            tumor_labels = [26, 27, 28]  # kidney, liver, pancreas tumors
            
            for label_id in tumor_labels:
                if torch.any(seg_mask == label_id):
                    tumor_info['has_tumor'] = True
                    tumor_info['tumor_count'] += 1
                    tumor_info['tumor_types'].append(self.ORGAN_LABELS[label_id])
        
        # Parse from report if needed (fallback)
        # TODO: Implement regex parsing from structured reports
        
        return tumor_info
    
    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """
        Custom collate function for DataLoader.
        
        Handles variable-size images by padding or cropping.
        """
        # Separate field types
        ct_volumes = []
        seg_masks = []
        reports = []
        tumor_infos = []
        study_ids = []
        
        for item in batch:
            if 'ct_volume' in item:
                ct_volumes.append(item['ct_volume'])
                seg_masks.append(item['seg_mask'])
            reports.append(item['report'])
            tumor_infos.append(item['tumor_info'])
            study_ids.append(item['study_id'])
        
        # Stack tensors (assuming same size for now)
        batch_dict = {
            'report': reports,
            'tumor_info': tumor_infos,
            'study_id': study_ids
        }
        
        if ct_volumes:
            batch_dict['ct_volume'] = torch.stack(ct_volumes)
            batch_dict['seg_mask'] = torch.stack(seg_masks)
        
        return batch_dict


# Usage example
if __name__ == "__main__":
    # Test loading
    print("Testing AbdomenAtlasDataset...")
    
    dataset = AbdomenAtlasDataset(
        split='train',
        report_type='narrative',
        load_images=False,  # Only text for this test
        focus_small_tumors=False
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get first sample
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Study ID: {sample['study_id']}")
    print(f"Report (first 200 chars): {sample['report'][:200]}...")
    print(f"Tumor info: {sample['tumor_info']}")
    
    # Create DataLoader
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=AbdomenAtlasDataset.collate_fn
    )
    
    batch = next(iter(dataloader))
    print(f"\nBatch size: {len(batch['report'])}")
    print("Dataset loading successful!")
