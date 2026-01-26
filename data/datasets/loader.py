import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from typing import Dict, Any, Optional

class AbdomenAtlasDataset(Dataset):
    """
    Dataset loader for AbdomenAtlas 3.0 Mini.
    
    Expected Structure:
        root_dir/
          data/
            AbdomenAtlas3.0MiniWithMeta.csv
            image_only/
              BDMAP_00000001/
                ct.nii.gz
            mask_only/ (optional)
              BDMAP_00000001/
                segmentation.nii.gz
    """
    
    def __init__(
        self,
        root_dir: str,
        csv_file: str = "data/AbdomenAtlas3.0MiniWithMeta.csv", 
        split: str = "train",
        transform = None,
        max_length: int = 512,
        cache_data: bool = False
    ):
        """
        Args:
            root_dir: Path to dataset_Abdomen_Atlas_3.0_mini_small
            csv_file: Relative path to CSV from root_dir
            split: 'train', 'val', or 'test' (will simple random split if column not present)
            transform: MONAI transforms
        """
        self.root_dir = root_dir
        self.csv_path = os.path.join(root_dir, csv_file)
        self.transform = transform
        
        # Load metadata
        try:
            # The CSV might be messy, let's try reading it carefully
            self.df = pd.read_csv(self.csv_path)
            print(f"Loaded dataset from {self.csv_path}: {len(self.df)} cases found.")
        except Exception as e:
            print(f"Error loading CSV: {e}")
            self.df = pd.DataFrame()

        # Filter by existing files ON DISK first
        # This is crucial for the 'mini' dataset which only has a few folders but full CSV
        image_dir = os.path.join(self.root_dir, "data", "image_only")
        available_ids = set()
        if os.path.exists(image_dir):
            for name in os.listdir(image_dir):
                if os.path.isdir(os.path.join(image_dir, name)):
                    available_ids.add(name)
        
        print(f"Found {len(available_ids)} folders in {image_dir}")
        
        # Intersect with CSV IDs
        csv_ids = set(self.df['id'].tolist()) if 'id' in self.df.columns else set()
        valid_ids = sorted(list(available_ids.intersection(csv_ids)))
        
        print(f"Intersecting with CSV: {len(valid_ids)} valid cases available.")
        
        if len(valid_ids) == 0:
            print("WARNING: No valid intersection found. Checking first folder name vs CSV ID...")
            if len(available_ids) > 0:
                print(f"First disk folder: {list(available_ids)[0]}")
            if len(csv_ids) > 0:
                print(f"First CSV ID: {list(csv_ids)[0]}")
        
        # Split logic
        total = len(valid_ids)
        if total == 0:
            self.case_ids = []
        else:
            # For very small datasets, ensure at least 1 in val if possible
            if total < 5:
                # Too small to split properly, put all in train, duplicate for val?
                if split == 'train':
                    self.case_ids = valid_ids
                else: 
                     self.case_ids = valid_ids # Overlap for debugging
            else:
                train_end = int(total * 0.8)
                if train_end == 0 and total > 0: train_end = 1
                
                if split == 'train':
                    self.case_ids = valid_ids[:train_end]
                elif split == 'val':
                     # If val is empty due to small size, take last one
                    val_ids = valid_ids[train_end:]
                    if len(val_ids) == 0: val_ids = valid_ids[-1:]
                    self.case_ids = val_ids
                else:
                    self.case_ids = valid_ids[train_end:] # Test = Val for now

        print(f"[{split}] Final sample count: {len(self.case_ids)}")

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        
        # Paths
        ct_path = os.path.join(self.root_dir, "data", "image_only", case_id, "ct.nii.gz")
        
        # Load Report (Last column usually, or find the text column)
        # Using the column name from the CSV inspection 'CT Arterial Phase ...' is actually inside the data?
        # Let's inspect the dataframe row.
        row = self.df[self.df['id'] == case_id].iloc[0]
        
        # Heuristic to find the report text (longest string column)
        # The user's CSV snippet shows the report at the end
        report_text = ""
        for col in self.df.columns:
            val = str(row[col])
            if len(val) > 50 and "FINDINGS" in val: # Simple heuristic
                report_text = val
                break
        
        if not report_text:
            # Fallback if specific column known
            if 'report' in self.df.columns:
                report_text = row['report']
            else:
                # Use the last column as valid fallback often
                report_text = str(row.iloc[-1])

        # Load Image
        try:
            # Use Nibabel to load NIfTI
            ct_nifti = nib.load(ct_path)
            ct_data = ct_nifti.get_fdata().astype(np.float32)
            
            # Simple normalization if no transform provided
            if self.transform is None:
                # Clip to abdominal window [-150, 250]
                ct_data = np.clip(ct_data, -150, 250)
                ct_data = (ct_data - np.min(ct_data)) / (np.max(ct_data) - np.min(ct_data) + 1e-6)
                
                # Resize to fixed size (e.g. 96x96x96) for batching simplicity in this demo
                # Real implementation should use MONAI transforms
                import torch.nn.functional as F
                ct_tensor = torch.tensor(ct_data).unsqueeze(0).unsqueeze(0) # B,C,D,H,W
                ct_tensor = F.interpolate(ct_tensor, size=(96,96,96), mode='trilinear')
                ct_data = ct_tensor.squeeze().numpy()

            # To Tensor
            ct_tensor = torch.tensor(ct_data).unsqueeze(0) # Add channel dim: [1, D, H, W]

        except Exception as e:
            print(f"Error loading {ct_path}: {e}")
            ct_tensor = torch.zeros((1, 96, 96, 96))
            report_text = "Error loading case."

        return {
            "id": case_id,
            "ct_volume": ct_tensor,
            "report": report_text
        }
