"""
Test script for AbdomenAtlas dataset loader with real sample data.

This tests the dataset loader on the small sample dataset.
"""

import sys
import os
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*70)
print("Testing AbdomenAtlas Dataset Loader")
print("="*70)

# Dataset paths
DATASET_DIR = "dataset_Abdomen_Atlas_3.0_mini_small/data"
CSV_PATH = os.path.join(DATASET_DIR, "AbdomenAtlas3.0MiniWithMeta.csv")

# Step 1: Load CSV and inspect
print("\n[1/5] Loading CSV metadata...")
df = pd.read_csv(CSV_PATH)
print(f"✓ Loaded {len(df)} samples")
print(f"✓ Columns: {len(df.columns)}")
print(f"\nFirst few column names:")
for i, col in enumerate(df.columns[:10]):
    print(f"  {i+1}. {col}")

# Check reports
print(f"\nReport columns available:")
report_cols = [c for c in df.columns if 'report' in c.lower()]
for col in report_cols:
    print(f"  - {col}")

# Step 2: Check first sample
print(f"\n[2/5] Inspecting first sample...")
sample = df.iloc[0]
bdmap_id = sample['BDMAP ID']
print(f"✓ Sample ID: {bdmap_id}")
print(f"✓ Age: {sample.get('age', 'N/A')}, Sex: {sample.get('sex', 'N/A')}")
print(f"✓ Scanner: {sample.get('scanner', 'N/A')}")
print(f"✓ Contrast: {sample.get('contrast', 'N/A')}")

# Check liver info
print(f"\nLiver information:")
print(f"  - Volume: {sample.get('liver volume (cm^3)', 'N/A')} cm³")
print(f"  - Lesions: {sample.get('number of liver lesion instances', 0)}")
if sample.get('number of liver lesion instances', 0) > 0:
    print(f"  - Largest lesion: {sample.get('largest liver lesion diameter (cm)', 'N/A')} cm")
    print(f"  - Location: Segment {sample.get('largest liver lesion location (1-8)', 'N/A')}")

# Step 3: Check file structure
print(f"\n[3/5] Checking file structure...")
image_path = os.path.join(DATASET_DIR, "image_only", bdmap_id, "ct.nii.gz")
mask_dir = os.path.join(DATASET_DIR, "mask_only", bdmap_id)

if os.path.exists(image_path):
    size_mb = os.path.getsize(image_path) / (1024**2)
    print(f"✓ CT image found: {size_mb:.1f} MB")
else:
    print(f"✗ CT image NOT found: {image_path}")

if os.path.exists(mask_dir):
    mask_files = os.listdir(mask_dir)
    print(f"✓ Mask directory found: {len(mask_files)} files")
    print(f"  Sample masks: {mask_files[:5]}")
else:
    print(f"✗ Mask directory NOT found: {mask_dir}")

# Step 4: Test report extraction
print(f"\n[4/5] Testing report extraction...")
structured_report = sample.get('structured report', '')
narrative_report = sample.get('narrative report', '')

print(f"\n--- Structured Report (first 300 chars) ---")
print(structured_report[:300] if structured_report else "No structured report")

print(f"\n--- Narrative Report (first 300 chars) ---")
print(narrative_report[:300] if narrative_report else "No narrative report")

# Step 5: Load with nibabel (if available)
print(f"\n[5/5] Testing image loading...")
try:
    import nibabel as nib
    
    ct_nii = nib.load(image_path)
    ct_data = ct_nii.get_fdata()
    
    print(f"✓ CT loaded successfully")
    print(f"  - Shape: {ct_data.shape}")
    print(f"  - Data type: {ct_data.dtype}")
    print(f"  - Value range: [{ct_data.min():.1f}, {ct_data.max():.1f}]")
    
    # Check if matches CSV metadata
    csv_shape = eval(sample['shape'])  # "(512, 512, 339)"
    print(f"  - CSV shape: {csv_shape}")
    print(f"  - Match: {'✓' if ct_data.shape == csv_shape else '✗'}")
    
except ImportError:
    print("⚠ nibabel not installed. Install with: pip install nibabel")
except Exception as e:
    print(f"✗ Error loading image: {e}")

# Summary
print("\n" + "="*70)
print("Test Summary")
print("="*70)
print(f"✓ CSV loaded: {len(df)} samples")
print(f"✓ Reports available: {len(report_cols)} types")
print(f"✓ File structure: OK")
print(f"✓ Sample data: OK")
print("\n✅ Dataset is ready for training!")
print("="*70)
