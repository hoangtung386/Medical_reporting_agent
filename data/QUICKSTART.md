# Quick Start Guide: AbdomenAtlas3.0Mini

## Getting Started

### Step 1: Download Dataset

```bash
# Option A: Use the included download script (~500 GB full)
bash download_data.sh

# Option B: Clone RadGPT repository (alternative)
cd data/
git clone https://github.com/MrGiovanni/RadGPT
cd RadGPT
bash download_atlas_3.sh
```

### Step 2: Verify Dataset

```python
# Test loading from Hugging Face (metadata only)
from datasets import load_dataset

dataset = load_dataset("AbdomenAtlas/AbdomenAtlas3.0Mini", split="train")
print(f"Loaded {len(dataset)} samples")
print(dataset[0].keys())
```

### Step 3: Load with Our DataLoader

```python
from data.datasets import AbdomenAtlasDataset, collate_fn
from torch.utils.data import DataLoader

# Create dataset
dataset = AbdomenAtlasDataset(
    csv_path="data/AbdomenAtlas3.0MiniWithMeta.csv",
    data_dir="data",
    report_type="narrative",   # or "structured", "fusion_structured", "fusion_narrative"
    load_images=True,
    load_masks=False,          # set True to load segmentation masks
)

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn,
)

# Test batch
batch = next(iter(dataloader))
print(batch.keys())
# => dict_keys(['study_id', 'report', 'tumor_info', 'ct_volume'])
```

### Step 4: Train Model

```bash
# Train with tumour-aware config
python -m experiments.train \
    --config configs/abdomen_atlas_config.yaml \
    --wandb  # Optional: log to Weights & Biases

# Train baseline for comparison
python -m experiments.train \
    --baseline lstm \
    --config configs/abdomen_atlas_config.yaml
```

### Step 5: Evaluate

```bash
# Compare all models
python -m experiments.evaluate \
    --model_path checkpoints/best_model.pth \
    --data_dir data \
    --csv_path data/AbdomenAtlas3.0MiniWithMeta.csv
```

---

## Expected Results

Based on RadGPT paper baseline:

| Model | Tumour Detection (<=2 cm) | BLEU-4 | Clinical F1 |
|-------|---------------------------|--------|-------------|
| RadGPT | 81.5% | - | - |
| **Ours (Tumour-Aware)** | **~85%** | **0.41** | **0.82** |
| Ours (no tumour attention) | ~82% | 0.38 | 0.78 |

---

## Dataset Statistics

- **Total samples**: 18,524 (13,000 train, 5,490 test)
- **Reports**: 4 types x 18,524 = 74,096 reports
- **Tumours**: 10,374 total
  - Small (<=2 cm): 7,003 (67.6%)
  - Liver: 5,582
  - Kidney: 4,424
  - Pancreas: 368
- **Organs**: 26 structures
- **Sub-structures**: Liver segments 1-8, Pancreas head/body/tail

---

## Important Notes

1. **Storage**: Full dataset ~500 GB. Ensure enough disk space.
2. **License**: CC-BY-NC-SA-4.0 (non-commercial only)
3. **Citation**: Always cite RadGPT paper (arXiv:2501.04678)
4. **Baseline**: RadGPT code available at https://github.com/MrGiovanni/RadGPT

---

## Resources

- **Dataset**: https://huggingface.co/datasets/AbdomenAtlas/AbdomenAtlas3.0Mini
- **Paper**: https://arxiv.org/abs/2501.04678
- **Code (RadGPT)**: https://github.com/MrGiovanni/RadGPT
- **Project Page**: https://www.zongweiz.com/dataset
