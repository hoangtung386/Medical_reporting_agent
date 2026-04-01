# Vision-Guided Medical Report Generation via Segmentation-Aware Attention

**Research-grade implementation of automated radiology report generation from 3D medical images.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

This repository contains the code for our research on **tumour-aware medical report generation**. Our approach leverages **3D segmentation features** with a focus on **early cancer detection** to guide the report generation process through a novel **segmentation-aware cross-attention mechanism**.

**Dataset**: [**AbdomenAtlas3.0Mini**](https://huggingface.co/datasets/AbdomenAtlas/AbdomenAtlas3.0Mini) (18,524 CT-report pairs) with focus on tumour detection (10,374 tumours including 7,003 small tumours <=2 cm).

### Key Contributions

1. **Tumour-Aware Segmentation-Guided Attention** -- Novel cross-attention mechanism with separate attention for tumours vs. organs, enabling targeted early detection
2. **Early Detection Focus** -- Prioritise small tumours (<=2 cm) for early cancer detection with specialised loss weighting
3. **Vision-Language Fusion** -- Effective integration of 3D segmentation features (26 organs + 3 tumour types) with medical language models
4. **Comprehensive Baselines** -- Systematic comparison with LSTM, Transformer, and RadGPT baselines
5. **Multi-Report Evaluation** -- Assess on structured, narrative, and enhanced report types

### Why This Approach?

**Problem with existing methods:**
- Generic vision-language models ignore fine-grained anatomical structure
- Lack of spatial grounding leads to hallucinations
- No explicit connection between visual findings and textual descriptions

**Our solution:**
- Use 3D segmentation to provide **anatomical grounding**
- Cross-attention mechanism ensures the model "looks at" relevant regions
- Deterministic measurement extraction for factual accuracy

---

## Architecture

```
CT/MRI Volume (3D)
        |
+------------------------+
|  Segmentation Module   |  (SwinUNETR + SuPreM weights)
|  - Multi-scale features|
|  - Organ masks         |
+----------+-------------+
           |
     +-----+------+
     |            |
+-----------+ +----------------+
| Measure-  | | Encoder        |
| ments     | | Features       |
| (determ.) | | (for attention)|
+-----+-----+ +-------+--------+
      |               |
      +-------+-------+
              |
   +---------------------+
   | Report Generator     |
   | (MedGemma-2B + LoRA) |
   | + Seg-Aware Attention |
   +----------+-----------+
              |
       Radiology Report
```

### Core Components

| Component | Description | Location |
|-----------|-------------|----------|
| **Segmentation Model** | 3D SwinUNETR with multi-scale feature extraction | `models/segmentation/` |
| **Cross-Attention** | Novel segmentation-aware attention layer | `models/generation/attention.py` |
| **Report Generator** | MedGemma-2B with LoRA + cross-attention fusion | `models/generation/medgemma.py` |
| **Training Utilities** | Trainer wrapper with loss computation | `models/generation/trainer.py` |
| **Baseline Models** | LSTM, Transformer, SimpleCNN-LSTM for comparison | `models/baselines/` |
| **Measurements** | Deterministic volume / dimension calculation | `utils/measurements.py` |
| **Metrics** | BLEU, ROUGE, METEOR, Clinical Accuracy | `utils/metrics.py` |
| **RAG (Optional)** | Clinical guideline retrieval | `utils/rag.py` |

---

## Dataset

### AbdomenAtlas3.0Mini

We use [**AbdomenAtlas3.0Mini**](https://huggingface.co/datasets/AbdomenAtlas/AbdomenAtlas3.0Mini), a comprehensive dataset for tumour-focused medical report generation:

**Statistics**:
- **18,524 CT-report pairs** (13,000 train / 5,490 test)
- **10,374 tumours** (liver, kidney, pancreas)
  - **7,003 small tumours <=2 cm** (early detection focus)
- **3 report types**: Structured, Narrative, Enhanced
- **26 anatomical structures** + tumour annotations
- **Per-voxel segmentation** with WHO-standard measurements

**Download**:
```bash
# Download via included script (~500 GB full)
bash download_data.sh

# Or see the quick-start guide for lighter alternatives
cat data/QUICKSTART.md
```

**Citation**:
```bibtex
@article{bassi2025radgpt,
  title={RadGPT: Constructing 3D Image-Text Tumor Datasets},
  author={Bassi, Pedro R. A. S. and others},
  journal={arXiv preprint arXiv:2501.04678},
  year={2025}
}
```

---

## Quick Start

### 1. Installation

**With pip (standard):**
```bash
git clone https://github.com/hoangtung386/Medical_reporting_agent.git
cd Medical_reporting_agent

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -e ".[dev]"          # install project + dev tools
```

**With uv (faster):**
```bash
git clone https://github.com/hoangtung386/Medical_reporting_agent.git
cd Medical_reporting_agent

uv venv
uv pip install -e ".[dev]"

# Or use the lock-file workflow:
# uv lock && uv sync --extra dev
```

### 2. Download Pre-trained Weights

```bash
# SuPreM weights for SwinUNETR segmentation
mkdir -p checkpoints
wget https://github.com/MrGiovanni/SuPreM/releases/download/v0.1.0/supervised_suprem_swinunetr_2100.pth \
     -O checkpoints/suprem_swinunetr.pth

# MedGemma-2B downloads automatically on first run via HuggingFace
```

### 3. Run Demo

```bash
# Simple demo with mock data
python main.py

# Run on a real CT scan
python main.py --input data/sample_ct.nii.gz --output results/report.txt
```

**Expected output:**
```
GENERATED RADIOLOGY REPORT
======================================================================
CLINICAL INDICATION: Routine abdominal CT without contrast

FINDINGS:
Liver: Normal size and attenuation. No focal lesions identified.
Spleen: Unremarkable.
Kidneys: Bilateral kidneys are normal in size...
...
======================================================================
```

### 4. Run Tests

```bash
pytest                           # with pip
# or
uv run pytest                    # with uv
```

---

## Experiment Reproduction

### Training

**Option 1: Tumour-Aware Model (Recommended)**
```bash
python -m experiments.train \
    --config configs/abdomen_atlas_config.yaml \
    --wandb
```

**Option 2: General Model**
```bash
python -m experiments.train \
    --config configs/base_config.yaml
```

**Option 3: Baseline Models**
```bash
# LSTM baseline
python -m experiments.train --baseline lstm --config configs/abdomen_atlas_config.yaml

# Transformer baseline
python -m experiments.train --baseline transformer --config configs/abdomen_atlas_config.yaml
```

### Evaluation

```bash
python -m experiments.evaluate --model_path checkpoints/best_model.pth
```

**Sample results:**

| Model | BLEU-4 | ROUGE-L | METEOR | Clinical F1 | Tumour Detection* |
|-------|--------|---------|--------|-------------|-------------------|
| **Ours (Tumour-Aware)** | **0.412** | **0.587** | **0.445** | **0.823** | **~85%** |
| Ours (w/o tumour attn) | 0.398 | 0.572 | 0.431 | 0.811 | ~82% |
| RadGPT (baseline) | - | - | - | - | 81.5% |
| Transformer | 0.345 | 0.521 | 0.389 | 0.742 | - |
| LSTM | 0.302 | 0.478 | 0.351 | 0.698 | - |

*Sensitivity for small tumours (<=2 cm)

---

## Project Structure

```
Medical_reporting_agent/
|
|-- configs/                        # Configuration files
|   |-- base_config.yaml            #   General / default settings
|   +-- abdomen_atlas_config.yaml   #   Tumour-aware experiment config
|
|-- data/                           # Data management
|   |-- QUICKSTART.md               #   Dataset download & usage guide
|   +-- datasets/
|       |-- __init__.py
|       +-- abdomen_atlas.py        #   AbdomenAtlas dataset loader + collate
|
|-- models/                         # Core models
|   |-- __init__.py
|   |-- segmentation/
|   |   |-- __init__.py
|   |   +-- swinunetr.py            #   SwinUNETR segmentation + wrapper
|   |-- generation/
|   |   |-- __init__.py
|   |   |-- attention.py            #   SegmentationAwareAttention (novel)
|   |   |-- medgemma.py             #   Report generator (MedGemma + LoRA)
|   |   +-- trainer.py              #   Training loss / optimiser wrapper
|   +-- baselines/
|       |-- __init__.py
|       |-- lstm.py                 #   LSTM baseline
|       |-- transformer_baseline.py #   Transformer baseline
|       +-- simple_cnn_lstm.py      #   CNN-LSTM baseline
|
|-- utils/                          # Utilities
|   |-- __init__.py
|   |-- logging.py                  #   Project-wide logging setup
|   |-- measurements.py             #   Deterministic volume/dimension calc
|   |-- metrics.py                  #   BLEU, ROUGE, METEOR, Clinical F1
|   +-- rag.py                      #   RAG retrieval (optional)
|
|-- experiments/                    # Training & evaluation scripts
|   |-- __init__.py
|   |-- train.py
|   +-- evaluate.py
|
|-- tests/                          # Test suite (pytest)
|   |-- test_measurements.py
|   |-- test_metrics.py
|   |-- test_segmentation.py
|   |-- test_generation.py
|   +-- test_dataset.py
|
|-- docs/                           # Historical / design documentation
|   |-- BEFORE_AFTER.md
|   |-- RESTRUCTURE_SUMMARY.md
|   |-- DATASET_ANALYSIS.md
|   +-- DATASET_TESTING.md
|
|-- paper/                          # Paper materials
|   +-- draft.md
|
|-- main.py                         # End-to-end demo script
|-- download_data.sh                # Dataset download helper
|-- requirements.txt                # Pip requirements (flat)
|-- pyproject.toml                  # Project metadata, uv / pip editable
|-- REFACTORING_GUIDE.md            # Refactoring log & team handoff guide
|-- .gitignore
+-- README.md
```

---

## Research Details

### Dataset

We use **AbdomenAtlas 3.0** for training:
- 9,262 CT scans with expert annotations
- 25 abdominal organs labelled
- Paired with radiology reports

**Data format:**
```python
{
    "ct_volume": torch.Tensor,       # [1, D, H, W]
    "report": str,                   # Ground-truth radiology report
    "tumor_info": {                  # Tumour metadata from CSV
        "liver":    {"volume_cm3": ..., "lesion_count": ..., ...},
        "kidney":   {...},
        "pancreas": {...},
    },
    "study_id": str,
}
```

### Novel Components Explained

#### 1. Segmentation-Aware Cross-Attention

Located in `models/generation/attention.py`:

```python
class SegmentationAwareAttention(nn.Module):
    """
    Cross-attention between LLM hidden states and segmentation features.
    Allows the model to attend to specific anatomical regions.
    """
    def forward(self, llm_hidden, seg_features):
        # Query: from LLM hidden states
        # Key/Value: from segmentation encoder
        attention_output = self.cross_attn(
            query=llm_hidden,
            key=seg_features,
            value=seg_features,
        )
        return attention_output
```

**Why this works:**
- When generating "liver: unremarkable", the model attends to the liver region
- Grounds language generation in visual anatomy
- Reduces hallucinations through explicit visual reference

#### 2. Deterministic Measurements

Unlike AI-based approaches we use pure mathematics (`utils/measurements.py`):

```python
def calculate_volumes(masks, spacing):
    """Pure numpy/scipy -- no neural network."""
    volumes = {}
    for organ_id in unique_labels:
        voxel_count = np.sum(masks == organ_id)
        volume_mm3 = voxel_count * np.prod(spacing)
        volumes[organ_id] = volume_mm3
    return volumes
```

**Benefits:** 100 % reproducible, no hallucinations, clinically verifiable.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{your2026vision,
  title={Vision-Guided Medical Report Generation via Segmentation-Aware Attention},
  author={Le Vu Hoang Tung},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

---

## Contributing

This is research code. To maintain reproducibility:

1. **Don't** modify core model architecture without documenting changes
2. **Do** add new baselines in `models/baselines/`
3. **Do** report bugs via GitHub Issues
4. **Do** submit improvements via Pull Requests

---

## License

MIT License -- see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- **SuPreM**: Pre-trained segmentation weights ([GitHub](https://github.com/MrGiovanni/SuPreM))
- **MedGemma**: Medical language model ([Hugging Face](https://huggingface.co/google/medgemma-2b))
- **MONAI**: Medical imaging framework ([GitHub](https://github.com/Project-MONAI/MONAI))
- **AbdomenAtlas**: Dataset ([Paper](https://www.nature.com/articles/s41597-022-01719-2))

---

## Contact

For research inquiries:
- **Author**: Le Vu Hoang Tung
- **Email**: levuhoangtung1542003@gmail.com
- **GitHub Issues**: [For technical questions](https://github.com/hoangtung386/Medical_reporting_agent/issues)

---

## Version History

- **v1.1.0** (2026-04): Refactored project structure
  - Split large modules into single-responsibility files
  - Added test suite (pytest)
  - PEP 8 compliance throughout
  - Unified configuration with `base_config.yaml`
  - Added `uv` support via `pyproject.toml`
  - Integrated cross-attention into LLM forward pass
  - Fixed critical bugs in training / evaluation scripts

- **v1.0.0** (2026-01): Initial research release
  - Core segmentation-guided architecture
  - 3 baseline models for comparison
  - Evaluation pipeline with 6 metrics
  - Comprehensive documentation

---

**Built for advancing medical AI research**
