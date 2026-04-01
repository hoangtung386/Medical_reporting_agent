# Refactoring Guide - Medical Report Generation Project

Tai lieu nay ghi lai qua trinh refactor du an, cac van de da duoc sua,
va trang thai hien tai cua codebase de team tiep nhan va phat trien tiep.

---

## 1. Tong quan thay doi

### Thong ke truoc va sau refactor

| Hang muc | Truoc | Sau |
|----------|-------|-----|
| Tong so file Python | 14 | 28 |
| File lon nhat | 464 dong (medgemma.py) | ~335 dong (medgemma.py) |
| Bug nghiem trong | 3 | 0 |
| File thua / trung lap | 3 | 0 |
| Test files | 1 (manual script) | 5 (pytest, 34 test cases) |
| Logging | print() khap noi | logging module nhat quan |
| Doc files o root | 4 | 0 (chuyen vao docs/) |
| Config structure | Khong nhat quan giua 2 file | Thong nhat voi base_config.yaml |
| Cross-attention | Khai bao nhung KHONG goi | Tich hop vao generation loop |
| Type hints | Python 3.10 syntax khong co __future__ | Co `from __future__ import annotations` |
| Package management | Chi co requirements.txt | Co pyproject.toml (pip + uv) |
| .gitignore | Khong co | Day du (dataset, checkpoints, IDE, ...) |

---

## 2. Cac bug nghiem trong da sua

### BUG 1: `experiments/train.py` - Bien `optimizer` khong ton tai

```python
# TRUOC (crash khi save checkpoint):
'optimizer_state_dict': optimizer.state_dict()

# SAU:
'optimizer_state_dict': trainer.optimizer.state_dict()
```

### BUG 2: `experiments/evaluate.py` - Test loader la placeholder rong

```python
# TRUOC (khong bao gio chay duoc):
test_loader = []

# SAU (load data that su):
test_dataset = AbdomenAtlasDataset(csv_path=..., data_dir=..., load_images=True)
test_loader = DataLoader(test_dataset, collate_fn=collate_fn, ...)
```

### BUG 3: `experiments/evaluate.py` - Baseline output khong decode

```python
# TRUOC (luon tra ve chuoi co dinh):
report = "Generated report placeholder"

# SAU (goi method cua model):
report = model.generate_report(ct_volume)
```

### BUG 4: `models/generation/medgemma.py` - Cross-attention khong dung trong generation

```python
# TRUOC: cross_attention_layers duoc goi nhung ket qua bi bo,
# generate() van dung combined_embeds goc:
hidden_states = cross_attn(hidden_states, seg_sequence)  # ket qua nay bi mat
outputs = self.llm.generate(inputs_embeds=combined_embeds, ...)  # dung lai embeds goc

# SAU: Custom generation loop, moi buoc decode deu chay qua cross-attention:
for _ in range(max_length):
    llm_outputs = self.llm(inputs_embeds=current_embeds, output_hidden_states=True)
    hidden_states = llm_outputs.hidden_states[-1]
    for cross_attn in self.cross_attention_layers:
        hidden_states = cross_attn(hidden_states, seg_sequence)
    next_logits = self.llm.lm_head(hidden_states[:, -1:, :])
    ...
```

### BUG 5: `experiments/train.py` - Import tu file da xoa

```python
# TRUOC:
from data.datasets.loader import AbdomenAtlasDataset  # file khong con ton tai

# SAU:
from data.datasets.abdomen_atlas import AbdomenAtlasDataset
```

### BUG 6: `models/generation/medgemma.py` - Hardcoded feature dimensions

```python
# TRUOC (64 = 4*4*4, thay doi pool_size se loi im lang):
self.seg_to_sequence = nn.Linear(seg_feature_size * 64, self.llm_hidden_size * 8)

# SAU (tinh dong):
pool_elements = math.prod(pool_size)
self.seg_to_sequence = nn.Linear(
    seg_feature_size * pool_elements,
    self.llm_hidden_size * num_seg_tokens,
)
```

---

## 3. Cac file da xoa

| File | Ly do xoa |
|------|-----------|
| `src/utils.py` | Trung lap voi `utils/measurements.py` (phien ban day du hon) |
| `data/datasets/loader.py` | Trung lap voi `abdomen_atlas.py` (dung heuristic khong chinh xac) |
| `configs/train_config.yaml` | Thay the boi `base_config.yaml` (cau truc thong nhat) |

---

## 4. Cac file da tach / di chuyen

| Hanh dong | Tu | Den | Ly do |
|-----------|-----|-----|-------|
| TACH | `medgemma.py` (464 dong, 3 class) | `attention.py` + `medgemma.py` + `trainer.py` | Single-responsibility |
| TACH | `baselines/__init__.py` (289 dong, 3 class) | `lstm.py` + `transformer_baseline.py` + `simple_cnn_lstm.py` | Moi model 1 file |
| CHUYEN | `BEFORE_AFTER.md` (root) | `docs/BEFORE_AFTER.md` | Gon root |
| CHUYEN | `RESTRUCTURE_SUMMARY.md` (root) | `docs/RESTRUCTURE_SUMMARY.md` | Gon root |
| CHUYEN | `DATASET_ANALYSIS.md` (root) | `docs/DATASET_ANALYSIS.md` | Gon root |
| CHUYEN | `DATASET_TESTING.md` (root) | `docs/DATASET_TESTING.md` | Gon root |
| CHUYEN | `test_dataset.py` (root) | `tests/test_dataset.py` | Gom vao test suite |

---

## 5. Cac file moi tao

| File | Muc dich |
|------|----------|
| `utils/logging.py` | setup_logger() thay the print() toan project |
| `models/generation/attention.py` | SegmentationAwareAttention (novel cross-attention) |
| `models/generation/trainer.py` | ReportGeneratorTrainer (tach training logic) |
| `models/baselines/lstm.py` | LSTMReportGenerator + decode() + generate_report() |
| `models/baselines/transformer_baseline.py` | TransformerBaseline + decode() + generate_report() |
| `models/baselines/simple_cnn_lstm.py` | SimpleCNNLSTM + decode() + generate_report() |
| `configs/base_config.yaml` | Config chuan, thong nhat structure |
| `tests/test_measurements.py` | 11 tests cho utils/measurements |
| `tests/test_metrics.py` | 10 tests cho utils/metrics |
| `tests/test_segmentation.py` | 5 tests cho models/segmentation (mock mode) |
| `tests/test_generation.py` | 5 tests cho models/generation (mock mode) |
| `tests/test_dataset.py` | 3 tests cho data collate_fn |
| `experiments/__init__.py` | Package init |
| `tests/__init__.py` | Package init |
| `.gitignore` | Ignore dataset, checkpoints, IDE, ... |
| `pyproject.toml` | pip + uv support, pytest/black/ruff config |

---

## 6. Cau truc thu muc hien tai

```
Medical_reporting_agent/
|-- configs/
|   |-- base_config.yaml
|   +-- abdomen_atlas_config.yaml
|-- data/
|   |-- QUICKSTART.md
|   +-- datasets/
|       |-- __init__.py
|       +-- abdomen_atlas.py
|-- models/
|   |-- __init__.py
|   |-- segmentation/
|   |   |-- __init__.py
|   |   +-- swinunetr.py
|   |-- generation/
|   |   |-- __init__.py
|   |   |-- attention.py
|   |   |-- medgemma.py
|   |   +-- trainer.py
|   +-- baselines/
|       |-- __init__.py
|       |-- lstm.py
|       |-- transformer_baseline.py
|       +-- simple_cnn_lstm.py
|-- utils/
|   |-- __init__.py
|   |-- logging.py
|   |-- measurements.py
|   |-- metrics.py
|   +-- rag.py
|-- experiments/
|   |-- __init__.py
|   |-- train.py
|   +-- evaluate.py
|-- tests/
|   |-- __init__.py
|   |-- test_measurements.py
|   |-- test_metrics.py
|   |-- test_segmentation.py
|   |-- test_generation.py
|   +-- test_dataset.py
|-- docs/
|   |-- BEFORE_AFTER.md
|   |-- RESTRUCTURE_SUMMARY.md
|   |-- DATASET_ANALYSIS.md
|   +-- DATASET_TESTING.md
|-- paper/
|   +-- draft.md
|-- main.py
|-- download_data.sh
|-- requirements.txt
|-- pyproject.toml
|-- .gitignore
+-- README.md
```

---

## 7. Huong dan cho team tiep nhan

### Cai dat

```bash
# Voi pip
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Voi uv
uv venv && uv pip install -e ".[dev]"
```

### Chay test

```bash
pytest                  # voi pip
uv run pytest           # voi uv
```

### Chay demo

```bash
python main.py
```

### Train model

```bash
python -m experiments.train --config configs/abdomen_atlas_config.yaml --wandb
```

### Evaluate

```bash
python -m experiments.evaluate --model_path checkpoints/best_model.pth
```

### Coding conventions

- **PEP 8** voi line length 88 (black formatter)
- **Double quotes** cho strings
- **`from __future__ import annotations`** o dau moi file co type hints moi (PEP 604)
- **`logging`** module thay vi print() (tru main.py la CLI demo)
- **Type hints** tren tat ca method signatures
- **Docstrings**: Google style, double-quote

### Cau truc config

Tat ca config deu co section `training:` voi cac key:
`num_epochs`, `learning_rate`, `batch_size`, `weight_decay`, `eval_every`, `save_every`.
File `experiments/train.py` co `load_config()` tu dong normalize config cu ve dang nay.

---

## 8. Nhung gi team can lam tiep

| Muc | Mo ta | Uu tien |
|-----|-------|---------|
| Tao data/datasets/transforms.py | Data augmentation pipeline (MONAI transforms) | CAO |
| Implement gradient_accumulation | Dang co trong config nhung chua dung trong train.py | CAO |
| Implement scheduler | Config co scheduler: cosine nhung train.py chua dung | TRUNG BINH |
| Them attention visualization | Config co save_attention_maps: true nhung chua implement | TRUNG BINH |
| Them multi-GPU support | Dung accelerate/deepspeed (da co trong optional deps) | THAP |
| Hoan thien paper/draft.md | Dien references, cap nhat dataset section | THAP |
| Expand test coverage | Them integration tests, end-to-end tests | TRUNG BINH |
