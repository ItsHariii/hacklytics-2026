# Accuracy Improvement Guide — RespiLens Lung Sound Classifier

This document provides detailed instructions for training, evaluating, and deploying the improved RespiLens model, which targets **high 60s accuracy** (65–69%) on the ICBHI 4-class lung sound classification task.

---

## Table of Contents

1. [Overview of Changes](#overview-of-changes)
2. [Prerequisites](#prerequisites)
3. [Data Pipeline](#data-pipeline)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [Export & Deployment](#export--deployment)
7. [Troubleshooting](#troubleshooting)

---

## Overview of Changes

### What Was Wrong (Before ~50% Accuracy)

| Issue | Impact |
|-------|--------|
| **Multi-label binary formulation** | Two independent binary heads (crackle + wheeze) compounded errors. "Both" class required both heads correct → only 5.56% recall. |
| **ResNet18 (ImageNet) backbone** | ImageNet features (edges, textures) don't transfer well to mel-spectrograms. |
| **Excessive regularization** | Mixup α=0.4, SpecAugment masking up to 47% of bins, dropout 0.5, label smoothing 0.05 → severe underfitting. |
| **Only 2 training phases** | No full backbone fine-tuning. |

### What Was Fixed

| Change | Purpose |
|--------|---------|
| **Direct 4-class classification head** | Model learns 4-class boundaries directly; no compounding error. |
| **CNN14 (AudioSet) backbone** | Audio-specific pretraining on 2M clips; understands transients, harmonics, temporal modulation. |
| **Reduced regularization** | Dropout 0.5→0.3, SpecAugment 3×20→2×12 bins, mixup α 0.4→0.2. |
| **3-phase training** | Phase 1: heads only. Phase 2: partial unfreeze. Phase 3: full backbone fine-tune. |

---

## Prerequisites

### 1. Environment

Ensure you have the required dependencies installed:

```bash
cd /Users/harimanivannan/Documents/GitHub/Project
pip install -r ml/requirements.txt
```

Key dependencies: `torch`, `torchvision`, `librosa`, `numpy`, `scikit-learn`, `onnx`, `onnxruntime`, `soundfile`, `tqdm`.

### 2. Dataset

The ICBHI 2017 Respiratory Sound Database must be available at:

```
Project/data/ICBHI 2017/Respiratory_Sound_Database/Respiratory_Sound_Database/
├── audio_and_txt_files/   # .wav and .txt annotation files
└── patient_diagnosis.csv
```

### 3. CNN14 Pretrained Weights (Auto-Download)

On first training run, the PANNs CNN14 weights (~300MB) will be downloaded automatically from Zenodo. Ensure you have:

- Internet access
- ~300MB free disk space in `ml/models/`

If download fails, the trainer falls back to ResNet18.

---

## Data Pipeline

### Step 1: Prepare Dataset

Segment audio into respiratory cycles and create patient-level train/val/test splits:

```bash
cd /Users/harimanivannan/Documents/GitHub/Project
python -m ml.src.dataset
```

**Output:** `ml/data/processed/{train,val,test}/` — audio segments (.npy) + metadata.json

### Step 2: Extract Features

Extract mel-spectrograms and tabular features for all splits:

```bash
python -m ml.src.features
```

**Output:** `ml/data/features/{train,val,test}/` — mel_spectrograms.npy, tabular_features_norm.npy, labels.npy, disease_labels.npy

### Step 3: Offline Augmentation (Recommended)

Expand training set with audio augmentations for better generalization:

```bash
python -m ml.src.augment
```

**Output:** `ml/data/features/train_aug/` — original + 4× augmented samples (~15k total)

---

## Training

### Full Training Command

```bash
cd /Users/harimanivannan/Documents/GitHub/Project
python -m ml.src.train
```

### What Happens During Training

| Phase | Description | Epochs | Patience | LR (Backbone) | LR (Heads) |
|-------|-------------|--------|----------|---------------|------------|
| **Phase 1** | Frozen backbone; train heads, fusion, projection | 20 | 8 | — | 1e-3 |
| **Phase 2** | Partial unfreeze (top layers: conv5, conv6, fc1 for CNN14) | 50 | 12 | 5e-5 | 5e-4 |
| **Phase 3** | Full backbone unfreeze; gentle fine-tune | 30 | 10 | 1e-5 | 1e-4 |

- **Early stopping:** Training stops if validation accuracy does not improve for the patience epochs in each phase.
- **Checkpoint:** Best model (by validation accuracy) is saved to `ml/models/best_model.pt`.
- **Config:** `ml/models/training_config.json` stores hyperparameters and results.

### Expected Training Time

- **Phase 1:** ~2–5 min (CPU) / ~1–2 min (GPU/MPS)
- **Phase 2:** ~10–20 min (CPU) / ~5–10 min (GPU/MPS)
- **Phase 3:** ~15–30 min (CPU) / ~8–15 min (GPU/MPS)

### Target Metrics

- **Validation accuracy (4-class):** Aim for **65–69%**
- **Test accuracy (4-class):** Typically similar to validation with patient-level split

---

## Evaluation

### Run Evaluation on Test Set

```bash
python -m ml.src.evaluate
```

**Output:**

- Console: 4-class accuracy, per-class precision/recall/F1, confusion matrix
- File: `ml/models/test_metrics.json`

### Metrics Reported

| Metric | Description |
|--------|-------------|
| **4-class accuracy** | Primary metric from direct class head |
| **Binary-reconstructed accuracy** | From crackle/wheeze heads (for comparison) |
| **Per-class precision/recall/F1** | For Normal, Crackles, Wheezes, Both |
| **Confusion matrix** | Predicted vs actual labels |
| **Crackle/Wheeze detection** | AUC, F1 from auxiliary binary heads |
| **Severity MAE** | Mean absolute error on 0–1 severity scale |

---

## Export & Deployment

### Export to ONNX

```bash
python -m ml.src.export_onnx
```

**Output:**

- `ml/models/model.onnx` — production-ready model
- `ml/models/onnx_export_meta.json` — export metadata, inference speed

### Inference (Python)

```python
from ml.src.inference import predict, predict_from_array, load_model

# Load model once at startup
load_model()  # or load_model("path/to/model.onnx")

# Predict from file
result = predict("path/to/audio.wav")

# Predict from numpy array
import numpy as np
audio = np.load("segment.npy")
result = predict_from_array(audio, sr=16000)

# Access results
print(result.classification)   # "normal", "crackles", "wheezes", "both"
print(result.confidence)      # 0.0–1.0
print(result.severity)        # 0–100
print(result.disease_probabilities)
print(result.shap_features)
```

### Backward Compatibility

The inference module auto-detects model format:

- **New (5 outputs):** Uses `class_logits` for classification
- **Legacy (4 outputs):** Uses binary crackle/wheeze logits for classification

---

## Troubleshooting

### CNN14 Download Fails

**Symptom:** `ConnectionError` or timeout when downloading PANNs weights.

**Fix:** Training will fall back to ResNet18. For CNN14, manually download:

```
https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth
```

Save to: `ml/models/Cnn14_mAP=0.431.pth`

### Out of Memory (OOM)

**Symptom:** CUDA/MPS OOM during training.

**Fix:** Reduce batch size in `train.py`:

```python
BATCH_SIZE = 16  # or 8
```

### Validation Accuracy Plateaus Low

**Possible causes:**

1. **No augmentation:** Run `python -m ml.src.augment` and ensure `train_aug` exists.
2. **Dataset too small:** ICBHI has ~5.2k cycles from 126 patients; patient-level split limits data.
3. **Class imbalance:** Class weights and weighted sampling are already applied; consider collecting more minority-class data.

### Test Accuracy Much Lower Than Validation

**Possible causes:**

1. **Patient-level split:** Test patients are unseen; some patient characteristics may differ.
2. **Equipment/recording variation:** ICBHI uses multiple stethoscope types; test set may have different distribution.

### ONNX Export Fails

**Symptom:** Shape mismatch or missing output.

**Fix:** Ensure you're using the latest trained model. Old checkpoints (pre-v3) may have a different architecture. Retrain with `python -m ml.src.train` and then export again.

---

## File Reference

| File | Purpose |
|------|---------|
| `ml/src/dataset.py` | ICBHI data prep, patient-level split |
| `ml/src/features.py` | Mel-spectrogram + tabular feature extraction |
| `ml/src/augment.py` | Offline audio augmentation |
| `ml/src/model.py` | RespiLensModel (CNN14/ResNet18, 4-class + binary heads) |
| `ml/src/train.py` | 3-phase training loop |
| `ml/src/evaluate.py` | Test set evaluation |
| `ml/src/export_onnx.py` | PyTorch → ONNX export |
| `ml/src/inference.py` | ONNX inference wrapper |
| `ml/src/explainability.py` | SHAP-style feature importance |

---

## Quick Reference: Full Pipeline

```bash
# 1. Prepare data
python -m ml.src.dataset

# 2. Extract features
python -m ml.src.features

# 3. Augment training set
python -m ml.src.augment

# 4. Train
python -m ml.src.train

# 5. Evaluate
python -m ml.src.evaluate

# 6. Export for deployment
python -m ml.src.export_onnx
```

---

*Last updated: February 2025*
