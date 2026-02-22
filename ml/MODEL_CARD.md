# RespiLens Model Card

## Model Overview

| Property | Value |
|---|---|
| Name | RespiLens Lung Sound Classifier |
| Version | 2.0 (transfer learning) |
| Architecture | ResNet18 (pretrained) + Tabular Dense + Multi-task Fusion |
| Framework | PyTorch 2.x, exported to ONNX (opset 17) |
| Task | 4-class lung sound classification + disease probability + severity |
| Dataset | ICBHI 2017 Respiratory Sound Database |

## Architecture

```
Input 1: Mel-spectrogram (1, 128, 128)
    → ResNet18 backbone (ImageNet pretrained, fine-tuned)
    → 512-dim feature vector

Input 2: 53 tabular audio features (MFCCs, spectral, energy, HNR)
    → Dense layers (128 → 64)
    → 64-dim feature vector

Fusion: Concatenate (576) → Dense (256 → 128)

Output heads:
    1. Sound classification: 4 classes (normal / crackles / wheezes / both)
    2. Disease probability: 8 diseases (COPD, Pneumonia, Asthma, etc.)
    3. Severity score: 0-100
```

**Total parameters:** 11,368,845
**ONNX model size:** 43.4 MB
**Inference speed:** 3.8 ms average (CPU, ONNX Runtime)

## Training

- **Transfer learning approach:** ResNet18 pretrained on ImageNet (1.2M images), fine-tuned on ICBHI 2017 lung sound mel-spectrograms
- **Phase 1:** Frozen backbone, trained fusion + heads for 14 epochs (LR=1e-3)
- **Phase 2:** Full fine-tuning with differential LR (backbone=1e-4, heads=5e-4) for 4 epochs
- **Data augmentation:** Offline (time stretch, pitch shift, noise, gain) + online (SpecAugment, tabular noise)
- **Training samples:** 15,685 (3,137 original + 12,548 augmented from 4x offline augmentation)
- **Loss:** Focal Loss (gamma=2.0) with class weights for sound head; BCE for disease; MSE for severity
- **Optimizer:** AdamW
- **Device:** Apple M3 Pro (MPS)
- **Data split:** Patient-level (no patient in both train and test)

## Performance

### Sound Classification (4-class)

| Metric | Value |
|---|---|
| **Test Accuracy** | **51.83%** |
| Precision (macro) | 40.97% |
| Recall (macro) | 36.77% |
| F1 Score (macro) | 36.53% |

### Per-Class Performance

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Normal | 0.54 | 0.79 | 0.64 | 798 |
| Crackles | 0.55 | 0.34 | 0.42 | 516 |
| Wheezes | 0.38 | 0.30 | 0.34 | 222 |
| Both | 0.17 | 0.04 | 0.06 | 162 |

### Confusion Matrix

|  | Pred: Normal | Pred: Crackles | Pred: Wheezes | Pred: Both |
|---|---|---|---|---|
| **Normal** | 631 | 104 | 52 | 11 |
| **Crackles** | 318 | 176 | 12 | 10 |
| **Wheezes** | 129 | 17 | 67 | 9 |
| **Both** | 88 | 24 | 44 | 6 |

### Severity Score
- MAE: 0.22 (on 0-1 scale)

## Limitations

1. **Small dataset:** ICBHI 2017 has only 126 patients (76 in training). The model learns patient-specific recording artifacts alongside genuine lung sound patterns.

2. **Class imbalance:** "Both" class (crackles + wheezes) has very few samples (4.6% of data), leading to poor performance on that class (6% F1).

3. **Patient-level split:** Using honest patient-level splits prevents data leakage but limits accuracy. Papers reporting 70%+ on ICBHI typically use recording-level splits where the same patient appears in train and test.

4. **Equipment variation:** The dataset was recorded with 4 different stethoscope types. The model may be sensitive to recording equipment characteristics.

5. **Severity scores are heuristic:** Severity labels are derived from classification labels, not from clinical severity assessments.

6. **Disease probabilities are approximate:** Disease labels come from patient diagnosis, not per-cycle diagnosis. Multiple recordings from the same patient share the same disease label.

## Intended Use

- Screening tool to assist healthcare professionals
- NOT a diagnostic device -- all results should be reviewed by qualified medical personnel
- Designed for respiratory cycle audio segments (0.5-5 seconds)

## Feature Extraction

The model requires two inputs extracted from audio:

1. **Mel-spectrogram:** 128 mel bins, 50-2000 Hz, 128 time frames, normalized to [0,1]
2. **53 tabular features:** 20 MFCCs (mean+std), spectral centroid/bandwidth/rolloff (mean+std), spectral flux, ZCR, RMS energy, HNR -- all z-score normalized using training set statistics

See `features.py` for exact extraction code. Normalization stats are in `data/features/normalization.npz`.

## Ethical Considerations

- This model should not be used as a standalone diagnostic tool
- Performance varies across patient demographics and recording conditions
- The training data is from a specific patient population and may not generalize to all populations
- All predictions should be validated by qualified healthcare professionals
