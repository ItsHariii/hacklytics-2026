"""
Explainability Module — Agent 3 (ML) → Deliverable to Agent 2

Computes SHAP-style feature attributions for RespiLens predictions.
Uses a combination of:
  1. Occlusion-based importance for tabular features (fast, no dependencies)
  2. Background-subtraction approach for spectral features

No PyTorch dependency — only numpy, librosa, onnxruntime.

Usage:
    from ml.src.explainability import explain_prediction
    shap_features = explain_prediction(audio, sr, session, norm_mean, norm_std)
"""

import numpy as np
import onnxruntime as ort
from pathlib import Path

from ml.src.features import (
    extract_mel_spectrogram,
    extract_tabular_features,
    TARGET_SR,
    TABULAR_FEATURE_NAMES,
    FEATURE_LABELS,
    N_TABULAR_FEATURES,
)


def explain_prediction(
    audio: np.ndarray,
    sr: int,
    session: ort.InferenceSession,
    norm_mean: np.ndarray = None,
    norm_std: np.ndarray = None,
    top_k: int = 5,
    n_background: int = 20,
) -> list:
    """
    Compute feature attributions for a single prediction.

    Uses perturbation-based importance:
    - For each tabular feature, measures the change in predicted class logit
      when the feature is replaced with its mean (zero in normalized space).
    - Combines with spectral region importance from mel-spectrogram masking.

    Args:
        audio: Audio waveform (1D numpy array).
        sr: Sample rate.
        session: ONNX inference session.
        norm_mean: Normalization mean for tabular features.
        norm_std: Normalization std for tabular features.
        top_k: Number of top features to return.
        n_background: Number of perturbation samples for spectral importance.

    Returns:
        List of top_k dicts with 'feature', 'value', 'label' keys.
    """
    # Extract features
    mel = extract_mel_spectrogram(audio, sr)
    tabular = extract_tabular_features(audio, sr)

    # Normalize
    if norm_mean is not None and norm_std is not None:
        tabular_norm = ((tabular - norm_mean) / norm_std).astype(np.float32)
    else:
        tabular_norm = tabular.copy()

    mel_input = mel[np.newaxis, np.newaxis, :, :]
    tab_input = tabular_norm[np.newaxis, :]

    # Baseline prediction
    baseline_out = session.run(None, {
        "mel_spectrogram": mel_input,
        "tabular_features": tab_input,
    })
    baseline_logits = baseline_out[0][0]
    pred_idx = int(np.argmax(baseline_logits))
    baseline_score = baseline_logits[pred_idx]

    # ── Tabular feature importance (occlusion) ──
    tab_importances = {}
    for i in range(N_TABULAR_FEATURES):
        perturbed = tab_input.copy()
        perturbed[0, i] = 0.0  # Replace with mean (zero in normalized space)

        out = session.run(None, {
            "mel_spectrogram": mel_input,
            "tabular_features": perturbed,
        })
        perturbed_score = out[0][0][pred_idx]
        importance = float(baseline_score - perturbed_score)
        tab_importances[TABULAR_FEATURE_NAMES[i]] = importance

    # ── Spectral region importance (frequency band masking) ──
    spectral_regions = {
        "low_frequency_energy": (0, 32),      # 0-500 Hz
        "mid_frequency_energy": (32, 64),      # 500-1000 Hz
        "high_frequency_energy": (64, 96),     # 1000-1500 Hz
        "very_high_frequency_energy": (96, 128), # 1500-2000 Hz
    }

    spectral_labels = {
        "low_frequency_energy": "Low frequency band energy (breathing sounds)",
        "mid_frequency_energy": "Mid frequency band energy (crackle indicators)",
        "high_frequency_energy": "High frequency band energy (wheeze indicators)",
        "very_high_frequency_energy": "Very high frequency band (fine crackles)",
    }

    for region_name, (start_bin, end_bin) in spectral_regions.items():
        masked_mel = mel_input.copy()
        masked_mel[0, 0, start_bin:end_bin, :] = 0.0

        out = session.run(None, {
            "mel_spectrogram": masked_mel,
            "tabular_features": tab_input,
        })
        perturbed_score = out[0][0][pred_idx]
        importance = float(baseline_score - perturbed_score)
        tab_importances[region_name] = importance

    # Merge all importances and sort by absolute value
    all_features = []
    for feat_name, importance in tab_importances.items():
        label = FEATURE_LABELS.get(feat_name, spectral_labels.get(feat_name, feat_name))
        all_features.append({
            "feature": feat_name,
            "value": round(importance, 4),
            "label": label,
        })

    all_features.sort(key=lambda x: abs(x["value"]), reverse=True)
    return all_features[:top_k]


def explain_batch(
    audio_list: list,
    sr: int,
    session: ort.InferenceSession,
    norm_mean: np.ndarray = None,
    norm_std: np.ndarray = None,
    top_k: int = 5,
) -> list:
    """Compute explanations for a batch of audio samples."""
    return [
        explain_prediction(audio, sr, session, norm_mean, norm_std, top_k)
        for audio in audio_list
    ]


if __name__ == "__main__":
    import time

    print("=" * 60)
    print("EXPLAINABILITY MODULE TEST")
    print("=" * 60)

    model_path = Path(__file__).resolve().parents[2] / "ml" / "models" / "model.onnx"
    features_dir = Path(__file__).resolve().parents[2] / "ml" / "data" / "features"

    if not model_path.exists():
        print(f"\n  Model not found at {model_path}")
        print("  Run export_onnx.py first.")
    else:
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

        # Load normalization
        norm_path = features_dir / "normalization.npz"
        norm = np.load(norm_path)
        norm_mean, norm_std = norm["mean"], norm["std"]

        # Test with synthetic audio
        print("\n  Testing with 3-second synthetic audio...")
        test_audio = np.random.randn(TARGET_SR * 3).astype(np.float32) * 0.01

        t0 = time.perf_counter()
        features = explain_prediction(test_audio, TARGET_SR, session, norm_mean, norm_std)
        elapsed = time.perf_counter() - t0

        print(f"\n  Top {len(features)} features:")
        for f in features:
            direction = "+" if f["value"] > 0 else "-"
            print(f"    [{direction}] {f['feature']:30s}  {f['value']:+.4f}  — {f['label']}")

        print(f"\n  Explanation time: {elapsed*1000:.1f} ms")

    print("\n" + "=" * 60)
