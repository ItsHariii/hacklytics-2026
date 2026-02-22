"""
Inference Module — Agent 3 (ML) → Deliverable to Agent 2

ONNX runtime inference wrapper for the RespiLens multi-label lung sound
classifier. Uses two binary heads (crackle + wheeze) and reconstructs
the 4-class label at inference time.

This module has NO PyTorch dependency — only numpy, librosa, onnxruntime.

Usage:
    from app.ml.inference import predict, load_model
    result = predict("path/to/audio.wav")
"""

import json
import numpy as np
import librosa
import onnxruntime as ort
from dataclasses import dataclass, asdict
from pathlib import Path

from app.ml.features import (
    extract_mel_spectrogram,
    extract_tabular_features,
    TARGET_SR,
    TABULAR_FEATURE_NAMES,
    FEATURE_LABELS,
)

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
ML_DIR = Path(__file__).resolve().parent
MODELS_DIR = ML_DIR
FEATURES_DIR = ML_DIR

# ──────────────────────────────────────────────
# Label mappings
# ──────────────────────────────────────────────
DISEASES = ["COPD", "Healthy", "URTI", "Pneumonia", "Bronchiectasis", "Bronchiolitis", "Asthma", "LRTI"]
CLASS_NAMES = ["normal", "crackles", "wheezes", "both"]


# ──────────────────────────────────────────────
# Prediction result type (matches api-contract.md)
# ──────────────────────────────────────────────
@dataclass
class PredictionResult:
    classification: str          # "normal", "crackles", "wheezes", "both"
    confidence: float            # 0.0–1.0
    severity: int                # 0–100
    disease_probabilities: list  # [{"disease": str, "probability": float}]
    shap_features: list          # [{"feature": str, "value": float, "label": str}]

    def to_dict(self):
        return asdict(self)


# ──────────────────────────────────────────────
# Model session manager
# ──────────────────────────────────────────────
_session = None
_norm_mean = None
_norm_std = None
_has_class_head = False


def load_model(model_path: str = None):
    """Load ONNX model and normalization stats. Call once at startup."""
    global _session, _norm_mean, _norm_std, _has_class_head

    if model_path is None:
        model_path = str(MODELS_DIR / "model.onnx")

    _session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],
    )

    output_names = [o.name for o in _session.get_outputs()]
    _has_class_head = "class_logits" in output_names

    norm_path = FEATURES_DIR / "normalization.npz"
    if norm_path.exists():
        norm = np.load(norm_path)
        _norm_mean = norm["mean"]
        _norm_std = norm["std"]
    else:
        _norm_mean = None
        _norm_std = None

    return _session


def get_session():
    """Get or lazily initialize the ONNX session."""
    global _session
    if _session is None:
        load_model()
    return _session


def get_norm_stats():
    """Return normalization stats (for use by localize module)."""
    return _norm_mean, _norm_std


# ──────────────────────────────────────────────
# Classification decoding
# ──────────────────────────────────────────────
def _decode_4class(class_logits):
    """Decode 4-class logits to label + confidence via softmax."""
    probs = _softmax(class_logits)
    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], round(float(probs[idx]), 4)


def _decode_multilabel(crackle_logit, wheeze_logit):
    """Convert binary logits to 4-class label + confidence."""
    crackle_prob = float(_sigmoid(crackle_logit))
    wheeze_prob = float(_sigmoid(wheeze_logit))

    has_crackles = crackle_prob > 0.5
    has_wheezes = wheeze_prob > 0.5

    if has_crackles and has_wheezes:
        classification = "both"
        confidence = min(crackle_prob, wheeze_prob)
    elif has_crackles:
        classification = "crackles"
        confidence = crackle_prob
    elif has_wheezes:
        classification = "wheezes"
        confidence = wheeze_prob
    else:
        classification = "normal"
        confidence = 1.0 - max(crackle_prob, wheeze_prob)

    return classification, round(confidence, 4)


# ──────────────────────────────────────────────
# Core prediction pipeline
# ──────────────────────────────────────────────
def _run_inference(mel_input, tab_input, tabular, session):
    """Shared inference logic — auto-detects model format (5-output or legacy 4-output)."""
    ort_inputs = {
        "mel_spectrogram": mel_input,
        "tabular_features": tab_input,
    }
    outputs = session.run(None, ort_inputs)

    if _has_class_head:
        class_logits, crackle_logit, wheeze_logit, disease_logits, severity_raw = outputs
        classification, confidence = _decode_4class(class_logits[0])
    else:
        crackle_logit, wheeze_logit, disease_logits, severity_raw = outputs
        classification, confidence = _decode_multilabel(
            crackle_logit[0][0], wheeze_logit[0][0])

    disease_probs = _softmax(disease_logits[0])
    disease_probabilities = [
        {"disease": d, "probability": round(float(p), 4)}
        for d, p in zip(DISEASES, disease_probs)
    ]
    disease_probabilities.sort(key=lambda x: x["probability"], reverse=True)

    severity = int(np.clip(severity_raw[0][0] * 100, 0, 100))

    shap_features = _compute_feature_importance(tabular, tab_input, mel_input, session)

    return PredictionResult(
        classification=classification,
        confidence=confidence,
        severity=severity,
        disease_probabilities=disease_probabilities,
        shap_features=shap_features,
    )


def predict(audio_path: str) -> PredictionResult:
    """Full pipeline: load audio → extract features → ONNX inference → result."""
    session = get_session()

    audio, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)

    mel = extract_mel_spectrogram(audio, sr)
    tabular = extract_tabular_features(audio, sr)

    if _norm_mean is not None and _norm_std is not None:
        tabular = ((tabular - _norm_mean) / _norm_std).astype(np.float32)

    mel_input = mel[np.newaxis, np.newaxis, :, :]
    tab_input = tabular[np.newaxis, :]

    return _run_inference(mel_input, tab_input, tabular, session)


def predict_from_array(audio: np.ndarray, sr: int = TARGET_SR) -> PredictionResult:
    """Predict from a numpy audio array (already loaded)."""
    session = get_session()

    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

    mel = extract_mel_spectrogram(audio, TARGET_SR)
    tabular = extract_tabular_features(audio, TARGET_SR)

    if _norm_mean is not None and _norm_std is not None:
        tabular = ((tabular - _norm_mean) / _norm_std).astype(np.float32)

    mel_input = mel[np.newaxis, np.newaxis, :, :]
    tab_input = tabular[np.newaxis, :]

    return _run_inference(mel_input, tab_input, tabular, session)


# ──────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _compute_feature_importance(
    tabular: np.ndarray,
    tab_input: np.ndarray,
    mel_input: np.ndarray,
    session: ort.InferenceSession,
    top_k: int = 5,
) -> list:
    """
    Compute feature importance via input perturbation (occlusion-based).
    Uses primary classification output (class_logits or crackle+wheeze) for importance.
    """
    baseline_out = session.run(None, {
        "mel_spectrogram": mel_input,
        "tabular_features": tab_input,
    })

    if _has_class_head:
        baseline_logits = baseline_out[0][0]
    else:
        baseline_logits = np.array([baseline_out[0][0][0], baseline_out[1][0][0]])

    importances = []
    for i in range(len(TABULAR_FEATURE_NAMES)):
        perturbed = tab_input.copy()
        perturbed[0, i] = 0.0

        perturbed_out = session.run(None, {
            "mel_spectrogram": mel_input,
            "tabular_features": perturbed,
        })

        if _has_class_head:
            pert_logits = perturbed_out[0][0]
        else:
            pert_logits = np.array([perturbed_out[0][0][0], perturbed_out[1][0][0]])

        delta = baseline_logits - pert_logits
        importance = float(np.sum(np.abs(delta)))
        sign = 1.0 if np.sum(delta) >= 0 else -1.0
        importances.append((i, importance * sign))

    importances.sort(key=lambda x: abs(x[1]), reverse=True)

    result = []
    for idx, value in importances[:top_k]:
        feat_name = TABULAR_FEATURE_NAMES[idx]
        label = FEATURE_LABELS.get(feat_name, feat_name)
        result.append({
            "feature": feat_name,
            "value": round(value, 4),
            "label": label,
        })

    return result


if __name__ == "__main__":
    import time

    print("=" * 60)
    print("INFERENCE MODULE TEST (Multi-Label)")
    print("=" * 60)

    model_path = MODELS_DIR / "model.onnx"
    if not model_path.exists():
        print(f"\n  Model not found at {model_path}")
        print("  Run export_onnx.py first.")
    else:
        load_model()
        print(f"\n  Model loaded: {model_path}")

        print("\n  Testing with synthetic audio...")
        test_audio = np.random.randn(TARGET_SR * 3).astype(np.float32) * 0.01
        t0 = time.perf_counter()
        result = predict_from_array(test_audio)
        elapsed = time.perf_counter() - t0

        print(f"\n  Result:")
        print(f"    Classification: {result.classification}")
        print(f"    Confidence:     {result.confidence:.4f}")
        print(f"    Severity:       {result.severity}")
        print(f"    Diseases:       {result.disease_probabilities[:3]}")
        print(f"    Top SHAP:       {result.shap_features[:3]}")
        print(f"    Inference time: {elapsed*1000:.1f} ms")

    print("\n" + "=" * 60)
