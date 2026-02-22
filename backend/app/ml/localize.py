"""
Temporal Localization — Sliding Window Event Detection

Slides the existing ONNX model across a full recording to identify
exactly where crackles and wheezes occur. Returns timestamped segments
that the frontend can highlight and play back for physician review.

No new ML dependencies — reuses the existing ONNX session and feature
extraction pipeline.
"""

import numpy as np
import onnxruntime as ort

from app.ml.features import (
    extract_mel_spectrogram,
    extract_tabular_features,
    TARGET_SR,
)

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _classify_window(
    audio_window: np.ndarray,
    sr: int,
    session: ort.InferenceSession,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
) -> tuple[str, float]:
    """Run ONNX inference on a single audio window. Returns (label, confidence)."""
    mel = extract_mel_spectrogram(audio_window, sr)
    tabular = extract_tabular_features(audio_window, sr)

    if norm_mean is not None and norm_std is not None:
        tabular = ((tabular - norm_mean) / norm_std).astype(np.float32)

    mel_input = mel[np.newaxis, np.newaxis, :, :]
    tab_input = tabular[np.newaxis, :]

    outputs = session.run(None, {
        "mel_spectrogram": mel_input,
        "tabular_features": tab_input,
    })

    crackle_prob = float(_sigmoid(outputs[0][0][0]))
    wheeze_prob = float(_sigmoid(outputs[1][0][0]))

    has_crackles = crackle_prob > 0.5
    has_wheezes = wheeze_prob > 0.5

    if has_crackles and has_wheezes:
        return "both", min(crackle_prob, wheeze_prob)
    elif has_crackles:
        return "crackles", crackle_prob
    elif has_wheezes:
        return "wheezes", wheeze_prob
    else:
        return "normal", 1.0 - max(crackle_prob, wheeze_prob)


def _merge_segments(raw_segments: list[dict], step_sec: float) -> list[dict]:
    """Merge overlapping/adjacent windows with the same sound type into
    contiguous segments, averaging their confidences."""
    if not raw_segments:
        return []

    raw_segments.sort(key=lambda s: s["start_sec"])
    merged = [raw_segments[0].copy()]

    for seg in raw_segments[1:]:
        prev = merged[-1]
        gap = seg["start_sec"] - prev["end_sec"]
        same_type = seg["sound_type"] == prev["sound_type"]

        if same_type and gap <= step_sec + 0.01:
            n_prev = prev.get("_count", 1)
            n_seg = 1
            prev["end_sec"] = seg["end_sec"]
            prev["confidence"] = (
                prev["confidence"] * n_prev + seg["confidence"]
            ) / (n_prev + n_seg)
            prev["_count"] = n_prev + n_seg
        else:
            merged.append(seg.copy())

    for seg in merged:
        seg.pop("_count", None)
        seg["confidence"] = round(seg["confidence"], 4)
        seg["start_sec"] = round(seg["start_sec"], 2)
        seg["end_sec"] = round(seg["end_sec"], 2)

    return merged


def localize_events(
    audio: np.ndarray,
    sr: int,
    session: ort.InferenceSession,
    norm_mean: np.ndarray = None,
    norm_std: np.ndarray = None,
    window_sec: float = 2.0,
    step_sec: float = 0.5,
    threshold: float = 0.6,
) -> list[dict]:
    """Slide the classifier across the audio and return detected event segments.

    Args:
        audio: Full recording waveform (1-D float array).
        sr: Sample rate (should be TARGET_SR = 16000).
        session: Loaded ONNX InferenceSession.
        norm_mean: Tabular feature normalization mean.
        norm_std: Tabular feature normalization std.
        window_sec: Length of each analysis window in seconds.
        step_sec: Hop between consecutive windows in seconds.
        threshold: Minimum confidence to keep a detection.

    Returns:
        List of dicts: [{"start_sec", "end_sec", "sound_type", "confidence"}]
    """
    window_samples = int(window_sec * sr)
    step_samples = int(step_sec * sr)
    total_samples = len(audio)

    if total_samples < window_samples // 2:
        return []

    raw_detections = []
    offset = 0

    while offset + window_samples // 2 <= total_samples:
        end = min(offset + window_samples, total_samples)
        window = audio[offset:end]

        if len(window) < sr * 0.3:
            break

        label, confidence = _classify_window(window, sr, session, norm_mean, norm_std)

        if label != "normal" and confidence >= threshold:
            raw_detections.append({
                "start_sec": offset / sr,
                "end_sec": end / sr,
                "sound_type": label,
                "confidence": confidence,
            })

        offset += step_samples

    return _merge_segments(raw_detections, step_sec)
