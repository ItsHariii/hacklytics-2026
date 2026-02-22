"""
Feature Extraction — Agent 3 (ML)

Extracts audio features for the RespiLens lung sound classifier.
This module is ALSO a deliverable to Agent 2 — it must work identically
at inference time using only librosa + numpy (no PyTorch).

Features extracted:
  - MFCCs (20 coefficients → mean + std = 40 values)
  - Spectral centroid, bandwidth, rolloff (mean + std = 6)
  - Spectral flux (mean + std = 2)
  - ZCR (mean + std = 2)
  - RMS energy (mean + std = 2)
  - HNR approximation (1)
  Total tabular features: 53

  - Mel-spectrogram (128 bins, fixed-length time frames)

Usage:
    python -m ml.src.features
"""

import os
import json
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
TARGET_SR = 16000
N_MELS = 128
N_MFCC = 20
HOP_LENGTH = 512
N_FFT = 2048
FMIN = 50
FMAX = 2000

# Fixed spectrogram length (pad/truncate to this many frames)
# ~2.5 seconds at 16kHz with hop=512 → ~78 frames; we use 128 for uniformity
SPEC_MAX_FRAMES = 128

# Feature names for SHAP plain-language labels
TABULAR_FEATURE_NAMES = []
for i in range(N_MFCC):
    TABULAR_FEATURE_NAMES.append(f"mfcc_{i}_mean")
    TABULAR_FEATURE_NAMES.append(f"mfcc_{i}_std")
for feat in ["spectral_centroid", "spectral_bandwidth", "spectral_rolloff"]:
    TABULAR_FEATURE_NAMES.append(f"{feat}_mean")
    TABULAR_FEATURE_NAMES.append(f"{feat}_std")
TABULAR_FEATURE_NAMES.extend(["spectral_flux_mean", "spectral_flux_std"])
TABULAR_FEATURE_NAMES.extend(["zcr_mean", "zcr_std"])
TABULAR_FEATURE_NAMES.extend(["rms_mean", "rms_std"])
TABULAR_FEATURE_NAMES.append("hnr")

N_TABULAR_FEATURES = len(TABULAR_FEATURE_NAMES)  # Should be 53

# Plain-language labels for SHAP (maps feature name → clinical description)
FEATURE_LABELS = {
    "mfcc_0_mean": "Average overall spectral shape (brightness)",
    "mfcc_0_std": "Variability in overall spectral shape",
    "mfcc_1_mean": "Low-frequency spectral energy balance",
    "mfcc_1_std": "Fluctuation in low-frequency energy",
    "mfcc_2_mean": "Mid-low frequency spectral pattern",
    "mfcc_2_std": "Variability in mid-low frequency pattern",
    "mfcc_3_mean": "Mid-frequency spectral energy pattern",
    "mfcc_3_std": "Fluctuation in mid-frequency energy",
    "mfcc_4_mean": "Spectral detail at ~1kHz region",
    "mfcc_5_mean": "Spectral detail at ~1.5kHz region",
    "mfcc_6_mean": "Spectral detail at ~2kHz region",
    "mfcc_7_mean": "High-frequency spectral energy pattern",
    "spectral_centroid_mean": "Average frequency center of gravity (wheeze indicator)",
    "spectral_centroid_std": "Variability in frequency center (breathing irregularity)",
    "spectral_bandwidth_mean": "Average frequency spread (broad = crackle, narrow = wheeze)",
    "spectral_bandwidth_std": "Variability in frequency spread",
    "spectral_rolloff_mean": "Frequency below which 85% of energy lies",
    "spectral_rolloff_std": "Variability in spectral rolloff",
    "spectral_flux_mean": "Average spectral change rate (crackle transient indicator)",
    "spectral_flux_std": "Variability in spectral change rate",
    "zcr_mean": "Average zero-crossing rate (high = crackles)",
    "zcr_std": "Variability in zero-crossing rate",
    "rms_mean": "Average sound energy level",
    "rms_std": "Energy variability (crackle bursts)",
    "hnr": "Harmonic-to-noise ratio (high = wheeze, low = crackle)",
}

# Fill in generic labels for remaining MFCCs
for i in range(N_MFCC):
    for suffix in ["mean", "std"]:
        key = f"mfcc_{i}_{suffix}"
        if key not in FEATURE_LABELS:
            desc = "mean" if suffix == "mean" else "variability in"
            FEATURE_LABELS[key] = f"Spectral coefficient #{i} ({desc} spectral detail)"


# ──────────────────────────────────────────────
# Core feature extraction functions
# ──────────────────────────────────────────────
def extract_mel_spectrogram(audio: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """
    Extract and normalize a mel-spectrogram, padded/truncated to fixed length.
    Returns: (N_MELS, SPEC_MAX_FRAMES) float32 array
    """
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr,
        n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH,
        fmin=FMIN, fmax=FMAX,
    )
    # Convert to log scale (dB).
    #
    # IMPORTANT: Avoid per-sample normalization (e.g., ref=np.max and per-sample min/max),
    # which erases global loudness / spectral-energy structure and creates a moving target
    # distribution for transfer learning. Use a fixed reference and fixed scaling instead.
    mel_db = librosa.power_to_db(mel, ref=1.0)

    # Pad or truncate to fixed length
    if mel_db.shape[1] < SPEC_MAX_FRAMES:
        pad_width = SPEC_MAX_FRAMES - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode="constant", constant_values=-80.0)
    else:
        mel_db = mel_db[:, :SPEC_MAX_FRAMES]

    # Fixed scaling: clip to [-80, 0] dB then map to [0, 1].
    mel_db = np.clip(mel_db, -80.0, 0.0)
    mel_db = (mel_db + 80.0) / 80.0

    return mel_db.astype(np.float32)


def extract_tabular_features(audio: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """
    Extract tabular (1D) features from audio segment.
    Returns: (N_TABULAR_FEATURES,) float32 array
    """
    features = []

    # MFCCs (20 × 2 = 40 values)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    for i in range(N_MFCC):
        features.append(np.mean(mfccs[i]))
        features.append(np.std(mfccs[i]))

    # Spectral centroid (2 values)
    cent = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    features.append(np.mean(cent))
    features.append(np.std(cent))

    # Spectral bandwidth (2 values)
    bw = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    features.append(np.mean(bw))
    features.append(np.std(bw))

    # Spectral rolloff (2 values)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    features.append(np.mean(rolloff))
    features.append(np.std(rolloff))

    # Spectral flux (2 values)
    spec = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH))
    flux = np.sqrt(np.sum(np.diff(spec, axis=1) ** 2, axis=0))
    features.append(np.mean(flux))
    features.append(np.std(flux))

    # Zero-crossing rate (2 values)
    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    features.append(np.mean(zcr))
    features.append(np.std(zcr))

    # RMS energy (2 values)
    rms = librosa.feature.rms(y=audio, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    features.append(np.mean(rms))
    features.append(np.std(rms))

    # HNR approximation (1 value) — ratio of harmonic to percussive energy
    harmonic, percussive = librosa.effects.hpss(audio)
    h_energy = np.sum(harmonic ** 2)
    p_energy = np.sum(percussive ** 2)
    hnr = 10 * np.log10(h_energy / (p_energy + 1e-10))
    features.append(hnr)

    return np.array(features, dtype=np.float32)


def extract_all_features(audio: np.ndarray, sr: int = TARGET_SR) -> dict:
    """
    Extract both mel-spectrogram and tabular features for a single audio segment.
    Returns dict with 'mel_spectrogram' and 'tabular' keys.
    """
    mel = extract_mel_spectrogram(audio, sr)
    tabular = extract_tabular_features(audio, sr)
    return {"mel_spectrogram": mel, "tabular": tabular}


# ──────────────────────────────────────────────
# Batch extraction from processed dataset
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "ml" / "data" / "processed"
FEATURES_DIR = PROJECT_ROOT / "ml" / "data" / "features"


def extract_split_features(split_name: str):
    """Extract features for all segments in a split (train/val/test)."""
    split_dir = PROCESSED_DIR / split_name
    meta_path = split_dir / "metadata.json"

    if not meta_path.exists():
        print(f"  ⚠️  No metadata found for {split_name}")
        return

    with open(meta_path, "r") as f:
        metadata = json.load(f)

    out_dir = FEATURES_DIR / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    all_mel = []
    all_tabular = []
    all_labels = []
    all_disease = []

    for entry in tqdm(metadata, desc=f"  Extracting {split_name} features"):
        # Load audio segment
        audio_path = split_dir / entry["audio_file"]
        audio = np.load(audio_path)

        # Extract features
        feats = extract_all_features(audio, sr=entry.get("sr", TARGET_SR))

        all_mel.append(feats["mel_spectrogram"])
        all_tabular.append(feats["tabular"])
        all_labels.append(entry["label_idx"])
        all_disease.append(entry["disease_onehot"])

    # Stack and save
    mel_array = np.stack(all_mel)          # (N, 128, 128)
    tab_array = np.stack(all_tabular)      # (N, 53)
    labels = np.array(all_labels)          # (N,)
    diseases = np.array(all_disease)       # (N, 8)

    np.save(out_dir / "mel_spectrograms.npy", mel_array)
    np.save(out_dir / "tabular_features.npy", tab_array)
    np.save(out_dir / "labels.npy", labels)
    np.save(out_dir / "disease_labels.npy", diseases)

    print(f"  {split_name}: mel={mel_array.shape}, tabular={tab_array.shape}, "
          f"labels={labels.shape}, diseases={diseases.shape}")

    return mel_array, tab_array, labels, diseases


def compute_and_save_normalization(train_tabular: np.ndarray):
    """Compute mean/std from training set for standard scaling. Save for inference."""
    mean = train_tabular.mean(axis=0)
    std = train_tabular.std(axis=0)
    std[std < 1e-8] = 1.0  # Prevent division by zero

    norm_path = FEATURES_DIR / "normalization.npz"
    np.savez(norm_path, mean=mean, std=std)
    print(f"  Saved normalization stats to {norm_path}")
    return mean, std


def compute_and_save_mel_normalization(train_mel: np.ndarray):
    """Compute mean/std for mel-spectrogram inputs from training set.

    Note: This is saved for analysis and future experiments; the current CNN14
    backbone expects the fixed [0,1] scaling used by extract_mel_spectrogram().
    """
    mean = train_mel.mean(axis=0)
    std = train_mel.std(axis=0)
    std[std < 1e-8] = 1.0

    norm_path = FEATURES_DIR / "mel_normalization.npz"
    np.savez(norm_path, mean=mean.astype(np.float32), std=std.astype(np.float32))
    print(f"  Saved mel normalization stats to {norm_path}")
    return mean, std


def normalize_features(tabular: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Standard-scale tabular features using precomputed stats."""
    return ((tabular - mean) / std).astype(np.float32)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("FEATURE EXTRACTION")
    print("=" * 60)

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    # Extract for each split
    print("\n[1/4] Extracting training features...")
    train_mel, train_tab, train_labels, train_diseases = extract_split_features("train")

    print("\n[2/4] Extracting validation features...")
    val_mel, val_tab, val_labels, val_diseases = extract_split_features("val")

    print("\n[3/4] Extracting test features...")
    test_mel, test_tab, test_labels, test_diseases = extract_split_features("test")

    # Compute normalization from training set
    print("\n[4/4] Computing normalization statistics...")
    mean, std = compute_and_save_normalization(train_tab)
    _mel_mean, _mel_std = compute_and_save_mel_normalization(train_mel)

    # Apply normalization to all splits
    for split_name, tab_data in [("train", train_tab), ("val", val_tab), ("test", test_tab)]:
        normalized = normalize_features(tab_data, mean, std)
        np.save(FEATURES_DIR / split_name / "tabular_features_norm.npy", normalized)

    # Save feature names for SHAP
    names_path = FEATURES_DIR / "feature_names.json"
    with open(names_path, "w") as f:
        json.dump({
            "tabular_feature_names": TABULAR_FEATURE_NAMES,
            "feature_labels": FEATURE_LABELS,
            "n_tabular": N_TABULAR_FEATURES,
            "n_mels": N_MELS,
            "spec_max_frames": SPEC_MAX_FRAMES,
        }, f, indent=2)

    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION COMPLETE")
    print(f"  Output: {FEATURES_DIR}")
    print(f"  Tabular features per sample: {N_TABULAR_FEATURES}")
    print(f"  Mel-spectrogram shape: ({N_MELS}, {SPEC_MAX_FRAMES})")
    print("=" * 60)


if __name__ == "__main__":
    main()
