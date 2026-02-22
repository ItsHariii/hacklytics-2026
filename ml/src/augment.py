"""
Offline Audio Augmentation — Agent 3 (ML)

Loads raw audio segments from processed/ directory, applies time-domain
augmentations, re-extracts features, and saves an expanded training set.

Augmentations:
  - Time stretching (0.85x - 1.15x)
  - Pitch shifting (+/- 2 semitones)
  - Additive Gaussian noise (SNR 15-30 dB)
  - Random gain (+/- 6 dB)
  - Time shifting (roll by up to 20%)

Usage:
    python -m ml.src.augment
"""

import json
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm

from ml.src.features import (
    extract_mel_spectrogram, extract_tabular_features, TARGET_SR,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "ml" / "data" / "processed"
FEATURES_DIR = PROJECT_ROOT / "ml" / "data" / "features"

# Class-aware augmentation: more copies for minority classes to balance training
AUG_PER_CLASS = {0: 3, 1: 5, 2: 7, 3: 8}  # normal, crackles, wheezes, both


def augment_audio(audio: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """Apply a random combination of augmentations to an audio segment."""
    aug = audio.copy()
    rng = np.random

    # Time stretch (0.85x - 1.15x)
    if rng.random() < 0.5:
        rate = rng.uniform(0.85, 1.15)
        aug = librosa.effects.time_stretch(aug, rate=rate)

    # Pitch shift (+/- 2 semitones)
    if rng.random() < 0.4:
        n_steps = rng.uniform(-2, 2)
        aug = librosa.effects.pitch_shift(aug, sr=sr, n_steps=n_steps)

    # Additive Gaussian noise (SNR 15-30 dB)
    if rng.random() < 0.6:
        snr_db = rng.uniform(15, 30)
        signal_power = np.mean(aug ** 2) + 1e-10
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = rng.normal(0, np.sqrt(noise_power), len(aug)).astype(np.float32)
        aug = aug + noise

    # Random gain (+/- 6 dB)
    if rng.random() < 0.5:
        gain_db = rng.uniform(-6, 6)
        aug = aug * (10 ** (gain_db / 20))

    # Time shift (roll by up to 20%)
    if rng.random() < 0.3:
        shift = int(len(aug) * rng.uniform(-0.2, 0.2))
        aug = np.roll(aug, shift)

    return aug.astype(np.float32)


def main():
    print("=" * 60)
    print("OFFLINE AUDIO AUGMENTATION")
    print("=" * 60)

    # Load training metadata
    meta_path = PROCESSED_DIR / "train" / "metadata.json"
    with open(meta_path) as f:
        metadata = json.load(f)

    total_aug = sum(AUG_PER_CLASS.get(e["label_idx"], 4) for e in metadata)
    print(f"\n  Original training samples: {len(metadata)}")
    print(f"  Class-aware augmentation: {AUG_PER_CLASS}")
    print(f"  Target augmented set size: {len(metadata) + total_aug}")

    # Load normalization stats from original features
    norm_path = FEATURES_DIR / "normalization.npz"
    norm = np.load(norm_path)
    norm_mean, norm_std = norm["mean"], norm["std"]

    # Collect augmented features
    aug_mels = []
    aug_tabs = []
    aug_labels = []
    aug_diseases = []

    for entry in tqdm(metadata, desc="  Augmenting"):
        audio = np.load(PROCESSED_DIR / "train" / entry["audio_file"])
        sr = entry.get("sr", TARGET_SR)
        n_aug = AUG_PER_CLASS.get(entry["label_idx"], 4)

        for _ in range(n_aug):
            aug_audio = augment_audio(audio, sr)

            # Re-extract features from augmented audio
            mel = extract_mel_spectrogram(aug_audio, sr)
            tab = extract_tabular_features(aug_audio, sr)

            aug_mels.append(mel)
            aug_tabs.append(tab)
            aug_labels.append(entry["label_idx"])
            aug_diseases.append(entry["disease_onehot"])

    aug_mels = np.stack(aug_mels)
    aug_tabs = np.stack(aug_tabs)
    aug_labels = np.array(aug_labels)
    aug_diseases = np.array(aug_diseases)

    # Normalize tabular features with original training stats
    aug_tabs_norm = ((aug_tabs - norm_mean) / norm_std).astype(np.float32)

    # Load original training features
    orig_mel = np.load(FEATURES_DIR / "train" / "mel_spectrograms.npy")
    orig_tab = np.load(FEATURES_DIR / "train" / "tabular_features_norm.npy")
    orig_labels = np.load(FEATURES_DIR / "train" / "labels.npy")
    orig_diseases = np.load(FEATURES_DIR / "train" / "disease_labels.npy")

    # Combine original + augmented
    combined_mel = np.concatenate([orig_mel, aug_mels], axis=0)
    combined_tab = np.concatenate([orig_tab, aug_tabs_norm], axis=0)
    combined_labels = np.concatenate([orig_labels, aug_labels], axis=0)
    combined_diseases = np.concatenate([orig_diseases, aug_diseases], axis=0)

    # Save to augmented feature directory
    out_dir = FEATURES_DIR / "train_aug"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "mel_spectrograms.npy", combined_mel)
    np.save(out_dir / "tabular_features_norm.npy", combined_tab)
    np.save(out_dir / "labels.npy", combined_labels)
    np.save(out_dir / "disease_labels.npy", combined_diseases)

    print(f"\n  Original:  {orig_mel.shape[0]} samples")
    print(f"  Augmented: {aug_mels.shape[0]} samples")
    print(f"  Combined:  {combined_mel.shape[0]} samples")
    print(f"  Saved to:  {out_dir}")

    # Class distribution
    from collections import Counter
    names = ["normal", "crackles", "wheezes", "both"]
    counts = Counter(combined_labels.tolist())
    print(f"\n  Class distribution (combined):")
    for idx in sorted(counts.keys()):
        print(f"    {names[idx]}: {counts[idx]} ({100*counts[idx]/len(combined_labels):.1f}%)")

    print("\n" + "=" * 60)
    print("AUGMENTATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
