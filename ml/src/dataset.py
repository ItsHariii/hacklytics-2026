"""
ICBHI 2017 Dataset Preparation — Agent 3 (ML)

Loads the ICBHI 2017 respiratory sound dataset, parses annotations,
segments audio into respiratory cycles, labels them, creates a
patient-level train/val/test split, and computes class weights.

Usage:
    python -m ml.src.dataset
"""

import os
import csv
import json
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ICBHI_ROOT = PROJECT_ROOT / "data" / "ICBHI 2017" / "Respiratory_Sound_Database" / "Respiratory_Sound_Database"
AUDIO_DIR = ICBHI_ROOT / "audio_and_txt_files"
DIAGNOSIS_FILE = ICBHI_ROOT / "patient_diagnosis.csv"
DEMOGRAPHICS_FILE = PROJECT_ROOT / "data" / "ICBHI 2017" / "demographic_info.txt"
PROCESSED_DIR = PROJECT_ROOT / "ml" / "data" / "processed"

# Target sample rate for all audio
TARGET_SR = 16000

# Minimum cycle duration in seconds (skip very short segments)
MIN_CYCLE_DURATION = 0.5

# Label mapping
LABEL_MAP = {
    (0, 0): "normal",
    (1, 0): "crackles",
    (0, 1): "wheezes",
    (1, 1): "both",
}

LABEL_TO_IDX = {"normal": 0, "crackles": 1, "wheezes": 2, "both": 3}

# Disease labels
DISEASES = [
    "COPD", "Healthy", "URTI", "Pneumonia",
    "Bronchiectasis", "Bronchiolitis", "Asthma", "LRTI",
]
DISEASE_TO_IDX = {d: i for i, d in enumerate(DISEASES)}


# ──────────────────────────────────────────────
# Parse patient diagnoses
# ──────────────────────────────────────────────
def load_patient_diagnoses() -> dict[int, str]:
    """Load patient_diagnosis.csv → {patient_id: disease_name}"""
    diagnoses = {}
    with open(DIAGNOSIS_FILE, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            patient_id = int(row[0].strip())
            disease = row[1].strip()
            diagnoses[patient_id] = disease
    print(f"  Loaded {len(diagnoses)} patient diagnoses")
    return diagnoses


# ──────────────────────────────────────────────
# Parse annotation files
# ──────────────────────────────────────────────
def parse_annotations(txt_path: str) -> list[dict]:
    """Parse a single annotation .txt file → list of cycle dicts."""
    cycles = []
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            start = float(parts[0])
            end = float(parts[1])
            crackles = int(parts[2])
            wheezes = int(parts[3])
            label = LABEL_MAP[(crackles, wheezes)]
            cycles.append({
                "start": start,
                "end": end,
                "crackles": crackles,
                "wheezes": wheezes,
                "label": label,
                "label_idx": LABEL_TO_IDX[label],
            })
    return cycles


# ──────────────────────────────────────────────
# Parse filename metadata
# ──────────────────────────────────────────────
def parse_filename(wav_name: str) -> dict:
    """Extract metadata from ICBHI filename format:
    {patient}_{recording}_{chest_location}_{acq_mode}_{equipment}.wav
    """
    stem = wav_name.replace(".wav", "").replace(".txt", "")
    parts = stem.split("_")
    return {
        "patient_id": int(parts[0]),
        "recording_index": parts[1],
        "chest_location": parts[2],
        "acquisition_mode": parts[3],
        "equipment": parts[4],
    }


# ──────────────────────────────────────────────
# Segment audio by respiratory cycles
# ──────────────────────────────────────────────
def segment_recording(wav_path: str, cycles: list[dict], target_sr: int = TARGET_SR) -> list[dict]:
    """Load audio and segment into respiratory cycles using annotation timestamps."""
    try:
        audio, sr = librosa.load(wav_path, sr=target_sr, mono=True)
    except Exception as e:
        print(f"  ⚠️  Failed to load {wav_path}: {e}")
        return []

    segments = []
    for i, cycle in enumerate(cycles):
        start_sample = int(cycle["start"] * target_sr)
        end_sample = int(cycle["end"] * target_sr)

        # Skip if out of bounds or too short
        if start_sample >= len(audio) or end_sample > len(audio):
            continue
        if (cycle["end"] - cycle["start"]) < MIN_CYCLE_DURATION:
            continue

        segment_audio = audio[start_sample:end_sample]
        if len(segment_audio) == 0:
            continue

        segments.append({
            **cycle,
            "audio": segment_audio,
            "sr": target_sr,
            "duration": cycle["end"] - cycle["start"],
            "cycle_index": i,
        })

    return segments


# ──────────────────────────────────────────────
# Build the full dataset
# ──────────────────────────────────────────────
def build_dataset() -> list[dict]:
    """Process all ICBHI recordings → list of respiratory cycle dicts."""
    diagnoses = load_patient_diagnoses()

    # Find all wav files
    wav_files = sorted(AUDIO_DIR.glob("*.wav"))
    print(f"  Found {len(wav_files)} wav files")

    all_cycles = []
    skipped = 0

    for wav_path in tqdm(wav_files, desc="  Processing recordings"):
        # Find matching annotation file
        txt_path = wav_path.with_suffix(".txt")
        if not txt_path.exists():
            skipped += 1
            continue

        # Parse filename metadata
        meta = parse_filename(wav_path.name)

        # Parse annotations
        cycles = parse_annotations(str(txt_path))
        if not cycles:
            skipped += 1
            continue

        # Segment audio
        segments = segment_recording(str(wav_path), cycles)

        # Attach metadata to each segment
        patient_id = meta["patient_id"]
        disease = diagnoses.get(patient_id, "Unknown")
        disease_idx = DISEASE_TO_IDX.get(disease, -1)

        for seg in segments:
            seg["patient_id"] = patient_id
            seg["recording_file"] = wav_path.name
            seg["chest_location"] = meta["chest_location"]
            seg["equipment"] = meta["equipment"]
            seg["disease"] = disease
            seg["disease_idx"] = disease_idx
            # One-hot disease vector
            disease_vec = [0] * len(DISEASES)
            if disease_idx >= 0:
                disease_vec[disease_idx] = 1
            seg["disease_onehot"] = disease_vec

        all_cycles.extend(segments)

    print(f"  Total respiratory cycles extracted: {len(all_cycles)}")
    print(f"  Recordings skipped: {skipped}")
    return all_cycles


# ──────────────────────────────────────────────
# Patient-level train/val/test split
# ──────────────────────────────────────────────
def patient_level_split(cycles: list[dict], train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42):
    """
    Split cycles into train/val/test ensuring no patient appears in multiple sets.
    Uses GroupShuffleSplit with patient_id as the group key.
    """
    patient_ids = np.array([c["patient_id"] for c in cycles])
    indices = np.arange(len(cycles))

    # First split: train+val vs test
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    trainval_idx, test_idx = next(gss1.split(indices, groups=patient_ids))

    # Second split: train vs val (from the trainval portion)
    trainval_patient_ids = patient_ids[trainval_idx]
    val_fraction = val_ratio / (train_ratio + val_ratio)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    rel_train_idx, rel_val_idx = next(gss2.split(trainval_idx, groups=trainval_patient_ids))

    train_idx = trainval_idx[rel_train_idx]
    val_idx = trainval_idx[rel_val_idx]

    train_cycles = [cycles[i] for i in train_idx]
    val_cycles = [cycles[i] for i in val_idx]
    test_cycles = [cycles[i] for i in test_idx]

    # Verify no patient leakage
    train_patients = set(c["patient_id"] for c in train_cycles)
    val_patients = set(c["patient_id"] for c in val_cycles)
    test_patients = set(c["patient_id"] for c in test_cycles)

    assert train_patients.isdisjoint(val_patients), "Patient leakage between train and val!"
    assert train_patients.isdisjoint(test_patients), "Patient leakage between train and test!"
    assert val_patients.isdisjoint(test_patients), "Patient leakage between val and test!"

    print(f"\n  Split results (patient-level, no leakage):")
    print(f"    Train: {len(train_cycles)} cycles from {len(train_patients)} patients")
    print(f"    Val:   {len(val_cycles)} cycles from {len(val_patients)} patients")
    print(f"    Test:  {len(test_cycles)} cycles from {len(test_patients)} patients")

    return train_cycles, val_cycles, test_cycles


# ──────────────────────────────────────────────
# Class distribution & weights
# ──────────────────────────────────────────────
def compute_class_weights(cycles: list[dict]) -> dict:
    """Compute inverse-frequency class weights for imbalanced labels."""
    label_counts = Counter(c["label"] for c in cycles)
    total = sum(label_counts.values())
    n_classes = len(LABEL_TO_IDX)

    weights = {}
    for label, idx in LABEL_TO_IDX.items():
        count = label_counts.get(label, 1)
        weights[label] = total / (n_classes * count)

    print(f"\n  Class distribution:")
    for label in LABEL_TO_IDX:
        count = label_counts.get(label, 0)
        pct = 100.0 * count / total if total > 0 else 0
        print(f"    {label:10s}: {count:5d} ({pct:5.1f}%) — weight: {weights[label]:.3f}")

    return weights


# ──────────────────────────────────────────────
# Save processed data to disk
# ──────────────────────────────────────────────
def save_split(cycles: list[dict], split_name: str, output_dir: Path):
    """Save audio segments and metadata for a split."""
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    audio_list = []
    meta_list = []

    for i, cycle in enumerate(tqdm(cycles, desc=f"  Saving {split_name}")):
        # Save audio segment as .npy
        audio_filename = f"{split_name}_{i:05d}.npy"
        np.save(split_dir / audio_filename, cycle["audio"])

        # Collect metadata (without audio array)
        meta = {k: v for k, v in cycle.items() if k != "audio"}
        meta["audio_file"] = audio_filename
        meta_list.append(meta)

    # Save metadata as JSON
    meta_path = split_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta_list, f, indent=2, default=str)

    print(f"  Saved {len(meta_list)} cycles to {split_dir}")
    return meta_list


# ──────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("ICBHI 2017 DATASET PREPARATION")
    print("=" * 60)

    # Step 1: Build dataset
    print("\n[1/4] Loading and segmenting audio recordings...")
    cycles = build_dataset()

    if not cycles:
        print("ERROR: No cycles extracted. Check dataset paths.")
        return

    # Step 2: Compute class distribution
    print("\n[2/4] Computing class distribution and weights...")
    class_weights = compute_class_weights(cycles)

    # Step 3: Patient-level split
    print("\n[3/4] Creating patient-level train/val/test split...")
    train, val, test = patient_level_split(cycles)

    # Step 4: Save to disk
    print("\n[4/4] Saving processed data to disk...")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    save_split(train, "train", PROCESSED_DIR)
    save_split(val, "val", PROCESSED_DIR)
    save_split(test, "test", PROCESSED_DIR)

    # Save class weights
    weights_path = PROCESSED_DIR / "class_weights.json"
    with open(weights_path, "w") as f:
        json.dump(class_weights, f, indent=2)

    # Save label maps
    maps_path = PROCESSED_DIR / "label_maps.json"
    with open(maps_path, "w") as f:
        json.dump({
            "label_to_idx": LABEL_TO_IDX,
            "idx_to_label": {v: k for k, v in LABEL_TO_IDX.items()},
            "diseases": DISEASES,
            "disease_to_idx": DISEASE_TO_IDX,
        }, f, indent=2)

    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETE")
    print(f"  Output: {PROCESSED_DIR}")
    print(f"  Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
