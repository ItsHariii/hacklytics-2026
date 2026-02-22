"""
Model Evaluation — Agent 3 (ML)

Evaluate the 4-class model on the test set.
Reports direct 4-class metrics from the classification head,
plus per-binary-task metrics (crackle/wheeze) from auxiliary heads.

Usage:
    python -m ml.src.evaluate           # Default: TTA=10, also reports no-TTA
    python -m ml.src.evaluate --no-tta  # No TTA only (comparable to validation)
    python -m ml.src.evaluate --tta-rounds 5
"""

import argparse
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
)

from ml.src.model import RespiLensModel
from ml.src.train import LungSoundDataset


def _tta_augment_mel(mel):
    """Light augmentation for test-time augmentation (SpecAugment-lite)."""
    aug = mel.clone()
    B, C, H, W = aug.shape
    for b in range(B):
        for _ in range(random.randint(1, 2)):
            f = random.randint(0, min(8, H - 1))
            f0 = random.randint(0, H - f)
            aug[b, :, f0:f0+f, :] = 0
        for _ in range(random.randint(1, 2)):
            t = random.randint(0, min(8, W - 1))
            t0 = random.randint(0, W - t)
            aug[b, :, :, t0:t0+t] = 0
    return aug

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "ml" / "models"

LABEL_NAMES = ["normal", "crackles", "wheezes", "both"]

def _decode_from_binary_probs(crackle_probs: np.ndarray,
                              wheeze_probs: np.ndarray,
                              thr_c: float,
                              thr_w: float) -> np.ndarray:
    has_c = (crackle_probs > thr_c).astype(np.int64)
    has_w = (wheeze_probs > thr_w).astype(np.int64)
    return has_c * 1 + has_w * 2


def _run_eval_loop(model, test_loader, DEVICE, tta_rounds):
    """Run evaluation with specified TTA rounds. Returns arrays and preds."""
    all_class_probs = []
    all_crackle_probs = []
    all_wheeze_probs = []
    all_crackle_true = []
    all_wheeze_true = []
    all_labels_4 = []
    all_severity_pred = []
    all_severity_true = []

    with torch.no_grad():
        for mel, tab, labels, has_c, has_w, diseases, severity in test_loader:
            mel, tab = mel.to(DEVICE), tab.to(DEVICE)

            cls_probs_sum = torch.zeros(mel.size(0), 4)
            c_probs_sum = torch.zeros(mel.size(0))
            w_probs_sum = torch.zeros(mel.size(0))
            sev_sum = torch.zeros(mel.size(0))

            for tta_i in range(tta_rounds):
                if tta_i == 0:
                    mel_aug, tab_aug = mel, tab
                else:
                    mel_aug = _tta_augment_mel(mel)
                    tab_aug = tab + torch.randn_like(tab) * 0.02

                cls_l, cl, wl, _, sv = model(mel_aug, tab_aug)
                cls_probs_sum += torch.softmax(cls_l, dim=1).cpu()
                c_probs_sum += torch.sigmoid(cl.squeeze(-1)).cpu()
                w_probs_sum += torch.sigmoid(wl.squeeze(-1)).cpu()
                sev_sum += sv.squeeze(-1).cpu()

            cls_prob = (cls_probs_sum / tta_rounds).numpy()
            c_prob = (c_probs_sum / tta_rounds).numpy()
            w_prob = (w_probs_sum / tta_rounds).numpy()

            all_class_probs.extend(cls_prob)
            all_crackle_probs.extend(c_prob)
            all_wheeze_probs.extend(w_prob)
            all_crackle_true.extend(has_c.numpy())
            all_wheeze_true.extend(has_w.numpy())
            all_labels_4.extend(labels.numpy())
            all_severity_pred.extend((sev_sum / tta_rounds).numpy())
            all_severity_true.extend(severity.numpy())

    return (
        np.array(all_class_probs),
        np.array(all_crackle_probs),
        np.array(all_wheeze_probs),
        np.array(all_crackle_true),
        np.array(all_wheeze_true),
        np.array(all_labels_4),
        np.array(all_severity_pred),
        np.array(all_severity_true),
    )


def evaluate_model(tta_rounds=10, no_tta_only=False, out_path: Optional[str] = None,
                   decode_binary: bool = False, thr_c: float = 0.5, thr_w: float = 0.5,
                   checkpoint_path: Optional[str] = None):
    """Load best model and evaluate on test set.
    tta_rounds: number of TTA rounds (1 = no TTA, comparable to validation)
    no_tta_only: if True, only run without TTA (faster, for val/test alignment check)"""
    print("=" * 60)
    print("MODEL EVALUATION (4-Class + Binary)")
    print("=" * 60)

    DEVICE = "cpu"
    print(f"\n  Device: {DEVICE}")

    ckpt_path = Path(checkpoint_path) if checkpoint_path else (MODELS_DIR / "best_model.pt")
    print(f"  Checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    backbone = checkpoint.get("backbone", "resnet18")
    if decode_binary:
        # If caller passes explicit thresholds, respect them.
        # Otherwise, fall back to checkpoint-tuned thresholds when available.
        if thr_c == 0.5 and "thr_crackle" in checkpoint:
            thr_c = float(checkpoint.get("thr_crackle"))
        if thr_w == 0.5 and "thr_wheeze" in checkpoint:
            thr_w = float(checkpoint.get("thr_wheeze"))

    model = RespiLensModel(pretrained=False, backbone=backbone).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"  Loaded checkpoint: phase {checkpoint.get('phase', '?')}, epoch {checkpoint['epoch']}, backbone={backbone}")
    print(f"  Training val accuracy: {checkpoint['val_acc']:.2f}%")
    if decode_binary:
        print(f"  Decoding via binary heads (thr_c={thr_c:.2f}, thr_w={thr_w:.2f})")

    test_ds = LungSoundDataset("test")
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
    print(f"  Test samples: {len(test_ds)}")

    # Run without TTA first (comparable to validation)
    print(f"\n  Running without TTA (comparable to validation)...")
    (all_class_probs, all_crackle_probs, all_wheeze_probs,
     all_crackle_true, all_wheeze_true, all_labels_4,
     all_severity_pred, all_severity_true) = _run_eval_loop(model, test_loader, DEVICE, tta_rounds=1)
    if decode_binary:
        preds_no_tta = _decode_from_binary_probs(all_crackle_probs, all_wheeze_probs, thr_c, thr_w)
        acc_no_tta = np.mean(preds_no_tta == all_labels_4)
    else:
        acc_no_tta = np.mean(np.argmax(all_class_probs, axis=1) == all_labels_4)
    print(f"  Accuracy (no TTA): {acc_no_tta*100:.2f}%")

    effective_tta = 1
    if no_tta_only:
        acc = acc_no_tta
        if decode_binary:
            preds_4 = preds_no_tta
        else:
            preds_4 = np.argmax(all_class_probs, axis=1)
        print(f"  (--no-tta: skipping TTA run)")
    else:
        # Run with TTA
        if tta_rounds > 1:
            print(f"\n  Running with TTA ({tta_rounds} rounds)...")
            (all_class_probs, all_crackle_probs, all_wheeze_probs,
             all_crackle_true, all_wheeze_true, all_labels_4,
             all_severity_pred, all_severity_true) = _run_eval_loop(
                model, test_loader, DEVICE, tta_rounds=tta_rounds)
            if decode_binary:
                preds_4 = _decode_from_binary_probs(all_crackle_probs, all_wheeze_probs, thr_c, thr_w)
                acc = np.mean(preds_4 == all_labels_4)
            else:
                acc = np.mean(np.argmax(all_class_probs, axis=1) == all_labels_4)
            effective_tta = tta_rounds
            print(f"  Accuracy (TTA={tta_rounds}): {acc*100:.2f}%")
            print(f"\n  Val alignment: no-TTA ({acc_no_tta*100:.2f}%) vs val ({checkpoint['val_acc']:.2f}%)")
        else:
            acc = acc_no_tta
        if decode_binary:
            preds_4 = _decode_from_binary_probs(all_crackle_probs, all_wheeze_probs, thr_c, thr_w)
        else:
            preds_4 = np.argmax(all_class_probs, axis=1)

    crackle_pred = (all_crackle_probs > 0.5).astype(int)
    wheeze_pred = (all_wheeze_probs > 0.5).astype(int)

    # ── Binary task metrics ──
    print(f"\n  === BINARY TASK METRICS (auxiliary heads) ===")

    crackle_acc = accuracy_score(all_crackle_true, crackle_pred)
    crackle_f1 = f1_score(all_crackle_true, crackle_pred, zero_division=0)
    crackle_prec = precision_score(all_crackle_true, crackle_pred, zero_division=0)
    crackle_rec = recall_score(all_crackle_true, crackle_pred, zero_division=0)
    crackle_auc = roc_auc_score(all_crackle_true, all_crackle_probs) if len(np.unique(all_crackle_true)) > 1 else 0.0

    print(f"  Crackle detection:")
    print(f"    Accuracy:  {crackle_acc*100:.2f}%")
    print(f"    Precision: {crackle_prec*100:.2f}%")
    print(f"    Recall:    {crackle_rec*100:.2f}%")
    print(f"    F1:        {crackle_f1*100:.2f}%")
    print(f"    AUC:       {crackle_auc:.4f}")

    wheeze_acc = accuracy_score(all_wheeze_true, wheeze_pred)
    wheeze_f1 = f1_score(all_wheeze_true, wheeze_pred, zero_division=0)
    wheeze_prec = precision_score(all_wheeze_true, wheeze_pred, zero_division=0)
    wheeze_rec = recall_score(all_wheeze_true, wheeze_pred, zero_division=0)
    wheeze_auc = roc_auc_score(all_wheeze_true, all_wheeze_probs) if len(np.unique(all_wheeze_true)) > 1 else 0.0

    print(f"\n  Wheeze detection:")
    print(f"    Accuracy:  {wheeze_acc*100:.2f}%")
    print(f"    Precision: {wheeze_prec*100:.2f}%")
    print(f"    Recall:    {wheeze_rec*100:.2f}%")
    print(f"    F1:        {wheeze_f1*100:.2f}%")
    print(f"    AUC:       {wheeze_auc:.4f}")

    # ── 4-class metrics (from class head) ──
    acc = accuracy_score(all_labels_4, preds_4)
    precision = precision_score(all_labels_4, preds_4, average="macro", zero_division=0)
    recall = recall_score(all_labels_4, preds_4, average="macro", zero_division=0)
    f1 = f1_score(all_labels_4, preds_4, average="macro", zero_division=0)

    print(f"\n  === 4-CLASS METRICS (direct class head) ===")
    print(f"  Accuracy:  {acc*100:.2f}%")
    print(f"  Precision: {precision*100:.2f}% (macro)")
    print(f"  Recall:    {recall*100:.2f}% (macro)")
    print(f"  F1 Score:  {f1*100:.2f}% (macro)")

    # Also report reconstructed from binary for comparison
    recon_4 = crackle_pred * 1 + wheeze_pred * 2
    recon_acc = accuracy_score(all_labels_4, recon_4)
    print(f"\n  (Binary-reconstructed accuracy: {recon_acc*100:.2f}% — for comparison)")

    print(f"\n  === PER-CLASS REPORT ===")
    print(classification_report(all_labels_4, preds_4, target_names=LABEL_NAMES, zero_division=0))

    print(f"  === CONFUSION MATRIX ===")
    cm = confusion_matrix(all_labels_4, preds_4)
    print(f"  {'':>10}", end="")
    for name in LABEL_NAMES:
        print(f"  {name:>8}", end="")
    print()
    for i, name in enumerate(LABEL_NAMES):
        print(f"  {name:>10}", end="")
        for j in range(len(LABEL_NAMES)):
            print(f"  {cm[i][j]:>8}", end="")
        print()

    # Severity MAE
    sev_mae = np.mean(np.abs(all_severity_pred - all_severity_true))
    print(f"\n  === SEVERITY ===")
    print(f"  MAE: {sev_mae:.4f}")

    # Save metrics
    metrics = {
        "formulation": "4-class direct + auxiliary binary",
        "backbone": backbone,
        "accuracy_4class": float(acc),
        "accuracy_no_tta": float(acc_no_tta),
        "tta_rounds": effective_tta,
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "binary_reconstructed_accuracy": float(recon_acc),
        "crackle_detection": {
            "accuracy": float(crackle_acc),
            "precision": float(crackle_prec),
            "recall": float(crackle_rec),
            "f1": float(crackle_f1),
            "auc": float(crackle_auc),
        },
        "wheeze_detection": {
            "accuracy": float(wheeze_acc),
            "precision": float(wheeze_prec),
            "recall": float(wheeze_rec),
            "f1": float(wheeze_f1),
            "auc": float(wheeze_auc),
        },
        "severity_mae": float(sev_mae),
        "confusion_matrix": cm.tolist(),
        "per_class": {
            name: {
                "precision": float(precision_score(all_labels_4 == i, preds_4 == i, zero_division=0)),
                "recall": float(recall_score(all_labels_4 == i, preds_4 == i, zero_division=0)),
                "f1": float(f1_score(all_labels_4 == i, preds_4 == i, zero_division=0)),
            }
            for i, name in enumerate(LABEL_NAMES)
        },
    }

    metrics_path = Path(out_path) if out_path else (MODELS_DIR / "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics saved to: {metrics_path}")

    print("\n" + "=" * 60)
    print(f"  RESULT: {acc*100:.2f}% accuracy (4-class) | F1: {f1*100:.2f}%")
    print(f"  Crackle AUC: {crackle_auc:.4f} | Wheeze AUC: {wheeze_auc:.4f}")
    print("=" * 60)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RespiLens on test set")
    parser.add_argument("--no-tta", action="store_true",
                        help="Skip TTA; report only no-TTA accuracy (comparable to validation)")
    parser.add_argument("--tta-rounds", type=int, default=10,
                        help="Number of TTA rounds when not --no-tta (default: 10)")
    parser.add_argument("--out", type=str, default=None,
                        help="Optional output path for metrics JSON (default: ml/models/test_metrics.json)")
    parser.add_argument("--decode-binary", action="store_true",
                        help="Use crackle/wheeze heads to decode 4-class (legacy style).")
    parser.add_argument("--thr-crackle", type=float, default=0.5,
                        help="Crackle threshold for --decode-binary (overridden by checkpoint if present).")
    parser.add_argument("--thr-wheeze", type=float, default=0.5,
                        help="Wheeze threshold for --decode-binary (overridden by checkpoint if present).")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint .pt (default: ml/models/best_model.pt)")
    args = parser.parse_args()
    evaluate_model(
        tta_rounds=args.tta_rounds,
        no_tta_only=args.no_tta,
        out_path=args.out,
        decode_binary=args.decode_binary,
        thr_c=args.thr_crackle,
        thr_w=args.thr_wheeze,
        checkpoint_path=args.checkpoint,
    )
