"""
Training Loop — Agent 3 (ML)

Three-phase transfer learning with direct 4-class classification + auxiliary
binary heads. Uses CNN14 (AudioSet-pretrained) backbone for audio-specific
feature extraction.

Phase 1: Frozen backbone — train heads, fusion, tabular, projection
Phase 2: Partial unfreeze — fine-tune top backbone layers with lower LR
Phase 3: Full unfreeze — gentle full-model fine-tuning

Key improvements over v2:
  - Direct 4-class CrossEntropy head (eliminates compounding binary errors)
  - CNN14 backbone (AudioSet pretrained >> ImageNet for audio)
  - Reduced regularization (less dropout, lighter SpecAugment, softer mixup)
  - 3-phase training schedule

Usage:
    python -m ml.src.train
"""

import json
import random
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from collections import Counter

from ml.src.model import RespiLensModel, count_parameters

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = PROJECT_ROOT / "ml" / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "ml" / "models"
PROCESSED_DIR = PROJECT_ROOT / "ml" / "data" / "processed"


# ──────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
class LungSoundDataset(Dataset):
    """PyTorch dataset with 4-class labels, binary targets, and SpecAugment."""

    def __init__(self, split: str, augment: bool = False):
        split_dir = FEATURES_DIR / split
        self.mel = torch.from_numpy(
            np.load(split_dir / "mel_spectrograms.npy")).float().unsqueeze(1)
        self.tabular = torch.from_numpy(
            np.load(split_dir / "tabular_features_norm.npy")).float()
        labels_np = np.load(split_dir / "labels.npy")
        self.diseases = torch.from_numpy(
            np.load(split_dir / "disease_labels.npy")).float()
        self.augment = augment

        self.labels = torch.from_numpy(labels_np.astype(np.int64)).long()

        self.has_crackles = torch.zeros(len(labels_np), dtype=torch.float32)
        self.has_wheezes = torch.zeros(len(labels_np), dtype=torch.float32)
        for i, lbl in enumerate(labels_np):
            if lbl in (1, 3):
                self.has_crackles[i] = 1.0
            if lbl in (2, 3):
                self.has_wheezes[i] = 1.0

        severity_base = {0: 10, 1: 45, 2: 55, 3: 80}
        rng = np.random.RandomState(42)
        sev_np = np.array([
            np.clip(severity_base[int(l)] + rng.normal(0, 10), 0, 100) / 100.0
            for l in labels_np
        ], dtype=np.float32)
        self.severity = torch.from_numpy(sev_np).float()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mel = self.mel[idx]
        tab = self.tabular[idx]
        if self.augment:
            mel = self._spec_augment(mel)
            tab = tab + torch.randn_like(tab) * 0.02
        return (mel, tab, self.labels[idx],
                self.has_crackles[idx], self.has_wheezes[idx],
                self.diseases[idx], self.severity[idx])

    @staticmethod
    def _spec_augment(mel, n_freq=2, n_time=2, f_param=8, t_param=10):
        """Light SpecAugment — small masks to avoid underfitting."""
        augmented = mel.clone()
        _, h, w = augmented.shape
        for _ in range(n_freq):
            f = random.randint(0, min(f_param, h - 1))
            f0 = random.randint(0, h - f)
            augmented[:, f0:f0+f, :] = 0
        for _ in range(n_time):
            t = random.randint(0, min(t_param, w - 1))
            t0 = random.randint(0, w - t)
            augmented[:, :, t0:t0+t] = 0
        if random.random() < 0.3:
            augmented = augmented + torch.randn_like(augmented) * 0.02
        return augmented


# ──────────────────────────────────────────────
# Losses
# ──────────────────────────────────────────────
FOCAL_GAMMA = 2.0


class FocalLoss(nn.Module):
    """
    Focal loss: FL(p) = -alpha * (1-p)^gamma * log(p)
    Down-weights easy examples (e.g. Normal) to focus on hard/minority classes.
    Optional label_smoothing (e.g. 0.05) improves calibration and generalization.
    """

    def __init__(self, class_weights=None, gamma=2.0, label_smoothing=0.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        if class_weights is not None:
            self.register_buffer("weight", torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.weight = None

    def forward(self, logits, targets):
        n_classes = logits.size(1)
        probs = F.softmax(logits, dim=1)

        if self.label_smoothing > 0:
            smooth = self.label_smoothing / (n_classes - 1)
            one_hot = F.one_hot(targets, n_classes).float()
            soft_targets = one_hot * (1 - self.label_smoothing) + (1 - one_hot) * smooth
            pt = (probs * soft_targets).sum(dim=1)
            log_probs = F.log_softmax(logits, dim=1)
            ce = -(soft_targets * log_probs).sum(dim=1)
        else:
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            ce = -torch.log(pt + 1e-8)

        alpha_t = self.weight[targets] if self.weight is not None else 1.0
        focal_weight = (1 - pt) ** self.gamma
        loss = alpha_t * focal_weight * ce
        return loss.mean() if self.reduction == "mean" else loss.sum()


class FocalSoftLoss(nn.Module):
    """Focal-style loss for mixup batches with soft targets."""

    def __init__(self, gamma=2.0, aux_weight=0.3):
        super().__init__()
        self.gamma = gamma
        self.aux_weight = aux_weight

    def forward(self, class_logits, crackle_logit, wheeze_logit,
                soft_labels, soft_crackles, soft_wheezes):
        probs = F.softmax(class_logits, dim=1)
        p_max = (probs * soft_labels).sum(dim=1)
        focal_weight = (1 - p_max) ** self.gamma
        log_probs = F.log_softmax(class_logits, dim=1)
        loss_ce = -(focal_weight * (soft_labels * log_probs).sum(dim=1)).mean()

        loss_c = F.binary_cross_entropy_with_logits(
            crackle_logit.squeeze(-1), soft_crackles)
        loss_w = F.binary_cross_entropy_with_logits(
            wheeze_logit.squeeze(-1), soft_wheezes)
        loss_aux = loss_c + loss_w

        return loss_ce + self.aux_weight * loss_aux, loss_ce, loss_aux


class CombinedLoss(nn.Module):
    """
    Primary: 4-class Focal Loss with class weights (gamma=2).
    Auxiliary: BCE for binary crackle/wheeze heads (lower weight).
    """

    def __init__(self, class_weights=None, crackle_pos_weight=1.0,
                 wheeze_pos_weight=1.0, aux_weight=0.3, focal_gamma=2.0,
                 label_smoothing=0.05):
        super().__init__()
        self.aux_weight = aux_weight
        self.focal_loss = FocalLoss(
            class_weights=class_weights, gamma=focal_gamma,
            label_smoothing=label_smoothing,
        )
        self.crackle_bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([crackle_pos_weight]))
        self.wheeze_bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([wheeze_pos_weight]))

    def forward(self, class_logits, crackle_logit, wheeze_logit,
                labels, has_crackles, has_wheezes):
        loss_ce = self.focal_loss(class_logits, labels)

        loss_c = self.crackle_bce(crackle_logit.squeeze(-1), has_crackles)
        loss_w = self.wheeze_bce(wheeze_logit.squeeze(-1), has_wheezes)
        loss_aux = loss_c + loss_w

        return loss_ce + self.aux_weight * loss_aux, loss_ce, loss_aux


class SoftCombinedLoss(nn.Module):
    """Combined loss for mixup batches with soft targets (FocalSoftLoss)."""

    def __init__(self, aux_weight=0.3, focal_gamma=2.0):
        super().__init__()
        self.focal_soft = FocalSoftLoss(gamma=focal_gamma, aux_weight=aux_weight)

    def forward(self, class_logits, crackle_logit, wheeze_logit,
                soft_labels, soft_crackles, soft_wheezes):
        return self.focal_soft(
            class_logits, crackle_logit, wheeze_logit,
            soft_labels, soft_crackles, soft_wheezes)


# ──────────────────────────────────────────────
# Mixup
# ──────────────────────────────────────────────
def mixup_batch(mel, tab, labels, has_c, has_w, n_classes=4, alpha=0.2):
    """Mixup with one-hot label mixing for 4-class CE."""
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)
    batch_size = mel.size(0)
    perm = torch.randperm(batch_size)

    labels_onehot = F.one_hot(labels, n_classes).float()

    mixed_mel = lam * mel + (1 - lam) * mel[perm]
    mixed_tab = lam * tab + (1 - lam) * tab[perm]
    mixed_labels = lam * labels_onehot + (1 - lam) * labels_onehot[perm]
    mixed_c = lam * has_c + (1 - lam) * has_c[perm]
    mixed_w = lam * has_w + (1 - lam) * has_w[perm]

    return mixed_mel, mixed_tab, mixed_labels, mixed_c, mixed_w


# ──────────────────────────────────────────────
# Legacy (v2-style) decoding + training helpers
# ──────────────────────────────────────────────
def decode_4class_from_binary_probs(crackle_prob: np.ndarray,
                                   wheeze_prob: np.ndarray,
                                   thr_c: float = 0.5,
                                   thr_w: float = 0.5) -> np.ndarray:
    """Convert binary probabilities to 4-class labels: 0..3."""
    has_c = (crackle_prob > thr_c).astype(np.int64)
    has_w = (wheeze_prob > thr_w).astype(np.int64)
    return has_c * 1 + has_w * 2


@torch.no_grad()
def eval_binary_and_tune_thresholds(model, loader, device,
                                   thr_grid=(0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7)):
    """Evaluate binary heads and optionally tune thresholds on decoded 4-class acc."""
    model.eval()
    all_c = []
    all_w = []
    all_y = []

    for mel, tab, labels, has_c, has_w, diseases, severity in loader:
        mel, tab = mel.to(device), tab.to(device)
        labels = labels.to(device)

        cls_logits, cr_logit, wh_logit, _, _ = model(mel, tab)
        c_prob = torch.sigmoid(cr_logit.squeeze(-1)).detach().cpu().numpy()
        w_prob = torch.sigmoid(wh_logit.squeeze(-1)).detach().cpu().numpy()
        all_c.append(c_prob)
        all_w.append(w_prob)
        all_y.append(labels.detach().cpu().numpy())

    c = np.concatenate(all_c, axis=0)
    w = np.concatenate(all_w, axis=0)
    y = np.concatenate(all_y, axis=0)

    # Default thresholds
    best_thr_c = 0.5
    best_thr_w = 0.5
    best_acc = float(np.mean(decode_4class_from_binary_probs(c, w, best_thr_c, best_thr_w) == y))

    # Grid search thresholds to maximize decoded 4-class accuracy
    for tc in thr_grid:
        for tw in thr_grid:
            pred = decode_4class_from_binary_probs(c, w, tc, tw)
            acc = float(np.mean(pred == y))
            if acc > best_acc:
                best_acc = acc
                best_thr_c = float(tc)
                best_thr_w = float(tw)

    return best_acc * 100.0, best_thr_c, best_thr_w


# ──────────────────────────────────────────────
# Training & evaluation
# ──────────────────────────────────────────────
def train_epoch(model, loader, criterion, soft_criterion, optimizer, device,
                use_mixup=True, mixup_prob=0.3, mixup_alpha=0.2):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    total = 0

    for mel, tab, labels, has_c, has_w, diseases, severity in loader:
        mel, tab = mel.to(device), tab.to(device)
        labels = labels.to(device)
        has_c, has_w = has_c.to(device), has_w.to(device)

        if use_mixup and random.random() < mixup_prob:
            m_mel, m_tab, m_labels, m_c, m_w = mixup_batch(
                mel, tab, labels, has_c, has_w, alpha=mixup_alpha)
            optimizer.zero_grad()
            cls_logits, cr_logit, wh_logit, _, _ = model(m_mel, m_tab)
            loss, _, _ = soft_criterion(
                cls_logits, cr_logit, wh_logit, m_labels, m_c, m_w)
        else:
            optimizer.zero_grad()
            cls_logits, cr_logit, wh_logit, _, _ = model(mel, tab)
            loss, _, _ = criterion(
                cls_logits, cr_logit, wh_logit, labels, has_c, has_w)

            preds = cls_logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * mel.size(0)
        total += mel.size(0)

    if all_labels:
        acc = 100.0 * np.mean(np.array(all_preds) == np.array(all_labels))
    else:
        acc = 0.0
    return total_loss / total, acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    total = 0

    for mel, tab, labels, has_c, has_w, diseases, severity in loader:
        mel, tab = mel.to(device), tab.to(device)
        labels = labels.to(device)
        has_c, has_w = has_c.to(device), has_w.to(device)

        cls_logits, cr_logit, wh_logit, _, _ = model(mel, tab)
        loss, _, _ = criterion(
            cls_logits, cr_logit, wh_logit, labels, has_c, has_w)

        total_loss += loss.item() * mel.size(0)
        total += mel.size(0)

        preds = cls_logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = 100.0 * np.mean(np.array(all_preds) == np.array(all_labels))
    return total_loss / total, acc


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-binary", action="store_true",
                        help="Train legacy v2-style: optimize crackle/wheeze binary heads and decode to 4-class.")
    parser.add_argument("--backbone", type=str, default=None,
                        choices=["cnn14", "resnet18"],
                        help="Override backbone (default: cnn14 for 4-class mode, resnet18 for legacy-binary).")
    args = parser.parse_args()

    print("=" * 60)
    if args.legacy_binary:
        print("LEGACY TRAINING (v2 — binary crackle/wheeze → decoded 4-class)")
    else:
        print("4-CLASS TRANSFER LEARNING (v3 — CNN14 + direct classification)")
    print("=" * 60)

    DEVICE = get_device()
    BATCH_SIZE = 32
    print(f"\n  Device: {DEVICE}")

    aug_dir = FEATURES_DIR / "train_aug"
    if aug_dir.exists() and (aug_dir / "mel_spectrograms.npy").exists():
        train_split = "train_aug"
        print(f"  Using AUGMENTED training set")
    else:
        train_split = "train"
        print(f"  Using original training set")

    print("\n  Loading datasets...")
    train_ds = LungSoundDataset(train_split, augment=True)
    val_ds = LungSoundDataset("val", augment=False)
    print(f"    Train: {len(train_ds)} | Val: {len(val_ds)}")

    label_counts = Counter(train_ds.labels.numpy().tolist())
    names = ["normal", "crackles", "wheezes", "both"]
    for i in sorted(label_counts):
        print(f"    {names[i]}: {label_counts[i]} ({100*label_counts[i]/len(train_ds):.1f}%)")

    # Class weights for CE loss (inverse frequency, capped)
    n_total = len(train_ds)
    n_classes = 4
    class_weights = []
    for i in range(n_classes):
        count = label_counts.get(i, 1)
        w = n_total / (n_classes * count)
        class_weights.append(min(w, 4.0))
    print(f"\n  Class weights: {[f'{w:.2f}' for w in class_weights]}")

    # Binary pos weights
    n_crackle_pos = train_ds.has_crackles.sum().item()
    n_wheeze_pos = train_ds.has_wheezes.sum().item()
    crackle_pos_weight = min((n_total - n_crackle_pos) / max(n_crackle_pos, 1), 4.0)
    wheeze_pos_weight = min((n_total - n_wheeze_pos) / max(n_wheeze_pos, 1), 5.0)
    print(f"  Crackle pos_weight: {crackle_pos_weight:.2f}")
    print(f"  Wheeze pos_weight:  {wheeze_pos_weight:.2f}")

    if args.legacy_binary:
        # Legacy setup: keep natural distribution (better calibrated decoded accuracy).
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  drop_last=True, num_workers=0)
    else:
        # Weighted random sampler: balance minority classes without crushing Normal
        # Softer than before — Normal 0.85x avoids underfitting majority class
        OVERSAMPLE_FACTORS = {0: 0.85, 1: 1.3, 2: 2.5, 3: 3.5}
        sample_weights = []
        for i in range(len(train_ds)):
            lbl = train_ds.labels[i].item()
            count = label_counts[lbl]
            factor = OVERSAMPLE_FACTORS.get(lbl, 1.0)
            sample_weights.append(factor / count)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                                  drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                            num_workers=0)

    # Try CNN14 first, fall back to ResNet18
    if args.backbone is not None:
        BACKBONE = args.backbone
    else:
        BACKBONE = "resnet18" if args.legacy_binary else "cnn14"
    try:
        model = RespiLensModel(pretrained=True, backbone=BACKBONE).to(DEVICE)
        print(f"\n  Backbone: CNN14 (AudioSet pretrained)")
    except Exception as e:
        print(f"\n  CNN14 failed ({e}), falling back to ResNet18")
        BACKBONE = "resnet18"
        model = RespiLensModel(pretrained=True, backbone=BACKBONE).to(DEVICE)
        print(f"  Backbone: ResNet18 (ImageNet pretrained)")

    if args.legacy_binary:
        # Legacy v2-style: train only crackle/wheeze heads.
        crackle_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([crackle_pos_weight], device=DEVICE))
        wheeze_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([wheeze_pos_weight], device=DEVICE))
    else:
        criterion = CombinedLoss(
            class_weights=class_weights,
            crackle_pos_weight=crackle_pos_weight,
            wheeze_pos_weight=wheeze_pos_weight,
            aux_weight=0.3,
            focal_gamma=FOCAL_GAMMA,
            label_smoothing=0.05,
        ).to(DEVICE)
        soft_criterion = SoftCombinedLoss(aux_weight=0.3, focal_gamma=FOCAL_GAMMA).to(DEVICE)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    save_path = MODELS_DIR / ("best_model_legacy.pt" if args.legacy_binary else "best_model.pt")
    best_thr_c, best_thr_w = 0.5, 0.5

    # ────────────────────────────────────────────
    # PHASE 1: Freeze backbone, train heads only
    # ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 1: Frozen backbone — training heads + fusion + projection")
    print("=" * 60)

    model.freeze_backbone()
    print(f"  Trainable params: {count_parameters(model):,} (of {sum(p.numel() for p in model.parameters()):,})")

    P1_EPOCHS = 20
    P1_LR = 1e-3
    P1_PATIENCE = 8

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=P1_LR, weight_decay=1e-3
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6)

    pat_counter = 0
    print(f"\n  Phase 1: no mixup (clean signal), focal loss gamma={FOCAL_GAMMA}")
    print(f"\n  {'Ep':>4} | {'TrLoss':>8} | {'TrAcc':>7} | {'VaLoss':>8} | {'VaAcc':>7} | {'LR':>9} | {'t':>5}")
    print("  " + "-" * 65)

    for epoch in range(1, P1_EPOCHS + 1):
        t0 = time.time()
        if args.legacy_binary:
            model.train()
            total_loss = 0.0
            total = 0
            for mel, tab, labels, has_c, has_w, diseases, severity in train_loader:
                mel, tab = mel.to(DEVICE), tab.to(DEVICE)
                has_c, has_w = has_c.to(DEVICE), has_w.to(DEVICE)
                optimizer.zero_grad()
                cls_logits, cr_logit, wh_logit, _, _ = model(mel, tab)
                loss_c = crackle_bce(cr_logit.squeeze(-1), has_c)
                loss_w = wheeze_bce(wh_logit.squeeze(-1), has_w)
                loss = loss_c + loss_w
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += float(loss.item()) * mel.size(0)
                total += mel.size(0)
            tr_loss = total_loss / max(total, 1)
            tr_acc = 0.0
            va_loss = 0.0
            va_acc, thr_c, thr_w = eval_binary_and_tune_thresholds(model, val_loader, DEVICE)
        else:
            tr_loss, tr_acc = train_epoch(
                model, train_loader, criterion, soft_criterion, optimizer, DEVICE,
                use_mixup=False)
            va_loss, va_acc = evaluate(model, val_loader, criterion, DEVICE)
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step(va_acc)

        marker = ""
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            pat_counter = 0
            if args.legacy_binary:
                best_thr_c, best_thr_w = thr_c, thr_w
            torch.save({
                "epoch": epoch, "phase": 1, "backbone": BACKBONE,
                "model_state_dict": model.state_dict(),
                "val_acc": va_acc, "val_loss": va_loss,
                "train_acc": tr_acc, "train_loss": tr_loss,
                "legacy_binary": bool(args.legacy_binary),
                "thr_crackle": float(best_thr_c),
                "thr_wheeze": float(best_thr_w),
            }, save_path)
            marker = " *"
        else:
            pat_counter += 1

        if args.legacy_binary:
            print(f"  {epoch:4d} | {tr_loss:8.4f} | {'-':>6} | {va_loss:8.4f} | {va_acc:6.2f}% | {lr_now:.2e} | {elapsed:4.1f}s{marker}  (thr_c={best_thr_c:.2f}, thr_w={best_thr_w:.2f})")
        else:
            print(f"  {epoch:4d} | {tr_loss:8.4f} | {tr_acc:6.2f}% | {va_loss:8.4f} | {va_acc:6.2f}% | {lr_now:.2e} | {elapsed:4.1f}s{marker}")
        if pat_counter >= P1_PATIENCE:
            print(f"\n  Phase 1 early stopping at epoch {epoch}")
            break

    print(f"\n  Phase 1 best val acc: {best_val_acc:.2f}%")

    if args.legacy_binary:
        # In legacy mode, Phase 1 is usually sufficient; keep it simple and fast.
        print("\n" + "=" * 60)
        print("LEGACY: Skipping Phase 2/3 (keeping best Phase 1 checkpoint)")
        print("=" * 60)
        best_ckpt = torch.load(save_path, map_location="cpu", weights_only=False)
        config = {
            "architecture": f"{BACKBONE}_legacy_binary",
            "backbone": BACKBONE,
            "formulation": "multi-label binary (crackle+wheeze) decoded to 4-class",
            "best_checkpoint": {
                "phase": int(best_ckpt.get("phase", -1)),
                "epoch": int(best_ckpt.get("epoch", -1)),
                "val_acc": float(best_ckpt.get("val_acc", best_val_acc)),
                "thr_crackle": float(best_ckpt.get("thr_crackle", best_thr_c)),
                "thr_wheeze": float(best_ckpt.get("thr_wheeze", best_thr_w)),
            },
            "device": str(DEVICE),
            "batch_size": BATCH_SIZE,
            "train_split": train_split,
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "crackle_pos_weight": crackle_pos_weight,
            "wheeze_pos_weight": wheeze_pos_weight,
        }
        with open(MODELS_DIR / "training_config.json", "w") as f:
            json.dump(config, f, indent=2)

        print("\n" + "=" * 60)
        print(f"TRAINING COMPLETE — Best decoded 4-class accuracy: {best_val_acc:.2f}%")
        print(f"Model saved: {save_path}")
        print("=" * 60)
        return

    # ────────────────────────────────────────────
    # PHASE 2: Partial unfreeze (top layers)
    # ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 2: Partial fine-tuning (top backbone layers)")
    print("=" * 60)

    ckpt = torch.load(save_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    model.partial_unfreeze_backbone(freeze_bn=True)
    print(f"  Trainable params: {count_parameters(model):,} (BN frozen)")

    P2_EPOCHS = 30
    P2_LR_BACKBONE = 5e-5
    P2_LR_HEADS = 2e-4
    P2_PATIENCE = 10

    param_groups = model.get_param_groups(lr_backbone=P2_LR_BACKBONE, lr_heads=P2_LR_HEADS)
    optimizer = optim.AdamW(param_groups, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6)

    pat_counter = 0
    p2_start_acc = best_val_acc
    print(f"  Backbone LR: {P2_LR_BACKBONE}, Heads LR: {P2_LR_HEADS}, mixup alpha=0.15, prob=0.25")

    print(f"\n  {'Ep':>4} | {'TrLoss':>8} | {'TrAcc':>7} | {'VaLoss':>8} | {'VaAcc':>7} | {'bkLR':>9} | {'t':>5}")
    print("  " + "-" * 65)

    for epoch in range(1, P2_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(
            model, train_loader, criterion, soft_criterion, optimizer, DEVICE,
            use_mixup=True, mixup_prob=0.25, mixup_alpha=0.15)
        va_loss, va_acc = evaluate(model, val_loader, criterion, DEVICE)
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step()

        marker = ""
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            pat_counter = 0
            torch.save({
                "epoch": epoch, "phase": 2, "backbone": BACKBONE,
                "model_state_dict": model.state_dict(),
                "val_acc": va_acc, "val_loss": va_loss,
                "train_acc": tr_acc, "train_loss": tr_loss,
            }, save_path)
            marker = " *"
        else:
            pat_counter += 1

        print(f"  {epoch:4d} | {tr_loss:8.4f} | {tr_acc:6.2f}% | {va_loss:8.4f} | {va_acc:6.2f}% | {lr_now:.2e} | {elapsed:4.1f}s{marker}")
        if pat_counter >= P2_PATIENCE:
            print(f"\n  Phase 2 early stopping at epoch {epoch}")
            break

    print(f"\n  Phase 2 best val acc: {best_val_acc:.2f}% (Δ from P1: {best_val_acc - p2_start_acc:+.2f}%)")

    # ────────────────────────────────────────────
    # PHASE 3: Full backbone unfreeze
    # ────────────────────────────────────────────
    # Phase 3 often regresses on ICBHI (small dataset + BN/stat drift).
    # Default: skip Phase 3 and keep the Phase 2 checkpoint.
    RUN_PHASE3 = False
    p3_start_acc = best_val_acc

    if RUN_PHASE3:
        print("\n" + "=" * 60)
        print("PHASE 3: Full backbone fine-tuning (very low LR)")
        print("=" * 60)

        ckpt = torch.load(save_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

        model.unfreeze_backbone(freeze_bn=True)
        print(f"  Trainable params: {count_parameters(model):,} (BN frozen)")

        # Phase 3: lower aux_weight to 0.2
        criterion_p3 = CombinedLoss(
            class_weights=class_weights,
            crackle_pos_weight=crackle_pos_weight,
            wheeze_pos_weight=wheeze_pos_weight,
            aux_weight=0.2,
            focal_gamma=FOCAL_GAMMA,
            label_smoothing=0.05,
        ).to(DEVICE)

        P3_EPOCHS = 10
        P3_LR_BACKBONE = 5e-6
        P3_LR_HEADS = 5e-5
        P3_PATIENCE = 5

        param_groups = model.get_param_groups(lr_backbone=P3_LR_BACKBONE, lr_heads=P3_LR_HEADS)
        optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=P3_EPOCHS, eta_min=1e-7)

        pat_counter = 0
        print(f"  Backbone LR: {P3_LR_BACKBONE}, Heads LR: {P3_LR_HEADS}, aux_weight=0.2")

        print(f"\n  {'Ep':>4} | {'TrLoss':>8} | {'TrAcc':>7} | {'VaLoss':>8} | {'VaAcc':>7} | {'bkLR':>9} | {'t':>5}")
        print("  " + "-" * 65)

        for epoch in range(1, P3_EPOCHS + 1):
            t0 = time.time()
            tr_loss, tr_acc = train_epoch(
                model, train_loader, criterion_p3, soft_criterion, optimizer, DEVICE,
                use_mixup=False)
            va_loss, va_acc = evaluate(model, val_loader, criterion_p3, DEVICE)
            elapsed = time.time() - t0
            lr_now = optimizer.param_groups[0]["lr"]
            scheduler.step()

            marker = ""
            if va_acc > best_val_acc:
                best_val_acc = va_acc
                pat_counter = 0
                torch.save({
                    "epoch": epoch, "phase": 3, "backbone": BACKBONE,
                    "model_state_dict": model.state_dict(),
                    "val_acc": va_acc, "val_loss": va_loss,
                    "train_acc": tr_acc, "train_loss": tr_loss,
                }, save_path)
                marker = " *"
            else:
                pat_counter += 1

            print(f"  {epoch:4d} | {tr_loss:8.4f} | {tr_acc:6.2f}% | {va_loss:8.4f} | {va_acc:6.2f}% | {lr_now:.2e} | {elapsed:4.1f}s{marker}")
            if pat_counter >= P3_PATIENCE:
                print(f"\n  Phase 3 early stopping at epoch {epoch}")
                break

        print(f"\n  Phase 3 best val acc: {best_val_acc:.2f}% (Δ from P2: {best_val_acc - p3_start_acc:+.2f}%)")
    else:
        print("\n" + "=" * 60)
        print("PHASE 3: Skipped (keeping Phase 2 checkpoint)")
        print("=" * 60)

    # ────────────────────────────────────────────
    # Save config
    # ────────────────────────────────────────────
    best_ckpt = torch.load(save_path, map_location="cpu", weights_only=False)
    config = {
        "architecture": f"{BACKBONE}_4class_transfer_learning",
        "backbone": BACKBONE,
        "formulation": "4-class Focal + auxiliary binary BCE",
        "improvements": [
            "direct_4class_head", "focal_loss_gamma=2", "label_smoothing_0.05",
            "oversample_softened_normal_0.85x", "freeze_bn_phase2",
            "longer_phases_20_30_epochs", "mixup_phase2",
            f"phase3_skipped={not RUN_PHASE3}",
        ],
        "best_checkpoint": {
            "phase": int(best_ckpt.get("phase", -1)),
            "epoch": int(best_ckpt.get("epoch", -1)),
            "val_acc": float(best_ckpt.get("val_acc", best_val_acc)),
        },
        "phase1_best_val_acc": float(p2_start_acc),
        "phase2_best_val_acc": float(p3_start_acc),
        "phase3_best_val_acc": float(best_val_acc),
        "device": str(DEVICE),
        "batch_size": BATCH_SIZE,
        "train_split": train_split,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "class_weights": class_weights,
        "crackle_pos_weight": crackle_pos_weight,
        "wheeze_pos_weight": wheeze_pos_weight,
    }
    with open(MODELS_DIR / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE — Best 4-class accuracy: {best_val_acc:.2f}%")
    print(f"Model saved: {save_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
