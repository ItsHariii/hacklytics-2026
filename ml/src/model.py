"""
Multi-Task CNN Model — Agent 3 (ML)

Dual-input (mel-spectrogram + tabular features) model for lung sound
classification with direct 4-class head + auxiliary binary detectors:
  1. class_logits: 4-class (normal/crackles/wheezes/both) — primary
  2. has_crackles: binary (P(crackles present)) — auxiliary
  3. has_wheezes:  binary (P(wheezes present)) — auxiliary
  4. Disease probability: 8 diseases (multi-label, kept for inference only)
  5. Severity score: regression (0-1, kept for inference only)

Backbone options:
  - 'cnn14': PANNs CNN14 pretrained on AudioSet (2M audio clips) → 2048-dim
  - 'resnet18': ResNet18 pretrained on ImageNet (legacy) → 512-dim

Architecture:
  Branch 1: Mel-spectrogram → CNN14/ResNet18 → projection → 512-dim
  Branch 2: Tabular features → Dense layers → 64-dim
  Fusion: Concatenate (576) → Shared dense → output heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PANNS_WEIGHTS_PATH = PROJECT_ROOT / "ml" / "models" / "Cnn14_mAP=0.431.pth"
PANNS_WEIGHTS_URL = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth"


def download_panns_weights():
    """Download PANNs CNN14 pretrained weights (~300MB) if not already cached."""
    if PANNS_WEIGHTS_PATH.exists():
        return PANNS_WEIGHTS_PATH
    print(f"  Downloading PANNs CNN14 pretrained weights (~300MB)...")
    PANNS_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.hub.download_url_to_file(PANNS_WEIGHTS_URL, str(PANNS_WEIGHTS_PATH))
    print(f"  Saved to: {PANNS_WEIGHTS_PATH}")
    return PANNS_WEIGHTS_PATH


# ──────────────────────────────────────────────
# PANNs CNN14 backbone (AudioSet-pretrained)
# ──────────────────────────────────────────────
class _ConvBlock(nn.Module):
    """Two-layer conv block used in PANNs CNN14."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, pool_size=(2, 2)):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.avg_pool2d(x, kernel_size=pool_size)
        return x


class Cnn14Backbone(nn.Module):
    """PANNs CNN14 audio backbone pretrained on AudioSet (2M clips, 527 classes).

    Accepts mel-spectrogram input (B, 1, n_mels, n_time) and outputs
    a 2048-dim embedding. The pretrained weights understand audio-specific
    patterns like transients, harmonics, and temporal modulation.
    """
    EMBED_DIM = 2048

    def __init__(self, n_mels=128, pretrained=True):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(n_mels)

        self.conv_block1 = _ConvBlock(1, 64)
        self.conv_block2 = _ConvBlock(64, 128)
        self.conv_block3 = _ConvBlock(128, 256)
        self.conv_block4 = _ConvBlock(256, 512)
        self.conv_block5 = _ConvBlock(512, 1024)
        self.conv_block6 = _ConvBlock(1024, 2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)

        if pretrained:
            self._load_pretrained(n_mels)

    def _load_pretrained(self, n_mels):
        weights_path = download_panns_weights()
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)

        own = self.state_dict()
        loaded = 0
        for name, param in state.items():
            if name.startswith(("spectrogram_extractor", "logmel_extractor",
                                "fc_audioset", "spec_augmenter")):
                continue
            if name.startswith("bn0") and (len(param.shape) == 0 or param.shape[0] != n_mels):
                continue
            if name in own and own[name].shape == param.shape:
                own[name].copy_(param)
                loaded += 1

        self.load_state_dict(own)
        print(f"    Loaded {loaded} pretrained PANNs CNN14 parameters")

    def forward(self, x):
        # Our input: (B, 1, n_mels, n_time) normalized to [0, 1]
        # Restore dB scale: [0, 1] → [-80, 0] dB (matches PANNs training range)
        x = x * 80.0 - 80.0

        # PANNs convention: (B, 1, n_time, n_mels)
        x = x.transpose(2, 3)

        # Frequency-axis batch normalization
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = F.dropout(self.conv_block1(x, pool_size=(2, 2)), p=0.2, training=self.training)
        x = F.dropout(self.conv_block2(x, pool_size=(2, 2)), p=0.2, training=self.training)
        x = F.dropout(self.conv_block3(x, pool_size=(2, 2)), p=0.2, training=self.training)
        x = F.dropout(self.conv_block4(x, pool_size=(2, 2)), p=0.2, training=self.training)
        x = F.dropout(self.conv_block5(x, pool_size=(2, 2)), p=0.2, training=self.training)
        x = F.dropout(self.conv_block6(x, pool_size=(1, 1)), p=0.2, training=self.training)

        # Global pooling: mean over freq, then max+mean over time
        x = torch.mean(x, dim=3)
        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2

        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)

        return x  # (B, 2048)


# ──────────────────────────────────────────────
# ResNet18 backbone (ImageNet-pretrained, legacy)
# ──────────────────────────────────────────────
class SpectrogramBranch(nn.Module):
    """ResNet18 backbone for mel-spectrogram input (legacy).
    Outputs a 512-dim feature vector."""

    EMBED_DIM = 512

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        orig_conv1 = backbone.conv1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                self.conv1.weight.copy_(orig_conv1.weight.mean(dim=1, keepdim=True))

        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


# ──────────────────────────────────────────────
# Tabular branch
# ──────────────────────────────────────────────
class TabularBranch(nn.Module):
    """Dense branch for tabular features."""
    def __init__(self, n_features=53):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        return self.layers(x)


# ──────────────────────────────────────────────
# Main model
# ──────────────────────────────────────────────
class RespiLensModel(nn.Module):
    """
    Multi-label model for lung sound analysis with transfer learning.

    Inputs:
        mel: (B, 1, 128, 128)  -- mel-spectrogram
        tab: (B, 53)           -- tabular features

    Outputs:
        class_logits:   (B, 4)  -- 4-class classification logits (primary)
        crackle_logit:  (B, 1)  -- P(has crackles) logit
        wheeze_logit:   (B, 1)  -- P(has wheezes) logit
        disease_logits: (B, 8)  -- disease probability logits
        severity:       (B, 1)  -- severity score (0-1)
    """

    N_DISEASES = 8

    def __init__(self, n_tabular=53, pretrained: bool = True, backbone: str = "cnn14"):
        super().__init__()
        self.backbone_type = backbone

        if backbone == "cnn14":
            self.spec_branch = Cnn14Backbone(n_mels=128, pretrained=pretrained)
            self.spec_proj = nn.Sequential(
                nn.Linear(Cnn14Backbone.EMBED_DIM, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
            )
        else:
            self.spec_branch = SpectrogramBranch(pretrained=pretrained)
            self.spec_proj = nn.Identity()

        self.tab_branch = TabularBranch(n_tabular)

        # Fusion: 512 (projected spec) + 64 (tab) = 576
        self.fusion = nn.Sequential(
            nn.Linear(576, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        # Primary 4-class classification head
        self.class_head = nn.Linear(128, 4)

        # Binary detection heads (auxiliary + backward compat)
        self.crackle_head = nn.Linear(128, 1)
        self.wheeze_head = nn.Linear(128, 1)

        # Auxiliary heads (kept for inference compatibility)
        self.disease_head = nn.Linear(128, self.N_DISEASES)
        self.severity_head = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, mel, tab):
        spec_feat = self.spec_branch(mel)
        spec_feat = self.spec_proj(spec_feat)   # (B, 512)
        tab_feat = self.tab_branch(tab)         # (B, 64)

        fused = torch.cat([spec_feat, tab_feat], dim=1)  # (B, 576)
        fused = self.fusion(fused)  # (B, 128)

        class_logits = self.class_head(fused)
        crackle_logit = self.crackle_head(fused)
        wheeze_logit = self.wheeze_head(fused)
        disease_logits = self.disease_head(fused)
        severity = self.severity_head(fused)

        return class_logits, crackle_logit, wheeze_logit, disease_logits, severity

    def _collect_backbone_bn(self):
        """Collect all BatchNorm modules in spec_branch."""
        bns = []
        for m in self.spec_branch.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                bns.append(m)
        return bns

    def freeze_backbone_bn(self):
        """Keep backbone BatchNorm in eval mode during fine-tuning.
        Prevents BN statistics from drifting on small datasets."""
        self._freeze_bn = True
        for m in self._collect_backbone_bn():
            m.eval()

    def unfreeze_backbone_bn(self):
        """Restore normal train/eval behavior for backbone BN."""
        self._freeze_bn = False

    def train(self, mode=True):
        """Override to keep backbone BN in eval when freeze_bn is set."""
        super().train(mode)
        if mode and getattr(self, "_freeze_bn", False):
            for m in self._collect_backbone_bn():
                m.eval()
        return self

    def freeze_backbone(self):
        """Freeze spectrogram backbone for Phase 1 (heads only).
        For CNN14, keeps bn0 trainable so it can adapt to our input scale."""
        for param in self.spec_branch.parameters():
            param.requires_grad = False
        if self.backbone_type == "cnn14":
            for param in self.spec_branch.bn0.parameters():
                param.requires_grad = True

    def partial_unfreeze_backbone(self, freeze_bn=True):
        """Unfreeze the topmost backbone layers for Phase 2 fine-tuning.
        CNN14: bn0 + conv_block5 + conv_block6 + fc1
        ResNet18: layer3 + layer4
        If freeze_bn=True, backbone BatchNorm stays in eval mode."""
        for param in self.spec_branch.parameters():
            param.requires_grad = False

        if self.backbone_type == "cnn14":
            for module in [self.spec_branch.bn0,
                           self.spec_branch.conv_block5,
                           self.spec_branch.conv_block6,
                           self.spec_branch.fc1]:
                for param in module.parameters():
                    param.requires_grad = True
        else:
            for module in [self.spec_branch.layer3, self.spec_branch.layer4]:
                for param in module.parameters():
                    param.requires_grad = True

        if freeze_bn:
            self.freeze_backbone_bn()
        else:
            self.unfreeze_backbone_bn()

    def unfreeze_backbone(self, freeze_bn=True):
        """Unfreeze entire backbone.
        If freeze_bn=True, backbone BatchNorm stays in eval mode."""
        for param in self.spec_branch.parameters():
            param.requires_grad = True
        if freeze_bn:
            self.freeze_backbone_bn()
        else:
            self.unfreeze_backbone_bn()

    def get_param_groups(self, lr_backbone=1e-4, lr_heads=5e-4):
        """Return parameter groups with differential learning rates."""
        backbone_params = [p for p in self.spec_branch.parameters() if p.requires_grad]
        head_params = (
            list(self.tab_branch.parameters()) +
            list(self.fusion.parameters()) +
            list(self.class_head.parameters()) +
            list(self.crackle_head.parameters()) +
            list(self.wheeze_head.parameters()) +
            list(self.disease_head.parameters()) +
            list(self.severity_head.parameters())
        )
        if not isinstance(self.spec_proj, nn.Identity):
            head_params += list(self.spec_proj.parameters())

        return [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params, "lr": lr_heads},
        ]


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=" * 60)
    print("MODEL ARCHITECTURE TEST")
    print("=" * 60)

    for backbone in ["cnn14", "resnet18"]:
        print(f"\n  Backbone: {backbone}")
        model = RespiLensModel(pretrained=(backbone == "resnet18"), backbone=backbone)
        total = sum(p.numel() for p in model.parameters())
        print(f"    Total parameters:     {total:,}")
        print(f"    Trainable parameters: {count_parameters(model):,}")

        model.freeze_backbone()
        print(f"    After freeze:         {count_parameters(model):,} trainable")

        model.partial_unfreeze_backbone()
        print(f"    After partial unfreeze: {count_parameters(model):,} trainable")

        mel = torch.randn(4, 1, 128, 128)
        tab = torch.randn(4, 53)
        cls, crackle, wheeze, disease, severity = model(mel, tab)
        print(f"    Forward pass: class={cls.shape}, crackle={crackle.shape}, "
              f"wheeze={wheeze.shape}, disease={disease.shape}, severity={severity.shape}")
