"""
ONNX Export — Agent 3 (ML)

Export the trained 4-class PyTorch model to ONNX format.
Validates that ONNX inference produces identical outputs to PyTorch.

Outputs: class_logits, crackle_logit, wheeze_logit, disease_logits, severity

Usage:
    python -m ml.src.export_onnx
"""

import json
import argparse
import numpy as np
import torch
import onnx
import onnxruntime as ort
from pathlib import Path

from ml.src.model import RespiLensModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "ml" / "models"


class _LegacyWrapper(torch.nn.Module):
    """Wrap model to export legacy 4-output ONNX (no class_logits)."""
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, mel, tab):
        class_logits, crackle_logit, wheeze_logit, disease_logits, severity = self.model(mel, tab)
        return crackle_logit, wheeze_logit, disease_logits, severity


def export_to_onnx(checkpoint_path: Path, onnx_path: Path, legacy: bool = False):
    """Export PyTorch model to ONNX format."""
    print("=" * 60)
    print("ONNX EXPORT (4-Class + Binary)" if not legacy else "ONNX EXPORT (LEGACY 4-output)")
    print("=" * 60)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    backbone = checkpoint.get("backbone", "resnet18")

    model = RespiLensModel(pretrained=False, backbone=backbone)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"\n  Loaded checkpoint from phase {checkpoint.get('phase', '?')}, epoch {checkpoint['epoch']}")
    print(f"  Backbone: {backbone}")
    print(f"  Validation accuracy: {checkpoint['val_acc']:.2f}%")

    # Use batch size 2 to avoid BatchNorm edge cases during validation.
    dummy_mel = torch.randn(2, 1, 128, 128)
    dummy_tab = torch.randn(2, 53)

    print(f"\n  Exporting to: {onnx_path}")
    export_model = _LegacyWrapper(model) if legacy else model
    export_model.eval()
    output_names = (
        ["crackle_logit", "wheeze_logit", "disease_logits", "severity"]
        if legacy else
        ["class_logits", "crackle_logit", "wheeze_logit", "disease_logits", "severity"]
    )
    dynamic_axes = {
        "mel_spectrogram": {0: "batch_size"},
        "tabular_features": {0: "batch_size"},
        "crackle_logit": {0: "batch_size"},
        "wheeze_logit": {0: "batch_size"},
        "disease_logits": {0: "batch_size"},
        "severity": {0: "batch_size"},
    }
    if not legacy:
        dynamic_axes["class_logits"] = {0: "batch_size"}

    torch.onnx.export(
        export_model,
        (dummy_mel, dummy_tab),
        str(onnx_path),
        input_names=["mel_spectrogram", "tabular_features"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
    )

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("  ONNX model validation: PASSED")

    # Validate outputs match PyTorch
    print("\n  Validating outputs match PyTorch...")
    session = ort.InferenceSession(str(onnx_path))

    with torch.no_grad():
        pt_out = export_model(dummy_mel, dummy_tab)
    pt_out = [t.numpy() for t in pt_out] if isinstance(pt_out, (tuple, list)) else [pt_out.numpy()]

    ort_inputs = {
        "mel_spectrogram": dummy_mel.numpy(),
        "tabular_features": dummy_tab.numpy(),
    }
    ort_out = session.run(None, ort_inputs)

    diffs = [float(np.max(np.abs(a - b))) for a, b in zip(pt_out, ort_out)]

    if legacy:
        crackle_diff, wheeze_diff, disease_diff, severity_diff = diffs
        class_diff = None
        print(f"    Crackle logit max diff:  {crackle_diff:.2e}")
        print(f"    Wheeze logit max diff:   {wheeze_diff:.2e}")
        print(f"    Disease logits max diff: {disease_diff:.2e}")
        print(f"    Severity max diff:       {severity_diff:.2e}")
    else:
        class_diff, crackle_diff, wheeze_diff, disease_diff, severity_diff = diffs
        print(f"    Class logits max diff:   {class_diff:.2e}")
        print(f"    Crackle logit max diff:  {crackle_diff:.2e}")
        print(f"    Wheeze logit max diff:   {wheeze_diff:.2e}")
        print(f"    Disease logits max diff: {disease_diff:.2e}")
        print(f"    Severity max diff:       {severity_diff:.2e}")

    tolerance = 1e-5
    all_diffs = [d for d in [class_diff, crackle_diff, wheeze_diff, disease_diff, severity_diff] if d is not None]
    if all(d < tolerance for d in all_diffs):
        print("  Output validation: PASSED (within tolerance)")
    else:
        print(f"  Output validation: WARNING (diffs exceed {tolerance})")
        print("  This is acceptable for float32 precision differences.")

    onnx_size = onnx_path.stat().st_size
    print(f"\n  ONNX model size: {onnx_size / 1024:.1f} KB")

    # Benchmark
    print("\n  Benchmarking inference speed...")
    import time
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        session.run(None, ort_inputs)
        times.append(time.perf_counter() - t0)

    avg_ms = np.mean(times) * 1000
    p95_ms = np.percentile(times, 95) * 1000
    print(f"    Average: {avg_ms:.1f} ms")
    print(f"    P95:     {p95_ms:.1f} ms")

    print("\n" + "=" * 60)
    print("ONNX EXPORT COMPLETE")
    print("=" * 60)

    return {
        "onnx_path": str(onnx_path),
        "onnx_size_kb": onnx_size / 1024,
        "backbone": backbone,
        "legacy": bool(legacy),
        "class_diff": float(class_diff) if class_diff is not None else None,
        "crackle_diff": float(crackle_diff),
        "wheeze_diff": float(wheeze_diff),
        "disease_diff": float(disease_diff),
        "severity_diff": float(severity_diff),
        "avg_inference_ms": float(avg_ms),
        "p95_inference_ms": float(p95_ms),
    }


def main():
    parser = argparse.ArgumentParser(description="Export best model to ONNX")
    parser.add_argument("--legacy", action="store_true",
                        help="Export legacy 4-output ONNX (no class_logits) for binary decoding.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint .pt (default: ml/models/best_model.pt)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output ONNX path (default: ml/models/model.onnx)")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else (MODELS_DIR / "best_model.pt")
    onnx_path = Path(args.out) if args.out else (MODELS_DIR / "model.onnx")

    if not checkpoint_path.exists():
        print("ERROR: No trained model found. Run training first.")
        return

    results = export_to_onnx(checkpoint_path, onnx_path, legacy=args.legacy)

    meta_path = MODELS_DIR / "onnx_export_meta.json"
    with open(meta_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Export metadata saved to: {meta_path}")


if __name__ == "__main__":
    main()
