#!/usr/bin/env python3
"""
export_baseline.py
Konversi app/models/baseline.pt → app/models/baseline.onnx

Menangani dua kemungkinan:
1. Model standar YOLO (tidak butuh custom modules) → langsung export
2. Model dengan custom modules (SimAM, ASPP, FPN, PANet) → register dulu

Usage:
    cd /home/skripsi2025/Documents/TA-APP/YOLO-INFERENCE
    source .venv/bin/activate
    python3 scripts/export_baseline.py
"""

import os
import sys
from pathlib import Path

# Pastikan working dir adalah root project
PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

PT_PATH   = PROJECT_ROOT / "app" / "models" / "baseline.pt"
ONNX_PATH = PROJECT_ROOT / "app" / "models" / "baseline.onnx"

if not PT_PATH.exists():
    print(f"[ERROR] File tidak ditemukan: {PT_PATH}")
    sys.exit(1)

print(f"Input : {PT_PATH}")
print(f"Output: {ONNX_PATH}")

# ── Coba load langsung (tanpa custom modules) ──────────────────────────────
from ultralytics import YOLO

def export_direct():
    print("\n[1/2] Mencoba export langsung (standard YOLO)...")
    model = YOLO(str(PT_PATH))
    model.export(
        format="onnx",
        imgsz=640,
        opset=17,
        simplify=True,
        dynamic=False,
        half=False,
        batch=1,
    )
    # Ultralytics simpan di samping .pt dengan nama yang sama
    auto_output = PT_PATH.with_suffix(".onnx")
    if auto_output.exists() and auto_output != ONNX_PATH:
        auto_output.rename(ONNX_PATH)
        print(f"[OK] Dipindah ke: {ONNX_PATH}")
    elif ONNX_PATH.exists():
        print(f"[OK] Tersimpan di: {ONNX_PATH}")
    return True


def export_with_custom_modules():
    print("\n[1/2] Mencoba export dengan custom modules (SimAM, ASPP, FPN, PANet)...")
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import ultralytics.nn.modules as ulmodules
    import ultralytics.nn.tasks as tasks
    from ultralytics.nn.modules.conv import Conv

    class SimAM(nn.Module):
        def __init__(self, c1, c2=None, e_lambda=1e-4):
            super().__init__()
            self.e_lambda = e_lambda

        def forward(self, x):
            b, c, h, w = x.size()
            n = w * h - 1
            mu = x.mean(dim=[2, 3], keepdim=True)
            sq = (x - mu) ** 2
            sigma = sq.sum(dim=[2, 3], keepdim=True) / n
            y = sq / (4 * (sigma + self.e_lambda)) + 0.5
            return x * torch.sigmoid(y)

    class ASPP(nn.Module):
        def __init__(self, c1, c2=None, rates=(1, 6, 12, 18)):
            super().__init__()
            c2 = c2 or c1
            c_mid = c2 // 4
            self.branches = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(c1, c_mid, 3, padding=r, dilation=r, bias=False),
                    nn.BatchNorm2d(c_mid), nn.SiLU(),
                ) for r in rates
            ])
            self.global_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(c1, c_mid, 1, bias=False),
                nn.BatchNorm2d(c_mid), nn.SiLU(),
            )
            self.project = Conv(c_mid * (len(rates) + 1), c2, 1)

        def forward(self, x):
            size = x.shape[-2:]
            feats = [b(x) for b in self.branches]
            g = F.interpolate(self.global_pool(x), size=size,
                              mode="bilinear", align_corners=False)
            feats.append(g)
            return self.project(torch.cat(feats, dim=1))

    class FPN(nn.Module):
        def __init__(self, c1, c2=None):
            super().__init__()
            c2 = c2 or c1
            self.lateral = Conv(c1, c2, 1)
            self.refine = Conv(c2, c2, 3)

        def forward(self, x):
            return self.refine(self.lateral(x))

    class PANet(nn.Module):
        def __init__(self, c1, c2=None):
            super().__init__()
            c2 = c2 or c1
            self.down = Conv(c1, c2, 3, 2)

        def forward(self, x):
            return self.down(x)

    ulmodules.SimAM = SimAM
    ulmodules.ASPP  = ASPP
    ulmodules.FPN   = FPN
    ulmodules.PANet = PANet
    tasks.SimAM     = SimAM
    tasks.ASPP      = ASPP
    tasks.FPN       = FPN
    tasks.PANet     = PANet

    model = YOLO(str(PT_PATH))
    model.export(
        format="onnx",
        imgsz=640,
        opset=17,
        simplify=True,
        dynamic=False,
        half=False,
        batch=1,
    )
    auto_output = PT_PATH.with_suffix(".onnx")
    if auto_output.exists() and auto_output != ONNX_PATH:
        auto_output.rename(ONNX_PATH)
        print(f"[OK] Dipindah ke: {ONNX_PATH}")
    elif ONNX_PATH.exists():
        print(f"[OK] Tersimpan di: {ONNX_PATH}")
    return True


def export_torch_direct():
    """
    Fallback paling andal: pakai torch.onnx.export langsung, bypass exporter
    Ultralytics. Dipakai jika versi Ultralytics mengirim argumen yang tidak
    didukung torch (mis. 'dynamo' di torch lama Jetson).
    """
    print("\n[1/2] Mencoba export via torch.onnx.export langsung...")
    import torch

    model = YOLO(str(PT_PATH))
    net = model.model.float().eval()  # underlying nn.Module (DetectionModel)

    # Pastikan layer Detect mengeluarkan tensor inference (bukan training tuple)
    for m in net.modules():
        if hasattr(m, "export"):
            m.export = True
        if hasattr(m, "format"):
            m.format = "onnx"

    dummy = torch.zeros(1, 3, 640, 640, dtype=torch.float32)

    with torch.no_grad():
        torch.onnx.export(
            net,
            dummy,
            str(ONNX_PATH),
            input_names=["images"],
            output_names=["output0"],
            opset_version=17,
            do_constant_folding=True,
            dynamic_axes=None,  # shape statis
        )

    # Simplify opsional (kalau onnxslim tersedia)
    try:
        import onnxslim
        import onnx
        slimmed = onnxslim.slim(onnx.load(str(ONNX_PATH)))
        onnx.save(slimmed, str(ONNX_PATH))
        print("  [OK] onnxslim simplify berhasil.")
    except Exception as e:
        print(f"  [SKIP] onnxslim tidak dijalankan: {e}")

    print(f"[OK] Tersimpan di: {ONNX_PATH}")
    return True


# ── Coba export berlapis: Ultralytics → custom modules → torch langsung ────
try:
    export_direct()
except Exception as e:
    print(f"[WARN] Export langsung gagal: {e}")
    print("       Mencoba dengan custom modules...")
    try:
        export_with_custom_modules()
    except Exception as e2:
        print(f"[WARN] Export dengan custom modules gagal: {e2}")
        print("       Mencoba torch.onnx.export langsung...")
        try:
            export_torch_direct()
        except Exception as e3:
            print(f"[ERROR] Semua metode export gagal: {e3}")
            sys.exit(1)

# ── Verifikasi output ──────────────────────────────────────────────────────
if ONNX_PATH.exists():
    size_mb = ONNX_PATH.stat().st_size / (1024 * 1024)
    print(f"\n[2/2] Verifikasi ONNX...")
    try:
        import onnx
        m = onnx.load(str(ONNX_PATH))
        onnx.checker.check_model(m)
        inputs  = [i.name for i in m.graph.input]
        outputs = [o.name for o in m.graph.output]
        print(f"  File  : {ONNX_PATH.name} ({size_mb:.1f} MB)")
        print(f"  Inputs : {inputs}")
        print(f"  Outputs: {outputs}")
        print("\nExport BERHASIL. Model siap dipakai di gui_main.py.")
    except ImportError:
        print(f"  File: {ONNX_PATH.name} ({size_mb:.1f} MB)")
        print("  (install 'onnx' untuk verifikasi graph: uv pip install onnx)")
        print("\nExport BERHASIL.")
else:
    print(f"[ERROR] File ONNX tidak ditemukan setelah export: {ONNX_PATH}")
    sys.exit(1)
