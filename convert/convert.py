# convert.py — fully self-contained

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import ultralytics.nn.modules as modules
import ultralytics.nn.tasks as tasks
from ultralytics import YOLO
from ultralytics.nn.modules.conv import Conv


# Define directly in __main__ scope — matches how they were saved
class SimAM(nn.Module):
    def __init__(self, c1, c2=None, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        mu = x.mean(dim=[2, 3], keepdim=True)
        x_minus_mu_square = (x - mu) ** 2
        sigma = x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n
        y = x_minus_mu_square / (4 * (sigma + self.e_lambda)) + 0.5
        return x * torch.sigmoid(y)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling — multi-scale context aggregation."""

    def __init__(self, c1, c2=None, rates=(1, 6, 12, 18)):
        super().__init__()
        c2 = c2 or c1
        c_mid = c2 // 4  # keep param count reasonable
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(c1, c_mid, 3, padding=r, dilation=r, bias=False),
                    nn.BatchNorm2d(c_mid),
                    nn.SiLU(),
                )
                for r in rates
            ]
        )
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, c_mid, 1, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.SiLU(),
        )
        self.project = Conv(c_mid * (len(rates) + 1), c2, 1)

    def forward(self, x):
        size = x.shape[-2:]
        feats = [b(x) for b in self.branches]
        g = F.interpolate(
            self.global_pool(x), size=size, mode="bilinear", align_corners=False
        )
        feats.append(g)
        return self.project(torch.cat(feats, dim=1))


# ── FPN (top-down feature refinement) ────────────────────────────────────────
class FPN(nn.Module):
    """Lightweight FPN refinement — 1×1 projection then 3×3 refinement."""

    def __init__(self, c1, c2=None):
        super().__init__()
        c2 = c2 or c1
        self.lateral = Conv(c1, c2, 1)
        self.refine = Conv(c2, c2, 3)

    def forward(self, x):
        return self.refine(self.lateral(x))


# ── PANet (bottom-up path augmentation) ──────────────────────────────────────
class PANet(nn.Module):
    """Lightweight PANet bottom-up path — strided conv to downsample."""

    def __init__(self, c1, c2=None):
        super().__init__()
        c2 = c2 or c1
        self.down = Conv(c1, c2, 3, 2)  # stride-2 downsample

    def forward(self, x):
        return self.down(x)


# Register into Ultralytics so YAML parse_model() works too


modules.SimAM = SimAM
modules.ASPP = ASPP
tasks.SimAM = SimAM
tasks.ASPP = ASPP

# Now safe to load


model = YOLO("best_triplehead.pt")  # load the model
model.export(
    format="onnx",
    imgsz=640,
    opset=12,
    simplify=True,
    dynamic=False,
    half=False,
    batch=1,
)
print("ONNX export complete!")

# ── Graph Optimization Pass ────────────────────────────────────────
onnx_path = "best_triplehead.onnx"
root, ext = os.path.splitext(onnx_path)
dml_path = f"{root}.ort_optimized{ext}"

try:
    from onnxruntime.transformers import optimizer as ort_optimizer

    print(f"Optimizing for GPU/CPU: {onnx_path} → {dml_path}")
    opt = ort_optimizer.optimize_model(
        onnx_path,
        model_type="bert",       # generic graph optimizer pass
        num_heads=0,
        hidden_size=0,
        optimization_options=None,
    )
    opt.save_model_to_file(dml_path)
    print(f"ORT-optimized model saved: {dml_path}")
except ImportError:
    print("[SKIP] onnxruntime.transformers not available — skipping optimization.")
    print("       Install with: pip install onnxruntime  # or onnxruntime-gpu")
    print(f"       The standard ONNX model at {onnx_path} will still work on CPU.")
except Exception as e:
    print(f"[WARN] Graph optimization failed: {e}")
    print(f"       The standard ONNX model at {onnx_path} will still work on CPU.")

print("Export complete!")
