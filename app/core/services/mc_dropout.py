import os
import sys
import threading
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv


PT_MODEL_CACHE = {}
PT_MODEL_ERRORS = {}
PT_MODEL_LOCK = threading.Lock()
DROPOUT_TARGET_INDICES = (18, 21, 24)


class SimAM(nn.Module):
    def __init__(self, c1, c2=None, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x):
        _, _, h, w = x.size()
        n = w * h - 1
        mu = x.mean(dim=[2, 3], keepdim=True)
        x_minus_mu_square = (x - mu) ** 2
        sigma = x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n
        y = x_minus_mu_square / (4 * (sigma + self.e_lambda)) + 0.5
        return x * torch.sigmoid(y)


class ASPP(nn.Module):
    def __init__(self, c1, c2=None, rates=(1, 6, 12, 18)):
        super().__init__()
        c2 = c2 or c1
        c_mid = c2 // 4
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


def register_custom_yolo_modules():
    import ultralytics.nn.modules as modules
    import ultralytics.nn.tasks as tasks

    custom_modules = {
        "SimAM": SimAM,
        "ASPP": ASPP,
        "FPN": FPN,
        "PANet": PANet,
    }

    main_module = sys.modules.get("__main__")
    for name, cls in custom_modules.items():
        setattr(modules, name, cls)
        setattr(tasks, name, cls)
        if main_module is not None:
            setattr(main_module, name, cls)


def _get_network(model: Any) -> nn.Module:
    if hasattr(model, "model") and hasattr(model.model, "model"):
        return model.model
    return model


def _get_layers(model: Any):
    network = _get_network(model)
    layers = getattr(network, "model", None)
    if layers is None:
        raise AttributeError("PyTorch YOLO model has no model.model layer list.")
    return network, layers


def _copy_ultralytics_attrs(src: nn.Module, dst: nn.Module):
    for attr in ("i", "f", "type", "np"):
        if hasattr(src, attr):
            setattr(dst, attr, getattr(src, attr))


def inject_mc_dropout(model, dropout_p=0.25):
    """
    Injects nn.Dropout after layers 18, 21, 24 in model.model.
    Modifies model IN-PLACE.
    """
    network, layers = _get_layers(model)
    injected = []

    for target_idx in DROPOUT_TARGET_INDICES:
        if target_idx >= len(layers):
            continue

        original = layers[target_idx]
        has_dropout = any(isinstance(m, nn.Dropout) for m in original.modules())
        if has_dropout:
            wrapper = original
        else:
            wrapper = nn.Sequential(original, nn.Dropout(p=dropout_p))
            _copy_ultralytics_attrs(original, wrapper)
            layers[target_idx] = wrapper

        injected.append(f"model.model.{target_idx}")

    network.eval()
    for target_idx in DROPOUT_TARGET_INDICES:
        if target_idx >= len(layers):
            continue
        for module in layers[target_idx].modules():
            if isinstance(module, nn.Dropout):
                module.train()

    return injected


def verify_mc_dropout(model) -> str:
    """Returns a human-readable string listing Dropout layers and train modes."""
    _, layers = _get_layers(model)
    lines = []
    for idx, layer in enumerate(layers):
        dropouts = [m for m in layer.modules() if isinstance(m, nn.Dropout)]
        for drop_idx, dropout in enumerate(dropouts):
            lines.append(
                f"model.model.{idx}.Dropout[{drop_idx}] "
                f"p={dropout.p} training={dropout.training}"
            )
    return "\n".join(lines) if lines else "No Dropout layers found."


def load_pt_model(pt_path: str) -> Tuple[Any, bool]:
    """
    Loads YOLO .pt model using Ultralytics and injects MC Dropout once.
    Returns (model, is_loaded).
    """
    if not pt_path:
        return None, False

    cache_key = os.path.abspath(pt_path)
    with PT_MODEL_LOCK:
        if cache_key in PT_MODEL_CACHE:
            return PT_MODEL_CACHE[cache_key], True
        if cache_key in PT_MODEL_ERRORS:
            return None, False

        try:
            from ultralytics import YOLO

            register_custom_yolo_modules()
            model = YOLO(cache_key)
            injected = inject_mc_dropout(model.model)
            print(f"[MC Dropout] Loaded PT model: {cache_key}")
            print(f"[MC Dropout] Injected layers: {', '.join(injected) or 'none'}")
            print("[MC Dropout] Verification:")
            print(verify_mc_dropout(model.model))
            PT_MODEL_CACHE[cache_key] = model
            return model, True
        except Exception as e:
            PT_MODEL_ERRORS[cache_key] = str(e)
            print(f"[MC Dropout] Failed to load PT model '{cache_key}': {e}")
            return None, False


def _model_device(model: Any) -> torch.device:
    network = _get_network(model)
    try:
        return next(network.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _labels_from_model(model: Any) -> List[str]:
    names = getattr(model, "names", None) or getattr(_get_network(model), "names", None)
    if isinstance(names, dict):
        return [str(names[i]) for i in sorted(names)]
    if isinstance(names, (list, tuple)):
        return [str(name) for name in names]

    labels_path = os.path.join(os.getcwd(), "app", "models", "labels.txt")
    if os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    return []


def _select_layer_input(f, x, y):
    if f == -1:
        return x
    if isinstance(f, int):
        return y[f]
    return [x if j == -1 else y[j] for j in f]


def _forward_layers(layers, x, y, start_idx: int, end_idx: int):
    for idx in range(start_idx, end_idx):
        layer = layers[idx]
        layer_input = _select_layer_input(getattr(layer, "f", -1), x, y)
        x = layer(layer_input)
        y.append(x)
    return x, y


def _extract_prediction_tensor(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, (list, tuple)):
        for item in output:
            try:
                tensor = _extract_prediction_tensor(item)
                if tensor is not None and tensor.ndim >= 2:
                    return tensor
            except ValueError:
                continue
    raise ValueError("Could not extract YOLO prediction tensor from MC output.")


def _box_iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def _group_detections(sample_detections: List[List[Dict]]) -> List[List[Dict]]:
    groups = []
    for detections in sample_detections:
        for det in detections:
            best_iou = 0.0
            best_group = None
            for group in groups:
                if group[0]["label"] != det["label"]:
                    continue
                group_box = np.mean([g["bounding_box"] for g in group], axis=0)
                iou = _box_iou(det["bounding_box"], group_box)
                if iou >= 0.5 and iou > best_iou:
                    best_iou = iou
                    best_group = group
            if best_group is None:
                groups.append([det])
            else:
                best_group.append(det)
    return groups


def _final_nms(detections: List[Dict]) -> List[Dict]:
    if not detections:
        return detections

    from app.core.services.object_detection import YOLO_CONF_THRESH, YOLO_NMS_THRESH

    boxes_nms = []
    scores = []
    for det in detections:
        x1, y1, x2, y2 = det["bounding_box"]
        boxes_nms.append([x1, y1, max(0, x2 - x1), max(0, y2 - y1)])
        scores.append(float(det["score"]))

    indices = cv2.dnn.NMSBoxes(boxes_nms, scores, YOLO_CONF_THRESH, YOLO_NMS_THRESH)
    if len(indices) == 0:
        return []
    return [detections[int(i)] for i in np.asarray(indices).reshape(-1)]


def mc_inference(model, image_bgr: np.ndarray, n_samples=5) -> Dict:
    """
    Runs N stochastic forward passes using PyTorch model.
    Returns dict matching existing detection format.
    """
    from app.core.services.object_detection import postprocess_yolo, preprocess_yolo

    network, layers = _get_layers(model)
    network.eval()
    for idx in DROPOUT_TARGET_INDICES:
        if idx < len(layers):
            for module in layers[idx].modules():
                if isinstance(module, nn.Dropout):
                    module.train()

    input_np, scale, dw, dh, orig_w, orig_h = preprocess_yolo(image_bgr)
    input_tensor = torch.from_numpy(input_np).to(_model_device(model)).float()
    labels = _labels_from_model(model)
    n_samples = max(1, int(n_samples))

    with torch.no_grad():
        cached_x, cached_y = _forward_layers(layers, input_tensor, [], 0, 18)

    all_detections = []
    for _ in range(n_samples):
        sample_y = list(cached_y)
        with torch.no_grad():
            out, _ = _forward_layers(layers, cached_x, sample_y, 18, len(layers))
            pred = _extract_prediction_tensor(out).detach().cpu().numpy()
        all_detections.append(
            postprocess_yolo(pred, scale, dw, dh, labels, orig_w, orig_h)
        )

    mean_detections = []
    for group in _group_detections(all_detections):
        boxes = np.array([det["bounding_box"] for det in group], dtype=np.float32)
        scores = np.array([det["score"] for det in group], dtype=np.float32)
        mean_box = np.mean(boxes, axis=0)
        mean_detections.append({
            "label": group[0]["label"],
            "score": round(float(np.mean(scores)), 4),
            "bounding_box": [int(round(v)) for v in mean_box],
            "hazard_level": "safe",
            "distance_m": 0.0,
            "uncertainty": round(float(np.var(scores)), 6),
            "bbox_std": round(float(np.mean(np.std(boxes, axis=0))), 4),
        })

    return {
        "detections": _final_nms(mean_detections),
        "mc_samples": n_samples,
    }
