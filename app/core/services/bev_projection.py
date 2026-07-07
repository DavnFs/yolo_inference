"""
bev_projection.py
Proyeksi BEV (Bird's Eye View) yang benar secara matematis, fully vectorized.
Mendukung visualisasi Configuration Space (Minkowski Sum) agar sinkron dengan OGM.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    import cupy as cp

    _test = cp.zeros(1, dtype=cp.float32)
    del _test
    _CUPY_OK = True
    print(f"[BEV] CuPy {cp.__version__} — GPU BEV rendering aktif (CUDA device 0)")
except Exception:
    _CUPY_OK = False
    print("[BEV] CuPy GPU test gagal — fallback ke CPU rendering")

_PLASMA_LUT_CPU: Optional[np.ndarray] = None
_PLASMA_LUT_GPU = None


def _plasma_lut_cpu() -> np.ndarray:
    global _PLASMA_LUT_CPU
    if _PLASMA_LUT_CPU is None:
        idx = np.arange(256, dtype=np.uint8)
        lut = cv2.applyColorMap(idx.reshape(-1, 1), cv2.COLORMAP_PLASMA)
        _PLASMA_LUT_CPU = lut.reshape(256, 3)
    return _PLASMA_LUT_CPU


def _plasma_lut_gpu():
    global _PLASMA_LUT_GPU
    if _CUPY_OK and _PLASMA_LUT_GPU is None:
        _PLASMA_LUT_GPU = cp.asarray(_plasma_lut_cpu())
    return _PLASMA_LUT_GPU


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


# --- KONFIGURASI CONFIGURATION SPACE (MINKOWSKI SUM) ---
_EGO_WIDTH_M = 1.8
_EGO_LENGTH_M = 2.5
_MARGIN_SAFE_M = 0.15
_DELTA_W_M = _EGO_WIDTH_M + (2 * _MARGIN_SAFE_M)  # 2.1 m
_DELTA_H_M = _EGO_LENGTH_M + (2 * _MARGIN_SAFE_M)  # 2.8 m
_CLASS_DEPTH_PRIOR_M: dict = {
    "car": 4.5,
    "truck": 7.5,
    "bus": 12.0,
    "motorcycle": 2.2,
    "bicycle": 1.8,
    "person": 0.5,
    "traffic light": 0.4,
    "stop sign": 0.3,
    "fire hydrant": 0.4,
    "bench": 1.5,
    "train": 20.0,
    "large rock": 1.0,
    "small rock": 0.5,
    "large trash": 1.0,
    "small trash": 0.4,
}
_DEFAULT_DEPTH_PRIOR_M = 2.0

_CLASS_MAX_LATERAL_W_M: dict = {
    "car": 2.5,  # actual ~1.8m + buffer
    "truck": 3.0,
    "bus": 3.0,
    "motorcycle": 1.0,
    "bicycle": 0.8,
    "person": 1.0,
    "bench": 1.2,
    "train": 3.2,
}
_DEFAULT_MAX_LATERAL_W_M = 3.0


def _get_depth_prior(label: str) -> float:
    """Return class-based forward depth (Z) prior for BEV rendering."""
    lower = label.lower()
    for key, val in _CLASS_DEPTH_PRIOR_M.items():
        if key in lower:
            return val
    return _DEFAULT_DEPTH_PRIOR_M


def _get_max_lateral_w(label: str) -> float:
    """Cap lateral width to prevent side-view objects from over-expanding the corridor."""
    lower = label.lower()
    for key, val in _CLASS_MAX_LATERAL_W_M.items():
        if key in lower:
            return val
    return _DEFAULT_MAX_LATERAL_W_M


@dataclass
class Detection3D:
    label: str
    x_m: float  # lateral [meter]
    y_m: float  # vertikal [meter]
    z_m: float  # forward depth [meter]
    w_m: float  # width fisik [meter]
    h_m: float  # length/depth fisik [meter]
    confidence: float
    hazard_level: str
    bbox_2d: Tuple[int, int, int, int]
    angle_deg: float = 0.0


HAZARD_COLORS_BEV = {
    "danger": (0, 0, 255),  # Merah
    "warning": (0, 165, 255),  # Oranye
    "safe": (0, 200, 0),  # Hijau
    "out_of_roi": (128, 128, 128),  # Abu
}


def _fit_bev_box_from_points(
    region_depth: np.ndarray,
    x1c: int,
    intrinsics: Intrinsics,
    min_points: int = 8,
    max_extent_m: float = 15.0,
):
    """Back-project valid depth pixels inside a detection's crop into BEV
    (x, z) and fit a minimum-area rectangle around them.

    Ties the drawn box to what the sensor actually measured instead of a
    fixed per-class size, and gets real orientation as a side effect.
    Returns (cx_m, cz_m, w_m, l_m, angle_deg), or None if there aren't
    enough points, or the fit is obviously garbage (a loose 2D bbox
    pulling in background pixels behind the object).
    """
    ys, xs = np.where(region_depth > 0.1)
    if ys.size < min_points:
        return None

    z_vals = region_depth[ys, xs].astype(np.float32)
    u_vals = (xs + x1c).astype(np.float32)

    X = (u_vals - intrinsics.cx) * z_vals / intrinsics.fx
    Z = z_vals
    pts = np.stack([X, Z], axis=1).astype(np.float32)

    (cx_m, cz_m), (w_m, l_m), angle = cv2.minAreaRect(pts)

    if w_m <= 0.05 or l_m <= 0.05 or max(w_m, l_m) > max_extent_m:
        return None

    return float(cx_m), float(cz_m), float(w_m), float(l_m), float(angle)


def backproject_detections(
    detections: List[dict], depth_map: np.ndarray, intrinsics: Intrinsics
) -> List[Detection3D]:
    H, W = depth_map.shape
    results = []

    for det in detections:
        bbox = det.get("bounding_box", [0, 0, 0, 0])
        x1, y1, x2, y2 = bbox

        x1c = max(0, int(x1))
        x2c = min(W, int(x2))
        y1c = max(0, int(y1))
        y2c = min(H, int(y2))
        if x2c <= x1c or y2c <= y1c:
            continue

        region = depth_map[y1c:y2c, x1c:x2c]
        valid_pixels = region[region > 0.1]
        if valid_pixels.size < 5:
            continue

        label_str = det.get("label", "unknown")
        fit = _fit_bev_box_from_points(region, x1c, intrinsics)

        if fit is not None:
            x_m, z_m, w_m, h_m, angle = fit
            w_m = min(w_m, _get_max_lateral_w(label_str))
        else:
            # Fallback for sparse/far objects: same heuristic as before,
            # better than nothing when there's not enough clean depth.
            z = float(np.percentile(valid_pixels, 50))
            w_image_px = x2c - x1c
            w_m_raw = max((w_image_px * z) / intrinsics.fx, 0.3)
            w_m = min(w_m_raw, _get_max_lateral_w(label_str))
            h_m = _get_depth_prior(label_str)
            x_m = ((x1 + x2) * 0.5 - intrinsics.cx) * z / intrinsics.fx
            z_m = z
            angle = 0.0

        v_center = (y1 + y2) * 0.5
        y_m = (v_center - intrinsics.cy) * z_m / intrinsics.fy

        results.append(
            Detection3D(
                label=label_str,
                x_m=float(x_m),
                y_m=float(y_m),
                z_m=float(z_m),
                w_m=float(w_m),
                h_m=float(h_m),
                confidence=det.get("score", det.get("confidence", 0.0)),
                hazard_level=det.get("hazard_level", "safe"),
                bbox_2d=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                angle_deg=float(angle),
            )
        )

    return results


def _draw_detections_bev(bev, detections_3d, ego_px, ego_py, ppm_x, ppm_z, W, H):
    if not detections_3d:
        return

    for det in detections_3d:
        px = ego_px + det.x_m * ppm_x
        py = ego_py - det.z_m * ppm_z
        if not (0 <= px < W and 0 <= py < H):
            continue

        color = HAZARD_COLORS_BEV.get(det.hazard_level, (128, 128, 128))
        w_px = det.w_m * ppm_x
        h_px = det.h_m * ppm_z

        rect = ((px, py), (w_px, h_px), det.angle_deg)
        box_pts = cv2.boxPoints(rect).astype(np.int32)
        cv2.polylines(bev, [box_pts], isClosed=True, color=color, thickness=2)

        cv2.circle(bev, (int(px), int(py)), 3, color, -1)

        label_str = f"{det.label[:4]} {det.z_m:.1f}m"
        cv2.putText(
            bev,
            label_str,
            (int(px) + int(w_px / 2) + 4, int(py)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.28,
            (255, 255, 255),
            1,
        )


def _draw_ego_vehicle(
    bev, ego_px, ego_py, ppm_x, ppm_z, color=(0, 220, 220), outline=(255, 255, 255)
):
    """
    Menggambar Ego Vehicle secara proporsional (Vertikal) berdasarkan skala BEV.
    Panjang (Z) > Lebar (X). Menampilkan body fisik dan Configuration Space.
    """
    # 1. Configuration Space (Margin Keselamatan Ego)
    cw = int(_DELTA_W_M * ppm_x)
    cl = int(_DELTA_H_M * ppm_z)
    cv2.rectangle(
        bev,
        (ego_px - cw // 2, ego_py - cl),
        (ego_px + cw // 2, ego_py),
        (120, 120, 120),
        1,  # Garis abu-abu tipis untuk C-Space
    )

    # 2. Body Fisik Ego
    w_px = int(_EGO_WIDTH_M * ppm_x)
    l_px = int(_EGO_LENGTH_M * ppm_z)

    x1 = ego_px - w_px // 2
    x2 = ego_px + w_px // 2
    y1 = ego_py - l_px  # Depan mobil mengarah ke atas (Z maju)
    y2 = ego_py  # Belakang mobil (titik origin/axle)

    cv2.rectangle(bev, (x1, y1), (x2, y2), color, -1)
    cv2.rectangle(bev, (x1, y1), (x2, y2), outline, 1)

    # 3. Indikator Arah Depan (Garis kap mobil)
    cv2.line(bev, (ego_px, y1), (ego_px, y1 + 8), outline, 2)


def render_bev_opencv(
    detections_3d: List[Detection3D],
    canvas_wh: Tuple[int, int] = (400, 600),
    max_range_m: float = 40.0,
    lateral_range_m: float = 15.0,
    path_waypoints: Optional[List[Tuple]] = None,
) -> np.ndarray:
    W, H = canvas_wh
    bev = np.zeros((H, W, 3), dtype=np.uint8)

    ppm_z = H / max_range_m
    ppm_x = W / (2 * lateral_range_m)

    ego_px = W // 2
    ego_py = H - 10

    for d_m in range(10, int(max_range_m) + 1, 10):
        y_grid = int(ego_py - d_m * ppm_z)
        if 0 <= y_grid < H:
            cv2.line(bev, (0, y_grid), (W, y_grid), (40, 40, 40), 1)
            cv2.putText(
                bev,
                f"{d_m}m",
                (4, y_grid - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (70, 70, 70),
                1,
            )

    cv2.line(bev, (ego_px, 0), (ego_px, H), (40, 40, 40), 1)

    _draw_ego_vehicle(bev, ego_px, ego_py, ppm_x, ppm_z, color=(0, 180, 180))

    _draw_detections_bev(bev, detections_3d, ego_px, ego_py, ppm_x, ppm_z, W, H)

    if path_waypoints and len(path_waypoints) > 1:
        for i in range(len(path_waypoints) - 1):
            pt1 = (int(path_waypoints[i][0]), int(path_waypoints[i][1]))
            pt2 = (int(path_waypoints[i + 1][0]), int(path_waypoints[i + 1][1]))
            cv2.line(bev, pt1, pt2, (0, 200, 255), 2)
        if path_waypoints:
            cv2.circle(
                bev,
                (int(path_waypoints[0][0]), int(path_waypoints[0][1])),
                5,
                (255, 255, 0),
                -1,
            )
            cv2.circle(
                bev,
                (int(path_waypoints[-1][0]), int(path_waypoints[-1][1])),
                5,
                (0, 255, 0),
                -1,
            )

    return bev


def render_dense_bev(
    depth_map: np.ndarray,
    intrinsics: Intrinsics,
    canvas_wh: Tuple[int, int] = (600, 800),
    max_range_m: float = 40.0,
    lateral_range_m: float = 15.0,
    detections_3d: Optional[List["Detection3D"]] = None,
    path_waypoints: Optional[List[Tuple]] = None,
    downsample: int = 4,
) -> np.ndarray:
    W, H = canvas_wh
    bev = np.zeros((H, W, 3), dtype=np.uint8)

    ppm_z = H / max_range_m
    ppm_x = W / (2 * lateral_range_m)
    ego_px = W // 2
    ego_py = H - 10

    for d_m in range(5, int(max_range_m) + 1, 5):
        y_grid = int(ego_py - d_m * ppm_z)
        if 0 <= y_grid < H:
            lw = 1 if d_m % 10 != 0 else 1
            color = (30, 30, 30) if d_m % 10 != 0 else (50, 50, 50)
            cv2.line(bev, (0, y_grid), (W, y_grid), color, lw)
            if d_m % 10 == 0:
                cv2.putText(
                    bev,
                    f"{d_m}m",
                    (4, y_grid - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.28,
                    (70, 70, 70),
                    1,
                )
    cv2.line(bev, (ego_px, 0), (ego_px, H), (45, 45, 45), 1)

    img_h, img_w = depth_map.shape
    gpu_used = False
    if _CUPY_OK:
        try:
            depth_gpu = cp.asarray(depth_map)
            vs_g = cp.arange(0, img_h, 1, dtype=cp.int32)
            us_g = cp.arange(0, img_w, 1, dtype=cp.int32)
            uu_g, vv_g = cp.meshgrid(us_g, vs_g)
            uu_g = uu_g.ravel()
            vv_g = vv_g.ravel()

            Z_g = depth_gpu[vv_g, uu_g]
            valid_g = (Z_g > 0.3) & (Z_g <= max_range_m)
            Z_g = Z_g[valid_g]
            uu_g = uu_g[valid_g].astype(cp.float32)

            if Z_g.size > 0:
                X_g = (uu_g - intrinsics.cx) * Z_g / intrinsics.fx
                px_g = (cp.int32(ego_px) + (X_g * ppm_x)).astype(cp.int32)
                py_g = (cp.int32(ego_py) - (Z_g * ppm_z)).astype(cp.int32)

                z_u8 = (255 - cp.clip(Z_g / max_range_m, 0.0, 1.0) * 255).astype(
                    cp.uint8
                )
                cols = _plasma_lut_gpu()[z_u8]

                in_b = (px_g >= 0) & (px_g < W) & (py_g >= 0) & (py_g < H)
                bev_g = cp.asarray(bev)
                bev_g[py_g[in_b], px_g[in_b]] = cols[in_b]
                bev[:] = cp.asnumpy(bev_g)

            gpu_used = True
        except Exception as _gpu_err:
            print(f"[BEV] GPU error frame — fallback CPU: {_gpu_err}")

    if not gpu_used:
        vs = np.arange(0, img_h, downsample)
        us = np.arange(0, img_w, downsample)
        uu, vv = np.meshgrid(us, vs)
        uu = uu.ravel().astype(np.float32)
        vv = vv.ravel().astype(np.float32)

        Z = depth_map[vv.astype(np.int32), uu.astype(np.int32)]
        valid = (Z > 0.3) & (Z <= max_range_m)
        Z = Z[valid]
        uu = uu[valid]

        if Z.size > 0:
            X = (uu - intrinsics.cx) * Z / intrinsics.fx
            px = (ego_px + X * ppm_x).astype(np.int32)
            py = (ego_py - Z * ppm_z).astype(np.int32)

            z_uint8 = (255 - np.clip(Z / max_range_m, 0.0, 1.0) * 255).astype(np.uint8)
            colors_bgr = _plasma_lut_cpu()[z_uint8]

            in_b = (px >= 0) & (px < W) & (py >= 0) & (py < H)
            bev[py[in_b], px[in_b]] = colors_bgr[in_b]

    _draw_ego_vehicle(bev, ego_px, ego_py, ppm_x, ppm_z, color=(0, 220, 220))

    _draw_detections_bev(bev, detections_3d, ego_px, ego_py, ppm_x, ppm_z, W, H)

    if path_waypoints and len(path_waypoints) > 1:
        for i in range(len(path_waypoints) - 1):
            pt1 = (int(path_waypoints[i][0]), int(path_waypoints[i][1]))
            pt2 = (int(path_waypoints[i + 1][0]), int(path_waypoints[i + 1][1]))
            cv2.line(bev, pt1, pt2, (0, 220, 255), 2)
        cv2.circle(
            bev,
            (int(path_waypoints[0][0]), int(path_waypoints[0][1])),
            5,
            (255, 255, 0),
            -1,
        )
        cv2.circle(
            bev,
            (int(path_waypoints[-1][0]), int(path_waypoints[-1][1])),
            5,
            (0, 255, 100),
            -1,
        )

    return bev


def intrinsics_from_calib(calib: dict) -> Intrinsics:
    return Intrinsics(
        fx=calib["fx"],
        fy=calib["fy"],
        cx=calib["cx"],
        cy=calib["cy"],
    )


def intrinsics_from_frame_width(
    orig_w: int, stereo_fx: float = 721.53, native_w: int = 1242
) -> Intrinsics:
    from app.core.config import get_scaled_intrinsics

    scaled = get_scaled_intrinsics(orig_w)
    return Intrinsics(
        fx=scaled["fx"], fy=scaled["fy"], cx=scaled["cx"], cy=scaled["cy"]
    )
