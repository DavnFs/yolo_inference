"""
bev_projection.py
Proyeksi BEV (Bird's Eye View) yang benar secara matematis, fully vectorized.
Upgrade dari _flush_bev_update() di gui_main.py.

Derivasi:
    Model kamera pinhole:
        u = fx * (X/Z) + cx
        v = fy * (Y/Z) + cy

    Inverse (back-projection dari 2D + depth ke 3D):
        X = (u - cx) * Z / fx   [lateral, kiri negatif, kanan positif]
        Y = (v - cy) * Z / fy   [vertikal, atas negatif]
        Z = depth_map[v, u]     [forward, positif]

    Untuk BEV, buang Y (vertikal) dan plot di bidang XZ:
        X = lateral (kiri-kanan)
        Z = depth (ke depan)
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    import cupy as cp
    # Verify actual GPU op works (Jetson may have arch mismatch)
    _test = cp.zeros(1, dtype=cp.float32)
    del _test
    _CUPY_OK = True
    print(f"[BEV] CuPy {cp.__version__} — GPU BEV rendering aktif (CUDA device 0)")
except Exception:
    _CUPY_OK = False
    print("[BEV] CuPy GPU test gagal — fallback ke CPU rendering")

# Pre-computed PLASMA LUT: shape (256, 3) BGR, built once on first call
_PLASMA_LUT_CPU: Optional[np.ndarray] = None
_PLASMA_LUT_GPU = None  # cp.ndarray when CuPy available


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


@dataclass
class Detection3D:
    label: str
    x_m: float       # lateral [meter], kiri negatif
    y_m: float       # vertikal [meter], atas negatif (diabaikan di BEV)
    z_m: float       # forward depth [meter]
    confidence: float
    hazard_level: str
    bbox_2d: Tuple[int, int, int, int]  # x1, y1, x2, y2 piksel


HAZARD_COLORS_BEV = {
    'danger':     (0,   0,   255),   # Merah
    'warning':    (0,   165, 255),   # Oranye
    'safe':       (0,   200, 0),     # Hijau
    'out_of_roi': (128, 128, 128),   # Abu
}


def backproject_detections(
    detections: List[dict],
    depth_map: np.ndarray,
    intrinsics: Intrinsics
) -> List[Detection3D]:
    """
    Konversi list detections ke koordinat 3D menggunakan depth map.

    Gunakan MEDIAN depth dari region bbox (bukan titik tengah) karena:
    - Titik tengah bbox bisa jatuh di kaca atau area oklusi
    - Median robust terhadap noise di border dan surface reflektif
    - NumPy median O(n log n) masih cepat untuk patch kecil
    """
    H, W = depth_map.shape
    results = []

    u_centers, v_centers, depths, valid_dets = [], [], [], []

    for det in detections:
        bbox = det.get('bounding_box', [0, 0, 0, 0])
        x1, y1, x2, y2 = bbox

        x1c = max(0, int(x1)); x2c = min(W, int(x2))
        y1c = max(0, int(y1)); y2c = min(H, int(y2))

        if x2c <= x1c or y2c <= y1c:
            continue

        region = depth_map[y1c:y2c, x1c:x2c]
        valid_pixels = region[region > 0.1]

        if valid_pixels.size < 5:
            continue

        # Percentile 20 lebih konservatif — mendekati permukaan bodi kendaraan
        z = float(np.percentile(valid_pixels, 20))

        u_centers.append((x1 + x2) * 0.5)
        v_centers.append((y1 + y2) * 0.5)
        depths.append(z)
        valid_dets.append(det)

    if not valid_dets:
        return []

    # Vectorized back-projection
    u_arr = np.array(u_centers, dtype=np.float32)
    v_arr = np.array(v_centers, dtype=np.float32)
    z_arr = np.array(depths,    dtype=np.float32)

    X_arr = (u_arr - intrinsics.cx) * z_arr / intrinsics.fx
    Y_arr = (v_arr - intrinsics.cy) * z_arr / intrinsics.fy

    for i, det in enumerate(valid_dets):
        bbox = det.get('bounding_box', [0, 0, 0, 0])
        results.append(Detection3D(
            label=det.get('label', 'unknown'),
            x_m=float(X_arr[i]),
            y_m=float(Y_arr[i]),
            z_m=float(z_arr[i]),
            confidence=det.get('score', det.get('confidence', 0.0)),
            hazard_level=det.get('hazard_level', 'safe'),
            bbox_2d=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        ))

    return results


def render_bev_opencv(
    detections_3d: List[Detection3D],
    canvas_wh: Tuple[int, int] = (400, 600),
    max_range_m: float = 40.0,
    lateral_range_m: float = 15.0,
    path_waypoints: Optional[List[Tuple]] = None
) -> np.ndarray:
    """
    Render BEV grid ke OpenCV image.

    Sistem koordinat canvas:
    - Ego vehicle di BAWAH-TENGAH canvas
    - Z (depth, ke depan) = arah ke ATAS canvas
    - X (lateral, kanan) = arah ke KANAN canvas
    """
    W, H = canvas_wh
    bev = np.zeros((H, W, 3), dtype=np.uint8)

    ppm_z = H / max_range_m
    ppm_x = W / (2 * lateral_range_m)

    # Ego origin (pixel)
    ego_px = W // 2
    ego_py = H - 10

    # Grid lines setiap 10m
    for d_m in range(10, int(max_range_m) + 1, 10):
        y_grid = int(ego_py - d_m * ppm_z)
        if 0 <= y_grid < H:
            cv2.line(bev, (0, y_grid), (W, y_grid), (40, 40, 40), 1)
            cv2.putText(bev, f'{d_m}m', (4, y_grid - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (70, 70, 70), 1)

    # Ego center line
    cv2.line(bev, (ego_px, 0), (ego_px, H), (40, 40, 40), 1)

    # Ego vehicle marker (cyan rectangle)
    cv2.rectangle(bev,
                  (ego_px - 8, ego_py - 15),
                  (ego_px + 8, ego_py),
                  (0, 180, 180), -1)

    # Plot detections (vectorized coordinate conversion)
    if detections_3d:
        x_m_arr = np.array([d.x_m for d in detections_3d], dtype=np.float32)
        z_m_arr = np.array([d.z_m for d in detections_3d], dtype=np.float32)

        px_arr = (ego_px + x_m_arr * ppm_x).astype(np.int32)
        py_arr = (ego_py - z_m_arr * ppm_z).astype(np.int32)

        in_bounds = (px_arr >= 0) & (px_arr < W) & (py_arr >= 0) & (py_arr < H)

        for i, det in enumerate(detections_3d):
            if not in_bounds[i]:
                continue

            px, py = int(px_arr[i]), int(py_arr[i])
            color = HAZARD_COLORS_BEV.get(det.hazard_level, (128, 128, 128))
            radius = max(4, int(det.confidence * 12))

            cv2.circle(bev, (px, py), radius, color, -1)
            cv2.circle(bev, (px, py), radius + 2, color, 1)

            label_str = f"{det.label[:4]} {det.z_m:.1f}m"
            cv2.putText(bev, label_str, (px + 6, py + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1)

    # Path waypoints (format: list of (x_bev_px, y_bev_px) dalam BEV canvas space)
    if path_waypoints and len(path_waypoints) > 1:
        for i in range(len(path_waypoints) - 1):
            pt1 = (int(path_waypoints[i][0]), int(path_waypoints[i][1]))
            pt2 = (int(path_waypoints[i+1][0]), int(path_waypoints[i+1][1]))
            cv2.line(bev, pt1, pt2, (0, 200, 255), 2)
        if path_waypoints:
            cv2.circle(bev, (int(path_waypoints[0][0]), int(path_waypoints[0][1])),
                       5, (255, 255, 0), -1)
            cv2.circle(bev, (int(path_waypoints[-1][0]), int(path_waypoints[-1][1])),
                       5, (0, 255, 0), -1)

    return bev


def render_dense_bev(
    depth_map: np.ndarray,
    intrinsics: Intrinsics,
    canvas_wh: Tuple[int, int] = (600, 800),
    max_range_m: float = 40.0,
    lateral_range_m: float = 15.0,
    detections_3d: Optional[List['Detection3D']] = None,
    path_waypoints: Optional[List[Tuple]] = None,
    downsample: int = 4,
) -> np.ndarray:
    """
    Render dense point cloud BEV dari depth map.

    Setiap piksel depth map yang valid diproyeksikan ke bidang XZ:
        X = (u - cx) * Z / fx   [lateral]
        Z = depth_map[v, u]     [forward]
    Diwarnai berdasarkan jarak Z menggunakan PLASMA colormap.

    Detection dots dan path waypoints di-overlay di atasnya.

    Args:
        depth_map    : (H, W) float32, meter
        intrinsics   : camera intrinsics
        canvas_wh    : (width, height) canvas output
        max_range_m  : jarak maksimum yang dirender [m]
        lateral_range_m: jangkauan lateral kiri+kanan [m]
        detections_3d: list Detection3D untuk overlay dot
        path_waypoints: list (px, py) dalam canvas space
        downsample   : ambil 1 piksel dari setiap NxN blok (hemat CPU)
    """
    W, H = canvas_wh
    bev = np.zeros((H, W, 3), dtype=np.uint8)

    ppm_z = H / max_range_m
    ppm_x = W / (2 * lateral_range_m)
    ego_px = W // 2
    ego_py = H - 10

    # ── Grid lines ──────────────────────────────────────────────────────────
    for d_m in range(5, int(max_range_m) + 1, 5):
        y_grid = int(ego_py - d_m * ppm_z)
        if 0 <= y_grid < H:
            lw = 1 if d_m % 10 != 0 else 1
            color = (30, 30, 30) if d_m % 10 != 0 else (50, 50, 50)
            cv2.line(bev, (0, y_grid), (W, y_grid), color, lw)
            if d_m % 10 == 0:
                cv2.putText(bev, f'{d_m}m', (4, y_grid - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, (70, 70, 70), 1)
    cv2.line(bev, (ego_px, 0), (ego_px, H), (45, 45, 45), 1)

    # ── Dense point cloud ────────────────────────────────────────────────────
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
            Z_g  = Z_g[valid_g]
            uu_g = uu_g[valid_g].astype(cp.float32)

            if Z_g.size > 0:
                X_g  = (uu_g - intrinsics.cx) * Z_g / intrinsics.fx
                px_g = (cp.int32(ego_px) + (X_g * ppm_x)).astype(cp.int32)
                py_g = (cp.int32(ego_py) - (Z_g * ppm_z)).astype(cp.int32)

                z_u8  = (255 - cp.clip(Z_g / max_range_m, 0.0, 1.0) * 255).astype(cp.uint8)
                cols  = _plasma_lut_gpu()[z_u8]          # (N, 3) BGR on GPU

                in_b  = (px_g >= 0) & (px_g < W) & (py_g >= 0) & (py_g < H)
                bev_g = cp.asarray(bev)                  # upload canvas yg sudah ada grid
                bev_g[py_g[in_b], px_g[in_b]] = cols[in_b]
                bev[:] = cp.asnumpy(bev_g)

            gpu_used = True
        except Exception as _gpu_err:
            print(f"[BEV] GPU error frame — fallback CPU: {_gpu_err}")

    if not gpu_used:
        # CPU path (numpy): subsample untuk hemat memori
        vs = np.arange(0, img_h, downsample)
        us = np.arange(0, img_w, downsample)
        uu, vv = np.meshgrid(us, vs)
        uu = uu.ravel().astype(np.float32)
        vv = vv.ravel().astype(np.float32)

        Z = depth_map[vv.astype(np.int32), uu.astype(np.int32)]
        valid = (Z > 0.3) & (Z <= max_range_m)
        Z = Z[valid]; uu = uu[valid]

        if Z.size > 0:
            X = (uu - intrinsics.cx) * Z / intrinsics.fx
            px = (ego_px + X * ppm_x).astype(np.int32)
            py = (ego_py - Z * ppm_z).astype(np.int32)

            z_uint8    = (255 - np.clip(Z / max_range_m, 0.0, 1.0) * 255).astype(np.uint8)
            colors_bgr = _plasma_lut_cpu()[z_uint8]

            in_b = (px >= 0) & (px < W) & (py >= 0) & (py < H)
            bev[py[in_b], px[in_b]] = colors_bgr[in_b]

    # ── Ego vehicle marker ───────────────────────────────────────────────────
    cv2.rectangle(bev,
                  (ego_px - 8, ego_py - 15),
                  (ego_px + 8, ego_py),
                  (0, 220, 220), -1)
    cv2.rectangle(bev,
                  (ego_px - 8, ego_py - 15),
                  (ego_px + 8, ego_py),
                  (255, 255, 255), 1)

    # ── Detection overlay ────────────────────────────────────────────────────
    if detections_3d:
        x_m_arr = np.array([d.x_m for d in detections_3d], dtype=np.float32)
        z_m_arr = np.array([d.z_m for d in detections_3d], dtype=np.float32)
        dpx = (ego_px + x_m_arr * ppm_x).astype(np.int32)
        dpy = (ego_py - z_m_arr * ppm_z).astype(np.int32)
        in_bounds = (dpx >= 0) & (dpx < W) & (dpy >= 0) & (dpy < H)

        for i, det in enumerate(detections_3d):
            if not in_bounds[i]:
                continue
            px_d, py_d = int(dpx[i]), int(dpy[i])
            color = HAZARD_COLORS_BEV.get(det.hazard_level, (128, 128, 128))
            radius = max(6, int(det.confidence * 14))
            # Lingkaran dengan outline putih supaya kontras di atas point cloud
            cv2.circle(bev, (px_d, py_d), radius + 2, (255, 255, 255), 2)
            cv2.circle(bev, (px_d, py_d), radius, color, -1)
            label_str = f"{det.label[:5]} {det.z_m:.1f}m"
            cv2.putText(bev, label_str, (px_d + 8, py_d + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)

    # ── Path waypoints ───────────────────────────────────────────────────────
    if path_waypoints and len(path_waypoints) > 1:
        for i in range(len(path_waypoints) - 1):
            pt1 = (int(path_waypoints[i][0]),   int(path_waypoints[i][1]))
            pt2 = (int(path_waypoints[i+1][0]), int(path_waypoints[i+1][1]))
            cv2.line(bev, pt1, pt2, (0, 220, 255), 2)
        cv2.circle(bev,
                   (int(path_waypoints[0][0]),  int(path_waypoints[0][1])),
                   5, (255, 255, 0), -1)
        cv2.circle(bev,
                   (int(path_waypoints[-1][0]), int(path_waypoints[-1][1])),
                   5, (0, 255, 100), -1)

    return bev


def intrinsics_from_calib(calib: dict) -> Intrinsics:
    """Helper: konversi dict calib dari ROSBagRunner payload ke Intrinsics dataclass."""
    return Intrinsics(
        fx=calib['fx'],
        fy=calib['fy'],
        cx=calib['cx'],
        cy=calib['cy'],
    )


def intrinsics_from_frame_width(orig_w: int, stereo_fx: float = 721.53,
                                 native_w: int = 1242) -> Intrinsics:
    """
    Fallback intrinsics jika CameraInfo belum tersedia (misal mode StereoSimLoader).
    Approximasi: cx = orig_w / 2, cy = orig_h / 2 (tidak dipakai untuk X BEV).
    """
    scale = orig_w / native_w
    fx = stereo_fx * scale
    cx = orig_w / 2.0
    cy = cx * (375.0 / 1242.0)  # KITTI native aspect ratio approximation
    return Intrinsics(fx=fx, fy=fx, cx=cx, cy=cy)
