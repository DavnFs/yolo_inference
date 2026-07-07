import heapq
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np

from app.core.config import get_scaled_intrinsics
from app.core.services.object_detection import (
    BEV_HEIGHT,
    BEV_WIDTH,
    PIXELS_PER_METER,
)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
_DRIVINGSTEREO_NATIVE_W = 1242
_EGO_WIDTH_M = 1.8
_EGO_LENGTH_M = 2.5
_OBS_MIN_WIDTH_PX = 8
_INFLATION_RADIUS_M = (
    _EGO_WIDTH_M / 2.0
) + 0.5  # Inflasi berdasarkan lebar mobil + margin
_HOOD_CLEAR_ROWS = 40
_EMERGENCY_STOP_ROWS = 40
_EMERGENCY_STOP_THRESHOLD = 0.7
_CRUISE_DIST_M = 20.0
_LANE_CHECK_STRIP_PX = 40
_OVERTAKE_LATERAL_PX = 44
_OVERTAKE_CLEARANCE_PX = 40
_ACC_BACKSTOP_PX = 80
_TURN_PENALTY = 1.5
_GOAL_SNAP_RADIUS = 5
# Margin aman fisik antara bodi mobil ego dan objek (15 cm)
_MARGIN_SAFE_M = 0.15

# Configuration Space (Minkowski Sum): Dimensi Ego + Margin
_DELTA_W_M = _EGO_WIDTH_M + (2 * _MARGIN_SAFE_M)  # 1.8 + 0.3 = 2.1 m
_DELTA_H_M = _EGO_LENGTH_M + (2 * _MARGIN_SAFE_M)  # 2.5 + 0.3 = 2.8 m


def _clip_rect(x1, y1, x2, y2):
    return (
        max(0, min(BEV_WIDTH - 1, int(x1))),
        max(0, min(BEV_HEIGHT - 1, int(y1))),
        max(0, min(BEV_WIDTH - 1, int(x2))),
        max(0, min(BEV_HEIGHT - 1, int(y2))),
    )


def _snap_goal(grid: np.ndarray, goal: Tuple[int, int]) -> Tuple[int, int]:
    cx, cy = goal
    if 0 <= cx < BEV_WIDTH and 0 <= cy < BEV_HEIGHT and grid[cy, cx] <= 200:
        return goal
    best, best_d2 = goal, float("inf")
    for dy in range(-_GOAL_SNAP_RADIUS, _GOAL_SNAP_RADIUS + 1):
        for dx in range(-_GOAL_SNAP_RADIUS, _GOAL_SNAP_RADIUS + 1):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < BEV_WIDTH and 0 <= ny < BEV_HEIGHT and grid[ny, nx] <= 200:
                d2 = dx * dx + dy * dy
                if d2 < best_d2:
                    best_d2 = d2
                    best = (nx, ny)
    return best


def _get_road_color_mask(
    frame_bgr: np.ndarray, road_start_row: int, depth_h: int, depth_w: int
) -> np.ndarray:
    """Reject green/vegetation-colored pixels from the drivable mask.

    Depth alone can't tell asphalt from a flat grass verge, both satisfy
    the ground-plane equation. This adds an independent color cue so
    flood fill can't walk from the road onto the lawn just because
    they're coplanar.

    Returns a boolean mask, True where the pixel is allowed to remain
    drivable (i.e. NOT classified as vegetation).
    """
    if frame_bgr.shape[0] != depth_h or frame_bgr.shape[1] != depth_w:
        frame_bgr = cv2.resize(
            frame_bgr, (depth_w, depth_h), interpolation=cv2.INTER_LINEAR
        )

    region_bgr = frame_bgr[road_start_row:depth_h, :]
    hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    h_ch, s_ch = hsv[:, :, 0], hsv[:, :, 1]

    # OpenCV hue range is 0-179. Green/vegetation sits roughly 35-95.
    # Saturation gate avoids flagging desaturated/shadowed grass as road
    # by accident, but also means this will miss dead or dry grass,
    # see caveat below.
    is_vegetation = (h_ch >= 35) & (h_ch <= 95) & (s_ch >= 40)
    return ~is_vegetation


def estimate_driveable_mask(
    metric_depth_map: np.ndarray,
    image_shape: tuple,
    camera_intrinsics: dict = None,
    camera_height_m: float = 1.65,
    depth_tolerance_ratio: float = 0.40,
    frame_bgr: np.ndarray = None,
) -> np.ndarray:
    """Estimate driveable area from depth map using ground plane model.

    Uses the flat-ground assumption: for a camera mounted at height *h*,
    the expected depth at image row *v* on the ground plane is
    ``Z = (fy * h) / (v - cy)``.
    Pixels whose measured depth is close to this expectation (within
    *depth_tolerance_ratio*) are classified as road / ground.

    The resulting image-space mask is then back-projected into BEV
    coordinates so it can be fused with the polygon-based mask in
    :func:`build_occupancy_grid`.

    Args:
        metric_depth_map: (H, W) float32 depth in metres.
        image_shape: (height, width) of the source image.
        camera_intrinsics: dict with keys ``'fx'``, ``'fy'``, ``'cx'``,
            ``'cy'``.  Falls back to KITTI defaults when *None*.
        camera_height_m: Camera mounting height above ground [m].
        depth_tolerance_ratio: Fractional tolerance for ground
            classification (0.25 = 25 %).

    Returns:
        BEV-space driveable mask *(BEV_HEIGHT, BEV_WIDTH)*, ``uint8``,
        where ``255`` = driveable.
    """
    depth_h, depth_w = metric_depth_map.shape[:2]

    if camera_intrinsics is None:
        camera_intrinsics = get_scaled_intrinsics(depth_w, depth_h)

    fx = camera_intrinsics["fx"]
    fy = camera_intrinsics["fy"]
    cx_cam = camera_intrinsics["cx"]
    cy_cam = camera_intrinsics["cy"]
    camera_height_m = camera_intrinsics.get("camera_height_m", camera_height_m)

    # Only analyse bottom 55 % of image (where road typically appears)
    road_start_row = int(depth_h * 0.45)

    v_coords = np.arange(road_start_row, depth_h, dtype=np.float32)
    v_relative = np.maximum(v_coords - cy_cam, 1.0)
    expected_z = (fy * camera_height_m) / v_relative

    expected_z_map = np.broadcast_to(
        expected_z[:, np.newaxis], (len(v_coords), depth_w)
    )

    actual_depth = metric_depth_map[road_start_row:depth_h, :]

    valid_mask = actual_depth > 0.5
    depth_error = np.abs(actual_depth - expected_z_map)
    # Tighten tolerance to prevent classifying sidewalks/walls as road
    tolerance = expected_z_map * depth_tolerance_ratio + 0.5
    ground_mask_road = valid_mask & (depth_error < tolerance)

    # NEW: reject vegetation-colored pixels even if they passed the
    # flatness check
    if frame_bgr is not None:
        color_mask = _get_road_color_mask(frame_bgr, road_start_row, depth_h, depth_w)
        ground_mask_road = ground_mask_road & color_mask

    ground_mask = np.zeros((depth_h, depth_w), dtype=np.uint8)
    ground_mask[road_start_row:depth_h] = ground_mask_road.astype(np.uint8) * 255

    # Morphological cleanup
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    ground_mask = cv2.morphologyEx(ground_mask, cv2.MORPH_CLOSE, kernel_close)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
    ground_mask = cv2.morphologyEx(ground_mask, cv2.MORPH_OPEN, kernel_open)

    # Project ground mask to BEV grid
    bev_mask = np.zeros((BEV_HEIGHT, BEV_WIDTH), dtype=np.uint8)

    ground_pixels = np.where(ground_mask > 0)
    if len(ground_pixels[0]) == 0:
        return bev_mask

    v_px = ground_pixels[0].astype(np.float32)
    u_px = ground_pixels[1].astype(np.float32)
    z_vals = metric_depth_map[ground_pixels[0], ground_pixels[1]]

    valid = (z_vals > 0.5) & (z_vals < 50.0)
    v_px, u_px, z_vals = v_px[valid], u_px[valid], z_vals[valid]

    if len(z_vals) == 0:
        return bev_mask

    # Subsample for performance
    step = 4
    v_px, u_px, z_vals = v_px[::step], u_px[::step], z_vals[::step]

    x_m = ((u_px - cx_cam) * z_vals) / fx
    x_grid = (BEV_WIDTH / 2.0 + x_m * PIXELS_PER_METER).astype(np.int32)
    y_grid = (BEV_HEIGHT - z_vals * PIXELS_PER_METER).astype(np.int32)

    in_bounds = (
        (x_grid >= 0) & (x_grid < BEV_WIDTH) & (y_grid >= 0) & (y_grid < BEV_HEIGHT)
    )
    bev_mask[y_grid[in_bounds], x_grid[in_bounds]] = 255

    # Dilate to fill gaps between projected points
    kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bev_mask = cv2.dilate(bev_mask, kernel_fill, iterations=2)
    bev_mask = cv2.morphologyEx(
        bev_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)),
    )

    return bev_mask


def get_connected_drivable_mask(
    bev_ground_mask: np.ndarray, ego_x: int, ego_y: int
) -> np.ndarray:
    """
    Menggunakan Flood Fill untuk memastikan hanya area yang terhubung secara fisik
    dengan posisi Ego Vehicle yang dianggap sebagai jalan (drivable area).
    Ini mencegah mobil menganggap trotoar atau area datar yang terputus sebagai jalan.
    """
    # Pastikan mask biner (0 dan 255)
    binary_mask = np.zeros_like(bev_ground_mask)
    binary_mask[bev_ground_mask > 0] = 255

    # Tutup celah kecil (noise pada depth map) agar jalan tidak terputus secara tidak sengaja
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    # Flood fill dari posisi ego
    flood_mask = np.zeros(
        (binary_mask.shape[0] + 2, binary_mask.shape[1] + 2), np.uint8
    )
    # Kita gunakan nilai 128 untuk menandai area yang terhubung
    cv2.floodFill(binary_mask, flood_mask, (ego_x, ego_y), 128)

    # Hanya area yang terhubung (128) yang dianggap driveable
    connected_mask = np.zeros_like(bev_ground_mask)
    connected_mask[binary_mask == 128] = 255
    return connected_mask


def build_occupancy_grid(
    detections: list,
    metric_depth_map: np.ndarray,
    image_shape: tuple,
    road_polygon_abs: np.ndarray,
    camera_intrinsics: dict = None,
    frame_bgr: np.ndarray = None,
) -> np.ndarray:
    grid = np.zeros((BEV_HEIGHT, BEV_WIDTH), dtype=np.uint8)

    orig_w = image_shape[1]
    orig_h = image_shape[0]
    if camera_intrinsics is None:
        camera_intrinsics = get_scaled_intrinsics(orig_w, orig_h)

    fx = camera_intrinsics["fx"]
    fy = camera_intrinsics["fy"]
    cx = camera_intrinsics["cx"]
    cy = camera_intrinsics["cy"]

    # 1. Project the road polygon to BEV grid (Area Jalan / Drivable)
    camera_height_m = camera_intrinsics.get("camera_height_m", 1.65)
    bev_poly_pts = []
    for pt in road_polygon_abs:
        u, v = pt[0], pt[1]
        v_val = max(1.0, v - cy)
        z_m = (fy * camera_height_m) / v_val
        x_m = ((u - cx) * z_m) / fx

        x_grid = int(BEV_WIDTH / 2.0 + x_m * PIXELS_PER_METER)
        y_grid = int(BEV_HEIGHT - z_m * PIXELS_PER_METER)

        x_grid = max(-100, min(BEV_WIDTH + 100, x_grid))
        y_grid = max(-100, min(BEV_HEIGHT + 100, y_grid))
        bev_poly_pts.append([x_grid, y_grid])

    if len(bev_poly_pts) == 4:
        pt_bl = bev_poly_pts[0]
        pt_tl = bev_poly_pts[1]
        pt_tr = bev_poly_pts[2]
        pt_br = bev_poly_pts[3]

        dy_l = pt_tl[1] - pt_bl[1]
        if abs(dy_l) > 1e-3:
            x_new_tl = int(round(pt_tl[0] - pt_tl[1] * (pt_tl[0] - pt_bl[0]) / dy_l))
            x_new_bl = int(
                round(pt_bl[0] + (BEV_HEIGHT - pt_bl[1]) * (pt_tl[0] - pt_bl[0]) / dy_l)
            )
        else:
            x_new_tl = pt_tl[0]
            x_new_bl = pt_bl[0]

        dy_r = pt_tr[1] - pt_br[1]
        if abs(dy_r) > 1e-3:
            x_new_tr = int(round(pt_tr[0] - pt_tr[1] * (pt_tr[0] - pt_br[0]) / dy_r))
            x_new_br = int(
                round(pt_br[0] + (BEV_HEIGHT - pt_br[1]) * (pt_tr[0] - pt_br[0]) / dy_r)
            )
        else:
            x_new_tr = pt_tr[0]
            x_new_br = pt_br[0]

        extrapolated_pts = [
            [x_new_bl, BEV_HEIGHT],
            [x_new_tl, 0],
            [x_new_tr, 0],
            [x_new_br, BEV_HEIGHT],
        ]
        pts = np.array(extrapolated_pts, dtype=np.int32)
    else:
        pts = np.array(bev_poly_pts, dtype=np.int32)

    drivable_mask = np.zeros((BEV_HEIGHT, BEV_WIDTH), dtype=np.uint8)
    cv2.fillPoly(drivable_mask, [pts], 255)

    # Strictly enforce the road polygon. 
    # Using np.maximum allowed the car to drive on grass, and cv2.bitwise_and 
    # fragments the road because the stereo depth mask is too noisy.
    combined_mask = drivable_mask

    # Connected Component Analysis (Agar trotoar tidak dianggap jalan)
    ego_x = BEV_WIDTH // 2
    ego_y = BEV_HEIGHT - _HOOD_CLEAR_ROWS - 5

    # Gunakan fungsi get_connected_drivable_mask yang kita buat sebelumnya
    connected_driveable = get_connected_drivable_mask(combined_mask, ego_x, ego_y)

    # 2. Bangun Costmap OGM
    # 0   = Jalan Aman
    # 240 = Off-Road / Trotoar (Dinding)
    # 255 = Rintangan Fisik (Objek)
    grid[connected_driveable == 0] = 240

    # Hitung penambahan piksel untuk Configuration Space (Minkowski Sum)
    delta_w_px = int(_DELTA_W_M * PIXELS_PER_METER)
    delta_h_px = int(_DELTA_H_M * PIXELS_PER_METER)

    # 3. Proyeksikan Objek YOLO ke Grid dengan Configuration Space
    for det in detections:
        bbox = det.get("bounding_box", [0, 0, 0, 0])
        dist = float(det.get("distance_m") or 0.0)
        if dist <= 0:
            continue

        bbox_cx = (bbox[0] + bbox[2]) * 0.5
        X_meters = ((bbox_cx - cx) * dist) / fx

        x_bev = int(BEV_WIDTH // 2 + X_meters * PIXELS_PER_METER)
        y_bev = int(BEV_HEIGHT - dist * PIXELS_PER_METER)

        w_image_px = bbox[2] - bbox[0]
        W_meters = (w_image_px * dist) / fx
        h_image_px = bbox[3] - bbox[1]
        H_meters = (h_image_px * dist) / fy

        w_bev = int(W_meters * PIXELS_PER_METER)
        h_bev = int(H_meters * PIXELS_PER_METER)

        w_bev = max(w_bev, 4)
        h_bev = max(h_bev, 4)

        # --- MINKOWSKI SUM (CONFIGURATION SPACE) ---
        # Perbesar objek secara presisi (persegi panjang) sesuai dimensi mobil + margin 15cm
        w_bev_final = w_bev + delta_w_px
        h_bev_final = h_bev + delta_h_px

        # Gunakan y_bev sebagai TITIK TENGAH objek (bukan bumper belakang)
        x1_grid = x_bev - (w_bev_final // 2)
        y1_grid = y_bev - (h_bev_final // 2)
        x2_grid = x_bev + (w_bev_final // 2)
        y2_grid = y_bev + (h_bev_final // 2)

        x1, y1, x2, y2 = _clip_rect(x1_grid, y1_grid, x2_grid, y2_grid)
        cv2.rectangle(grid, (x1, y1), (x2, y2), 255, -1)

    # 4. Morfologi Penutup Celah (Sangat Kecil)
    # Hapus cv2.dilate raksasa! Kita hanya butuh 1 iterasi kotak 3x3 untuk menutupi noise depth map.
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    obstacle_mask = (grid == 255).astype(np.uint8)
    obstacle_mask = cv2.dilate(obstacle_mask, kernel_small, iterations=1)
    grid[obstacle_mask == 1] = 255

    grid[BEV_HEIGHT - _HOOD_CLEAR_ROWS :, :] = 0

    return grid


def astar(grid: np.ndarray, start: tuple, goal: tuple) -> List[Tuple[int, int]]:
    neighbors = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]

    def heuristic(a, b):
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    def valid(p):
        # PERUBAHAN KUNCI: Hanya izinkan sel dengan nilai < 240
        # 240 adalah Off-Road, 255 adalah Rintangan. Keduanya tidak boleh diinjak.
        return (
            0 <= p[0] < BEV_WIDTH and 0 <= p[1] < BEV_HEIGHT and grid[p[1], p[0]] < 240
        )

    open_heap = [(heuristic(start, goal), 0.0, start, (0, 0))]
    came_from = {}
    g_score = {start: 0.0}
    visited = set()
    max_iter = BEV_WIDTH * BEV_HEIGHT
    iterations = 0

    while open_heap and iterations < max_iter:
        iterations += 1
        _, current_g, current, parent_dir = heapq.heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        for dx, dy in neighbors:
            nxt = (current[0] + dx, current[1] + dy)
            if not valid(nxt):
                continue

            move_cost = 1.4142 if dx and dy else 1.0
            cell_val = int(grid[nxt[1], nxt[0]])

            # Penalti biaya yang lebih agresif untuk area yang bukan 0
            if cell_val > 0:
                move_cost += (cell_val / 255.0) ** 2 * 50.0  # Naikkan dari 15.0 ke 50.0

            if parent_dir != (0, 0) and (dx, dy) != parent_dir:
                move_cost += _TURN_PENALTY

            tentative_g = current_g + move_cost
            if tentative_g < g_score.get(nxt, float("inf")):
                came_from[nxt] = current
                g_score[nxt] = tentative_g
                f_score = tentative_g + heuristic(nxt, goal)
                heapq.heappush(open_heap, (f_score, tentative_g, nxt, (dx, dy)))

    return []


def _smooth_moving_average(
    waypoints: List[Tuple[int, int]], window: int = 15
) -> List[Tuple[int, int]]:
    if not waypoints or len(waypoints) < 3:
        return [(int(x), int(y)) for x, y in waypoints]

    # 1. Linear interpolation to increase density of waypoints (1 point every 2 pixels)
    points = np.array(waypoints, dtype=np.float32)
    dists = np.sqrt(np.diff(points[:, 0]) ** 2 + np.diff(points[:, 1]) ** 2)
    cum_dists = np.concatenate(([0], np.cumsum(dists)))

    new_dists = np.arange(0, cum_dists[-1], 2.0)
    if len(new_dists) < 5:
        new_dists = np.linspace(0, cum_dists[-1], 20)

    x_new = np.interp(new_dists, cum_dists, points[:, 0])
    y_new = np.interp(new_dists, cum_dists, points[:, 1])
    dense_points = np.stack((x_new, y_new), axis=1)

    # 2. Double-pass moving average smoothing (approximating Gaussian filter)
    smoothed = dense_points.copy()
    half = window // 2
    for _ in range(2):
        temp = smoothed.copy()
        for i in range(len(smoothed)):
            start = max(0, i - half)
            end = min(len(smoothed), i + half + 1)
            temp[i] = np.mean(smoothed[start:end], axis=0)
        smoothed = temp

    # Ensure exact start and end coordinates are preserved
    smoothed[0] = points[0]
    smoothed[-1] = points[-1]

    return [(int(round(x)), int(round(y))) for x, y in smoothed]


def smooth_path(
    waypoints: List[Tuple[int, int]], window: int = 15
) -> List[Tuple[int, int]]:
    """Smooth raw A* waypoints using cubic spline interpolation.

    Produces natural, curved trajectories instead of grid-aligned angular
    paths.  Falls back to moving-average if *scipy* is not installed.

    Args:
        waypoints: Raw A* path as list of (x, y) grid coordinates.
        window: Smoothing parameter (kept for API compat; controls
                fallback moving-average window size).

    Returns:
        Smoothed path as list of (x, y) integer coordinates.
    """
    if not waypoints or len(waypoints) < 3:
        return [(int(x), int(y)) for x, y in waypoints]

    try:
        from scipy.interpolate import CubicSpline
    except ImportError:
        return _smooth_moving_average(waypoints, window)

    points = np.array(waypoints, dtype=np.float64)

    # --- 1. Subsample control points to avoid spline overfitting ----------
    n_pts = len(points)
    if n_pts > 30:
        indices = np.linspace(0, n_pts - 1, min(25, max(10, n_pts // 3))).astype(int)
        indices = np.unique(indices)
        control_pts = points[indices]
    else:
        control_pts = points

    # --- 2. Parameterise by cumulative arc length -------------------------
    diffs = np.diff(control_pts, axis=0)
    seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
    t = np.concatenate(([0.0], np.cumsum(seg_lengths)))
    total_length = t[-1]

    if total_length < 1.0:
        return [(int(x), int(y)) for x, y in waypoints]

    # --- 3. Fit natural cubic spline on X(t) and Y(t) --------------------
    cs_x = CubicSpline(t, control_pts[:, 0], bc_type="natural")
    cs_y = CubicSpline(t, control_pts[:, 1], bc_type="natural")

    # --- 4. Evaluate at ~1 pt every 2 px along the curve ------------------
    n_eval = max(20, int(total_length / 2.0))
    t_eval = np.linspace(0, total_length, n_eval)
    x_smooth = cs_x(t_eval)
    y_smooth = cs_y(t_eval)

    # --- 5. Curvature constraint removed to preserve natural cubic spline curve ---

    # --- 6. Pin exact start / end -----------------------------------------
    x_smooth[0], y_smooth[0] = points[0]
    x_smooth[-1], y_smooth[-1] = points[-1]

    return [(int(round(x)), int(round(y))) for x, y in zip(x_smooth, y_smooth)]


def compute_path(
    detections: list,
    metric_depth_map: np.ndarray,
    image_shape: tuple,
    road_polygon_abs: np.ndarray,
    camera_intrinsics: dict = None,
    frame_bgr: np.ndarray = None,
) -> Dict:
    grid = build_occupancy_grid(
        detections,
        metric_depth_map,
        image_shape,
        road_polygon_abs,
        camera_intrinsics=camera_intrinsics,
        frame_bgr=frame_bgr,
    )

    def is_valid_node(p):
        # Hanya izinkan node yang BUKAN off-road (240) dan BUKAN rintangan (255)
        return (
            0 <= p[0] < BEV_WIDTH and 0 <= p[1] < BEV_HEIGHT and grid[p[1], p[0]] < 240
        )

    center_x = BEV_WIDTH // 2
    # Widen corridor to ~2 meters half-width (4m total lane width) to detect obstacles better
    ego_w_px = int(2.0 * PIXELS_PER_METER)
    ego_left = max(0, center_x - ego_w_px)
    ego_right = min(BEV_WIDTH, center_x + ego_w_px)

    starts = [
        (center_x, BEV_HEIGHT - 5),
    ]

    # --- 1. Emergency stop ---
    front_zone = grid[
        BEV_HEIGHT - _HOOD_CLEAR_ROWS - _EMERGENCY_STOP_ROWS : BEV_HEIGHT
        - _HOOD_CLEAR_ROWS,
        ego_left:ego_right,
    ]
    if np.mean(front_zone > 200) > _EMERGENCY_STOP_THRESHOLD:
        stop_point = (center_x, BEV_HEIGHT - 10)
        return {
            "waypoints": [stop_point],
            "path_points": [stop_point],
            "smooth_path": [stop_point],
            "path_found": False,
            "obstacle_grid": grid,
            "compute_time_ms": 0.0,
            "road_coverage_ratio": 0.0,
        }

    # --- 2. Goal selection (Always Go Long) ---
    goal_candidates: List[Tuple[int, int]] = []
    drivable_mask = (grid < 240).astype(np.uint8)
    valid_ys, valid_xs = np.where(drivable_mask > 0)
    
    if len(valid_ys) > 0:
        # Primary Goal: The furthest reachable valid area (minimum Y)
        min_y = np.min(valid_ys)
        top_y_limit = min_y + 20 
        top_xs = valid_xs[valid_ys <= top_y_limit]
        if len(top_xs) > 0:
            target_x = int(np.median(top_xs))
            target_y = int(np.median(valid_ys[valid_ys <= top_y_limit]))
            goal_candidates.append(_snap_goal(grid, (target_x, target_y)))

    valid_goals = [g for g in goal_candidates if is_valid_node(g)]
    if not valid_goals:
        valid_goals = [(center_x, 10)]

    ordered_starts = starts

    # --- 3. Run A* ---
    t0 = time.perf_counter()
    path = []
    for start in ordered_starts:
        if not is_valid_node(start):
            continue
        for goal in valid_goals:
            path = astar(grid, start, goal)
            if path:
                break
        if path:
            break

    # Fallback search: If long path failed (e.g., road is completely blocked across all lanes)
    # just plan a path to stop safely behind the closest obstacle in the ego lane.
    ego_corridor = grid[15 : BEV_HEIGHT - _HOOD_CLEAR_ROWS, ego_left:ego_right]
    blocked_rows = np.where(np.any(ego_corridor > 200, axis=1))[0]
    
    if (not path or len(path) < 2) and len(blocked_rows) > 0:
        closest_obs_y = 15 + blocked_rows[-1]
        stop_y = min(BEV_HEIGHT - 50, closest_obs_y + int(5 * PIXELS_PER_METER))
        stop_goal = _snap_goal(grid, (center_x, max(15, int(stop_y))))
        if is_valid_node(stop_goal):
            for start in ordered_starts:
                if not is_valid_node(start):
                    continue
                path = astar(grid, start, stop_goal)
                if path:
                    break

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if not path or len(path) < 2:
        stop_point = (center_x, BEV_HEIGHT - 10)
        return {
            "waypoints": [stop_point],
            "path_points": [stop_point],
            "smooth_path": [stop_point],
            "path_found": False,
            "obstacle_grid": grid,
            "compute_time_ms": round(elapsed_ms, 2),
            "road_coverage_ratio": 0.0,
        }

    smoothed = smooth_path(path, window=15)
    total_pixels = int(np.sum(grid > 0))
    road_pixels = int(np.sum(grid == 0))
    coverage = round(1.0 - (total_pixels / max(road_pixels + total_pixels, 1)), 4)

    return {
        "waypoints": smoothed,
        "path_points": smoothed,  # FORCE OGM to use the smoothed cubic spline path
        "smooth_path": smoothed,
        "path_found": True,
        "obstacle_grid": grid,
        "compute_time_ms": round(elapsed_ms, 2),
        "road_coverage_ratio": coverage,
    }
