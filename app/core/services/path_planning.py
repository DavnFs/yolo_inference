import heapq
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np

from app.core.services.object_detection import (
    BEV_DESTINATION_POINTS,
    BEV_HEIGHT,
    BEV_WIDTH,
    HAZARD_COLORS,
    PIXELS_PER_METER,
    ROAD_POLYGON_POINTS_RELATIVE,
    _get_frame_geometry,
)


def _clip_rect(x1, y1, x2, y2):
    return (
        max(0, min(BEV_WIDTH - 1, int(x1))),
        max(0, min(BEV_HEIGHT - 1, int(y1))),
        max(0, min(BEV_WIDTH - 1, int(x2))),
        max(0, min(BEV_HEIGHT - 1, int(y2))),
    )


def _point_in_any_bbox(px: int, py: int, detections: list) -> bool:
    for det in detections:
        x1, y1, x2, y2 = det.get("bounding_box", [0, 0, 0, 0])
        if x1 <= px <= x2 and y1 <= py <= y2:
            return True
    return False


def build_occupancy_grid(
    detections: list,
    metric_depth_map: np.ndarray,
    image_shape: tuple,
    road_polygon_abs: np.ndarray,
) -> np.ndarray:
    grid = np.zeros((BEV_HEIGHT, BEV_WIDTH), dtype=np.uint8)

    # Parameter Intrinsik Kamera K (Contoh Nilai Standar DrivingStereo)
    # Sesuaikan dengan nilai fx dan cx milik data loader Anda nanti
    fx = 721.53
    cx = image_shape[1] // 2

    # 1. PEMETAAN OBJEK DETEKSI (YOLO/RCNN)
    for det in detections:
        if det.get("hazard_level") not in ("danger", "warning"):
            continue

        bbox = det.get("bounding_box", [0, 0, 0, 0])
        dist = float(det.get("distance_m") or 0.0)
        if dist <= 0:
            continue

        # Hitung koordinat X dunia nyata menggunakan rumus Pinhole Camera
        bbox_cx = (bbox[0] + bbox[2]) * 0.5
        X_meters = ((bbox_cx - cx) * dist) / fx

        # Proyeksikan koordinat meter ke indeks piksel bidang BEV
        x_bev = int(BEV_WIDTH // 2 + X_meters * PIXELS_PER_METER)
        y_bev = int(BEV_HEIGHT - dist * PIXELS_PER_METER)

        # Enforce minimum physical vehicle width (1.8m) for mirrors & bbox inaccuracies
        bbox_w_meters = ((bbox[2] - bbox[0]) / fx) * dist
        bbox_w_meters = max(1.8, bbox_w_meters)
        obstacle_w = int(bbox_w_meters * PIXELS_PER_METER)
        obstacle_h = max(8, int(dist * 0.2 * PIXELS_PER_METER))

        x1, y1, x2, y2 = _clip_rect(
            x_bev - obstacle_w // 2,
            y_bev - obstacle_h,
            x_bev + obstacle_w // 2,
            y_bev,
        )
        cv2.rectangle(grid, (x1, y1), (x2, y2), 255, -1)

    # 2. C-SPACE INFLATION (Minkowski Sum — ego half-width + safety margin)
    #    Ego width 1.8m → radius 0.9m + 0.3m margin = 1.2m total inflation
    inflation_radius_px = int(1.2 * PIXELS_PER_METER)  # ~9 pixels
    kernel_size = inflation_radius_px * 2 + 1           # 19
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_size, kernel_size)
    )
    grid = cv2.dilate(grid, kernel, iterations=1)

    # Bersihkan baris paling bawah BEV (ego-hood noise zone — AFTER inflation
    # to prevent the large kernel from bleeding into the vehicle baseline)
    grid[BEV_HEIGHT - 40:, :] = 0

    # 4. BEV ROAD MASK CONSTRAINT — virtual walls outside driveable corridor
    #    Warp the camera-frame road polygon into BEV space using the cached
    #    forward perspective matrix so A* can never leave the lane.
    try:
        _, M_forward, _ = _get_frame_geometry(image_shape)
        h_img, w_img = image_shape[:2]
        road_mask_cam = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.fillPoly(road_mask_cam, [road_polygon_abs.astype(np.int32)], 255)
        bev_road_mask = cv2.warpPerspective(
            road_mask_cam, M_forward, (BEV_WIDTH, BEV_HEIGHT),
            flags=cv2.INTER_NEAREST,
        )
        grid[bev_road_mask == 0] = 255  # off-road → absolute obstacle
    except Exception:
        pass  # fallback: grid as-is if warp fails

    return grid


def astar(grid: np.ndarray, start: tuple, goal: tuple) -> List[Tuple[int, int]]:
    """
    A* dengan proximity clearance penalty, heading turn penalty,
    dan grid-boundary validation.
    Heap tuple: (f_score, g_score, (x, y), (dx, dy))
    """
    neighbors = [(-1, -1), (0, -1), (1, -1),
                 (-1, 0),           (1, 0),
                 (-1, 1),  (0, 1),  (1, 1)]
    TURN_PENALTY = 1.5

    def heuristic(a, b):
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    def valid(p):
        return (0 <= p[0] < BEV_WIDTH and 0 <= p[1] < BEV_HEIGHT
                and grid[p[1], p[0]] <= 200)

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

            # Base move cost (diagonal √2)
            move_cost = 1.4142 if dx and dy else 1.0

            # Proximity clearance penalty — exponential from grayscale value
            cell_val = int(grid[nxt[1], nxt[0]])
            if cell_val > 0:
                move_cost += (cell_val / 255.0) ** 2 * 15.0

            # Heading turn penalty
            if parent_dir != (0, 0) and (dx, dy) != parent_dir:
                move_cost += TURN_PENALTY

            tentative_g = current_g + move_cost
            if tentative_g < g_score.get(nxt, float("inf")):
                came_from[nxt] = current
                g_score[nxt] = tentative_g
                f_score = tentative_g + heuristic(nxt, goal)
                heapq.heappush(open_heap, (f_score, tentative_g, nxt, (dx, dy)))

    return []


def smooth_path(waypoints: List[Tuple[int, int]], window=5) -> List[Tuple[int, int]]:
    if not waypoints:
        return []
    if len(waypoints) < window:
        return [(int(x), int(y)) for x, y in waypoints]

    points = np.array(waypoints, dtype=np.float32)
    half = window // 2
    smoothed = []
    for i in range(len(points)):
        start = max(0, i - half)
        end = min(len(points), i + half + 1)
        smoothed.append(np.mean(points[start:end], axis=0))

    smoothed = np.array(smoothed, dtype=np.float32)
    if len(smoothed) > 20:
        indices = np.linspace(0, len(smoothed) - 1, 20).astype(int)
        smoothed = smoothed[indices]

    return [(int(round(x)), int(round(y))) for x, y in smoothed]


def compute_path(
    detections: list,
    metric_depth_map: np.ndarray,
    image_shape: tuple,
    road_polygon_abs: np.ndarray,
) -> Dict:
    grid = build_occupancy_grid(
        detections, metric_depth_map, image_shape, road_polygon_abs
    )

    # --- Validasi node helper (shared) ---
    def is_valid_node(p):
        return (0 <= p[0] < BEV_WIDTH and 0 <= p[1] < BEV_HEIGHT
                and grid[p[1], p[0]] <= 200)

    def relax_goal(g):
        """Jika goal terhalang, cari alternatif terdekat dalam radius 5."""
        if is_valid_node(g):
            return g
        best_alt, best_dist_sq = None, float('inf')
        for dy in range(-5, 6):
            for dx in range(-5, 6):
                alt = (g[0] + dx, g[1] + dy)
                if is_valid_node(alt):
                    d2 = dx * dx + dy * dy
                    if d2 < best_dist_sq:
                        best_dist_sq = d2
                        best_alt = alt
        return best_alt  # None jika tidak ada alternatif

    center_x = BEV_WIDTH // 2

    # --- Ego-Vehicle Dimensions (physical car width → BEV pixels) ---
    ego_width_m = 1.8
    ego_w_px = int((ego_width_m * PIXELS_PER_METER) / 2)  # half-width in pixels
    ego_left = max(0, center_x - ego_w_px)
    ego_right = min(BEV_WIDTH, center_x + ego_w_px)

    starts = [
        (center_x, BEV_HEIGHT - 10),
        (center_x - 20, BEV_HEIGHT - 10),
        (center_x + 20, BEV_HEIGHT - 10),
    ]

    # --- Emergency Stop: front zone just ahead of cleared hood ---
    #    Slice from BEV_HEIGHT-60 to BEV_HEIGHT-40 (above the hood clear zone)
    front_zone = grid[BEV_HEIGHT - 60:BEV_HEIGHT - 40, ego_left:ego_right]
    if np.all(front_zone > 200):
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

    # --- Center-Lane Bias vs Overtaking Maneuver ---
    # Full ego-lane corridor: from row 50 down to just above hood zone
    ego_corridor = grid[50:BEV_HEIGHT - 40, ego_left:ego_right]
    ego_lane_blocked = np.any(ego_corridor > 200)

    if not ego_lane_blocked:
        # Lane perfectly clear — drive straight ahead
        goal_candidates = [(center_x, 15)]
    else:
        # Obstacle in ego-lane — scan horizon for alternate overtaking lanes
        goal_candidates = []
        for y_h in range(10, 31):
            row = grid[y_h, :]
            valid_cols = np.where(row <= 200)[0]
            if len(valid_cols) > 0:
                mid = int(valid_cols[len(valid_cols) // 2])
                goal_candidates.append((mid, y_h))
        if not goal_candidates:
            goal_candidates = [(center_x, 15)]

    t0 = time.perf_counter()
    path = []
    for start in starts:
        if not is_valid_node(start):
            continue
        for goal in goal_candidates:
            relaxed = relax_goal(goal)
            if relaxed is None:
                continue
            path = astar(grid, start, relaxed)
            if path:
                break
        if path:
            break
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # --- Emergency Stop Condition ---
    # A* failed: all lanes blocked. Return a single dot at the vehicle position.
    if not path:
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

    smoothed = smooth_path(path, window=5)
    total_pixels = int(np.sum(grid > 0))
    road_pixels = int(np.sum(grid == 0))
    coverage = round(1.0 - (total_pixels / max(road_pixels + total_pixels, 1)), 4)

    return {
        "waypoints": smoothed,
        "path_points": path,
        "smooth_path": smoothed,
        "path_found": True,
        "obstacle_grid": grid,
        "compute_time_ms": round(elapsed_ms, 2),
        "road_coverage_ratio": coverage,
    }
