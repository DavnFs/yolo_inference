import heapq
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np

from app.core.services.object_detection import (
    BEV_HEIGHT,
    BEV_WIDTH,
    PIXELS_PER_METER,
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

    # Dynamic focal length: ~50° FoV for standard high-res stereo dashcams
    # fx = image_width * 1.28 gives FoV ≈ 2 * atan(1/(2*1.28)) ≈ 43.6°
    # For 1242px (KITTI): fx≈1590; for 1920px: fx≈2458 — matches real stereo rigs
    orig_w = image_shape[1]
    cx = orig_w // 2
    fx = orig_w * 1.28

    # 1. PEMETAAN OBJEK DETEKSI (YOLO/RCNN) — ALL detections with valid distance
    for det in detections:
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

        # Trust pinhole projection width directly (YOLO boxes already encapsulate margins)
        bbox_w_meters = ((bbox[2] - bbox[0]) / fx) * dist
        obstacle_w = max(4, int(bbox_w_meters * PIXELS_PER_METER))
        obstacle_h = int(2.5 * PIXELS_PER_METER)  # realistic vehicle length ~2.5m

        x1, y1, x2, y2 = _clip_rect(
            x_bev - obstacle_w // 2,
            y_bev - obstacle_h,
            x_bev + obstacle_w // 2,
            y_bev,
        )
        cv2.rectangle(grid, (x1, y1), (x2, y2), 255, -1)

    # 2. C-SPACE INFLATION (Minkowski Sum — ~0.6m safety padding)
    #    11x11 kernel ≈ 5px radius × 0.125m/px = 0.625m physical padding
    #    YOLO boxes already encapsulate some empty road, so no need for huge kernels
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    grid = cv2.dilate(grid, kernel, iterations=1)

    # Bersihkan baris paling bawah BEV (ego-hood noise zone — AFTER inflation
    # to prevent the large kernel from bleeding into the vehicle baseline)
    grid[BEV_HEIGHT - 40:, :] = 0

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
    #    Trigger if ≥70% of the corridor width is blocked (tolerates 1-pixel gaps)
    front_zone = grid[BEV_HEIGHT - 60:BEV_HEIGHT - 40, ego_left:ego_right]
    if np.mean(front_zone > 200) > 0.7:
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

    # --- Adaptive Cruise Control & Overtaking ---
    # Scan ego lane to find the CLOSEST obstacle. Distant obstacles get a
    # lane-keeping goal well behind them; only dangerously close obstacles
    # (<= 20m) trigger an overtaking maneuver.
    ego_lane_slice = grid[15:BEV_HEIGHT - 40, ego_left:ego_right]
    blocked_rows = np.where(np.any(ego_lane_slice > 200, axis=1))[0]

    if len(blocked_rows) > 0:
        closest_obs_y = 15 + blocked_rows[-1]
        dist_m = ((BEV_HEIGHT - 40) - closest_obs_y) / PIXELS_PER_METER
    else:
        closest_obs_y = 15
        dist_m = 999.0

    if dist_m > 20.0:
        # Obstacle is far — stay in lane, place goal behind it
        goal_candidates = [
            (center_x, min(BEV_HEIGHT - 50, closest_obs_y + 24))
        ]
    else:
        # Dangerously close — try to overtake via adjacent lanes
        # Examine the adjacent lanes ONLY from the obstacle's row down to the hood
        adj_check_rows = slice(closest_obs_y, BEV_HEIGHT - 40)
        left_lane_blocked = np.any(
            grid[adj_check_rows, max(0, ego_left - 30):ego_left] > 200
        )
        right_lane_blocked = np.any(
            grid[adj_check_rows, ego_right:min(BEV_WIDTH, ego_right + 30)] > 200
        )

        if not right_lane_blocked:
            goal_candidates = [(center_x + 28, max(15, closest_obs_y - 15))]
        elif not left_lane_blocked:
            goal_candidates = [(center_x - 28, max(15, closest_obs_y - 15))]
        else:
            # Both adjacent lanes blocked — fall back to stopping behind obstacle
            goal_candidates = [(center_x, closest_obs_y + 24)]

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
