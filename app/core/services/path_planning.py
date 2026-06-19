import heapq
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np

from app.core.services.object_detection import (
    BEV_HEIGHT,
    BEV_WIDTH,
    PIXELS_PER_METER,
    STEREO_FX,
)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
_DRIVINGSTEREO_NATIVE_W = 1242
_EGO_WIDTH_M = 1.8
_EGO_LENGTH_M = 2.5
_OBS_MIN_WIDTH_PX = 4
_INFLATION_RADIUS_M = 0.6
_HOOD_CLEAR_ROWS = 40
_EMERGENCY_STOP_ROWS = 40
_EMERGENCY_STOP_THRESHOLD = 0.7
_CRUISE_DIST_M = 20.0
_LANE_CHECK_STRIP_PX = 40
_OVERTAKE_LATERAL_PX = 44
_OVERTAKE_CLEARANCE_PX = 40
_ACC_BACKSTOP_PX = 80
_TURN_PENALTY = 0.5
_GOAL_SNAP_RADIUS = 5


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


def build_occupancy_grid(
    detections: list,
    metric_depth_map: np.ndarray,
    image_shape: tuple,
    road_polygon_abs: np.ndarray,
    camera_intrinsics: dict = None,
) -> np.ndarray:
    grid = np.zeros((BEV_HEIGHT, BEV_WIDTH), dtype=np.uint8)

    orig_w = image_shape[1]
    if camera_intrinsics is not None:
        fx = camera_intrinsics['fx']
        cx = camera_intrinsics['cx']
    else:
        cx = orig_w // 2
        fx = STEREO_FX * (orig_w / _DRIVINGSTEREO_NATIVE_W)

    for det in detections:
        bbox = det.get("bounding_box", [0, 0, 0, 0])
        dist = float(det.get("distance_m") or 0.0)
        if dist <= 0:
            continue

        # --- KOORDINAT LATERAL (KIRI-KANAN) ---
        bbox_cx = (bbox[0] + bbox[2]) * 0.5
        X_meters = ((bbox_cx - cx) * dist) / fx

        x_bev = int(BEV_WIDTH // 2 + X_meters * PIXELS_PER_METER)
        y_bev = int(BEV_HEIGHT - dist * PIXELS_PER_METER)

        bbox_w_meters = ((bbox[2] - bbox[0]) / fx) * dist

        # --- PHYSICAL CLAMPING ---
        bbox_w_meters = min(3.0, bbox_w_meters)

        obstacle_w = max(_OBS_MIN_WIDTH_PX, int(bbox_w_meters * PIXELS_PER_METER))
        obstacle_h = int(_EGO_LENGTH_M * PIXELS_PER_METER)

        x1, y1, x2, y2 = _clip_rect(
            x_bev - obstacle_w // 2,
            y_bev - obstacle_h,
            x_bev + obstacle_w // 2,
            y_bev,
        )
        cv2.rectangle(grid, (x1, y1), (x2, y2), 255, -1)

    inflation_px = max(1, int(_INFLATION_RADIUS_M * PIXELS_PER_METER))
    ksize = inflation_px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    grid = cv2.dilate(grid, kernel, iterations=1)

    grid[BEV_HEIGHT - _HOOD_CLEAR_ROWS :, :] = 0

    return grid


def astar(grid: np.ndarray, start: tuple, goal: tuple) -> List[Tuple[int, int]]:
    neighbors = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]

    def heuristic(a, b):
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    def valid(p):
        return (
            0 <= p[0] < BEV_WIDTH and 0 <= p[1] < BEV_HEIGHT and grid[p[1], p[0]] <= 200
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
            if cell_val > 0:
                move_cost += (cell_val / 255.0) ** 2 * 15.0

            if parent_dir != (0, 0) and (dx, dy) != parent_dir:
                move_cost += _TURN_PENALTY

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
    camera_intrinsics: dict = None,
) -> Dict:
    grid = build_occupancy_grid(
        detections, metric_depth_map, image_shape, road_polygon_abs,
        camera_intrinsics=camera_intrinsics
    )

    def is_valid_node(p):
        return (
            0 <= p[0] < BEV_WIDTH and 0 <= p[1] < BEV_HEIGHT and grid[p[1], p[0]] <= 200
        )

    center_x = BEV_WIDTH // 2
    ego_w_px = int((_EGO_WIDTH_M * PIXELS_PER_METER) / 2)
    ego_left = max(0, center_x - ego_w_px)
    ego_right = min(BEV_WIDTH, center_x + ego_w_px)

    starts = [
        (center_x, BEV_HEIGHT - 10),
        (center_x - 20, BEV_HEIGHT - 10),
        (center_x + 20, BEV_HEIGHT - 10),
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

    # --- 2. Goal selection ---
    ego_corridor = grid[15 : BEV_HEIGHT - _HOOD_CLEAR_ROWS, ego_left:ego_right]
    blocked_rows = np.where(np.any(ego_corridor > 200, axis=1))[0]

    goal_candidates: List[Tuple[int, int]] = []

    if len(blocked_rows) > 0:
        closest_obs_y = 15 + blocked_rows[-1]
        dist_m = ((BEV_HEIGHT - _HOOD_CLEAR_ROWS) - closest_obs_y) / PIXELS_PER_METER

        if dist_m > _CRUISE_DIST_M:
            # ACC: goal between ego and obstacle (not past it)
            acc_y = min(BEV_HEIGHT - 50, closest_obs_y + _ACC_BACKSTOP_PX)
            goal_candidates = [_snap_goal(grid, (center_x, int(acc_y)))]
        else:
            # Overtake: depth-bounded lane check at overtake depth only
            overtake_y = max(
                15,
                closest_obs_y
                - _OVERTAKE_CLEARANCE_PX
                - int(_EGO_LENGTH_M * PIXELS_PER_METER),
            )
            check_row_top = max(15, overtake_y - 10)
            check_row_bot = min(BEV_HEIGHT - _HOOD_CLEAR_ROWS, overtake_y + 10)

            left_blocked = np.any(
                grid[
                    check_row_top:check_row_bot,
                    max(0, ego_left - _LANE_CHECK_STRIP_PX) : ego_left,
                ]
                > 200
            )
            right_blocked = np.any(
                grid[
                    check_row_top:check_row_bot,
                    ego_right : min(BEV_WIDTH, ego_right + _LANE_CHECK_STRIP_PX),
                ]
                > 200
            )

            if not right_blocked:
                goal_candidates.append(
                    _snap_goal(grid, (center_x + _OVERTAKE_LATERAL_PX, overtake_y))
                )
            if not left_blocked:
                goal_candidates.append(
                    _snap_goal(grid, (center_x - _OVERTAKE_LATERAL_PX, overtake_y))
                )

            if not goal_candidates:
                stop_y = min(BEV_HEIGHT - 50, closest_obs_y + int(5 * PIXELS_PER_METER))
                goal_candidates = [_snap_goal(grid, (center_x, max(15, int(stop_y))))]
    else:
        # Clear road: aim at far horizon (y=15)
        goal_candidates = [_snap_goal(grid, (center_x, 15))]

    valid_goals = [g for g in goal_candidates if is_valid_node(g)]
    if not valid_goals:
        valid_goals = [(center_x, BEV_HEIGHT - 10)]

    # Bias start points
    goal_x = valid_goals[0][0]
    if goal_x > center_x:
        ordered_starts = [
            (center_x + 20, BEV_HEIGHT - 10),
            (center_x, BEV_HEIGHT - 10),
            (center_x - 20, BEV_HEIGHT - 10),
        ]
    elif goal_x < center_x:
        ordered_starts = [
            (center_x - 20, BEV_HEIGHT - 10),
            (center_x, BEV_HEIGHT - 10),
            (center_x + 20, BEV_HEIGHT - 10),
        ]
    else:
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
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

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
