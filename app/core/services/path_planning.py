import heapq
from typing import Dict, List, Tuple

import cv2
import numpy as np

from app.core.services.object_detection import (
    BEV_WIDTH, BEV_HEIGHT, PIXELS_PER_METER,
    ROAD_POLYGON_POINTS_RELATIVE, HAZARD_COLORS
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
    road_polygon_abs: np.ndarray
) -> np.ndarray:
    grid = np.zeros((BEV_HEIGHT, BEV_WIDTH), dtype=np.uint8)

    for det in detections:
        if det.get("hazard_level") not in ("danger", "warning"):
            continue

        bbox = det.get("bounding_box", [0, 0, 0, 0])
        dist = float(det.get("distance_m") or 0.0)
        if dist <= 0:
            continue

        bbox_cx = (bbox[0] + bbox[2]) * 0.5
        x_bev = int(BEV_WIDTH // 2 + ((bbox_cx / 640.0) - 0.5) * BEV_WIDTH * 0.8)
        y_bev = int(BEV_HEIGHT - dist * PIXELS_PER_METER)
        obstacle_w = max(6, int((bbox[2] - bbox[0]) / 640.0 * BEV_WIDTH * 0.8))
        obstacle_h = max(8, int(dist * 0.3 * PIXELS_PER_METER))

        x1, y1, x2, y2 = _clip_rect(
            x_bev - obstacle_w // 2,
            y_bev - obstacle_h,
            x_bev + obstacle_w // 2,
            y_bev,
        )
        cv2.rectangle(grid, (x1, y1), (x2, y2), 255, -1)

    h, w = image_shape
    road_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(road_mask, [road_polygon_abs.astype(np.int32)], 255)
    depth_mask = (metric_depth_map < 3.0) & (road_mask > 0)

    x, y, rw, rh = cv2.boundingRect(road_polygon_abs.astype(np.int32))
    xs = np.linspace(x, x + rw - 1, 5).astype(int)
    ys = np.linspace(y, y + rh - 1, 20).astype(int)

    for px in xs:
        for py in ys:
            if px < 0 or py < 0 or px >= w or py >= h:
                continue
            if not depth_mask[py, px]:
                continue
            if _point_in_any_bbox(px, py, detections):
                continue

            dist = float(metric_depth_map[py, px])
            x_norm = (px / max(1, w)) - 0.5
            x_bev = int(BEV_WIDTH // 2 + x_norm * BEV_WIDTH * 0.8)
            y_bev = int(BEV_HEIGHT - dist * PIXELS_PER_METER)
            x1, y1, x2, y2 = _clip_rect(x_bev - 2, y_bev - 2, x_bev + 2, y_bev + 2)
            cv2.rectangle(grid, (x1, y1), (x2, y2), 255, -1)

    return grid


def astar(grid: np.ndarray, start: tuple, goal: tuple) -> List[Tuple[int, int]]:
    def valid(point):
        x, y = point
        return (
            0 <= x < BEV_WIDTH and
            0 <= y < BEV_HEIGHT and
            grid[y, x] <= 127
        )

    if not valid(start) or not valid(goal):
        return []

    def heuristic(a, b):
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    neighbors = [
        (-1, -1), (0, -1), (1, -1),
        (-1, 0),           (1, 0),
        (-1, 1),  (0, 1),  (1, 1),
    ]
    open_heap = [(heuristic(start, goal), 0.0, start)]
    came_from = {}
    g_score = {start: 0.0}
    visited = set()
    max_iterations = BEV_WIDTH * BEV_HEIGHT
    iterations = 0

    while open_heap and iterations < max_iterations:
        iterations += 1
        _, current_g, current = heapq.heappop(open_heap)
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
            tentative_g = current_g + move_cost
            if tentative_g < g_score.get(nxt, float("inf")):
                came_from[nxt] = current
                g_score[nxt] = tentative_g
                f_score = tentative_g + heuristic(nxt, goal)
                heapq.heappush(open_heap, (f_score, tentative_g, nxt))

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
    road_polygon_abs: np.ndarray
) -> Dict:
    grid = build_occupancy_grid(detections, metric_depth_map, image_shape, road_polygon_abs)

    center_x = BEV_WIDTH // 2
    starts = [
        (center_x, BEV_HEIGHT - 10),
        (center_x - 20, BEV_HEIGHT - 10),
        (center_x + 20, BEV_HEIGHT - 10),
    ]
    goals = [
        (center_x, 10),
        (center_x - 20, 10),
        (center_x + 20, 10),
    ]

    path = []
    for start in starts:
        for goal in goals:
            path = astar(grid, start, goal)
            if path:
                break
        if path:
            break

    if not path:
        return {
            "waypoints": [],
            "path_found": False,
            "obstacle_grid": grid,
        }

    return {
        "waypoints": smooth_path(path, window=5),
        "path_found": True,
        "obstacle_grid": grid,
    }
