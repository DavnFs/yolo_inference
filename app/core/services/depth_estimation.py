import threading
from typing import Tuple

import cv2
import numpy as np
import onnxruntime as ort


DEPTH_INPUT_SIZE = 518
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEPTH_MODEL_CACHE = {}
DEPTH_MODEL_LOCK = threading.Lock()


def _build_session_options(use_gpu: bool) -> ort.SessionOptions:
    sess_options = ort.SessionOptions()
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    if use_gpu:
        sess_options.enable_cpu_mem_arena = False
        sess_options.enable_mem_pattern = True
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        try:
            sess_options.enable_mem_reuse = True
        except Exception:
            pass
    else:
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 1

    return sess_options


def _build_providers():
    available = ort.get_available_providers()
    print(f"[Depth] Available providers: {available}")

    if "DmlExecutionProvider" in available:
        print("[Depth] Using DirectML provider.")
        return [(
            "DmlExecutionProvider",
            {
                "device_id": 0,
                "performance_preference": 0,
                "disable_metacommands": False,
            }
        )], True

    print("[Depth] DirectML not available. Using CPU provider.")
    return ["CPUExecutionProvider"], False


def load_depth_model(model_path: str) -> ort.InferenceSession:
    with DEPTH_MODEL_LOCK:
        if model_path in DEPTH_MODEL_CACHE:
            return DEPTH_MODEL_CACHE[model_path]

        providers, use_gpu = _build_providers()
        sess_options = _build_session_options(use_gpu)
        try:
            session = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=providers,
            )
        except Exception:
            if use_gpu:
                print("[Depth] DirectML init failed. Retrying with default DirectML provider options.")
                session = ort.InferenceSession(
                    model_path,
                    sess_options=sess_options,
                    providers=["DmlExecutionProvider"],
                )
            else:
                raise

        input_name = session.get_inputs()[0].name
        dummy = np.zeros(
            (1, 3, DEPTH_INPUT_SIZE, DEPTH_INPUT_SIZE),
            dtype=np.float32,
        )
        try:
            session.run(None, {input_name: dummy})
            print("[Depth] Warmup complete.")
        except Exception as e:
            print(f"[Depth] Warmup failed: {e}")

        DEPTH_MODEL_CACHE[model_path] = session
        return session


def preprocess_depth(image_bgr: np.ndarray) -> Tuple[np.ndarray, tuple]:
    original_shape = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(
        image_rgb,
        (DEPTH_INPUT_SIZE, DEPTH_INPUT_SIZE),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32) / 255.0

    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    normalized = (resized - mean) / std
    input_tensor = normalized.transpose(2, 0, 1)[None].astype(np.float32)
    return np.ascontiguousarray(input_tensor), original_shape


def estimate_depth(session: ort.InferenceSession, image_bgr: np.ndarray) -> np.ndarray:
    input_tensor, original_shape = preprocess_depth(image_bgr)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: input_tensor})[0]
    depth = np.squeeze(output).astype(np.float32)
    depth = cv2.resize(
        depth,
        (original_shape[1], original_shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    return depth.astype(np.float32, copy=False)


def scale_depth_to_metric(
    depth_map: np.ndarray,
    road_polygon_relative: np.ndarray,
    image_shape: tuple
) -> np.ndarray:
    h, w = image_shape
    abs_poly = (road_polygon_relative * np.array([w, h], dtype=np.float32)).astype(np.float32)

    near_point = (abs_poly[0] + abs_poly[3]) * 0.5
    far_point = (abs_poly[1] + abs_poly[2]) * 0.5

    near_x = int(np.clip(round(near_point[0]), 0, w - 1))
    near_y = int(np.clip(round(near_point[1]), 0, h - 1))
    far_x = int(np.clip(round(far_point[0]), 0, w - 1))
    far_y = int(np.clip(round(far_point[1]), 0, h - 1))

    rel_near = float(depth_map[near_y, near_x])
    rel_far = float(depth_map[far_y, far_x])
    near_m = 2.0
    far_m = 15.0

    if abs(rel_near - rel_far) < 1e-6:
        fallback = depth_map.astype(np.float32) * 10.0
        return np.clip(fallback, 0.5, 50.0)

    denom = near_m - far_m
    b = (far_m * rel_far - near_m * rel_near) / denom
    a = near_m * (rel_near + b)

    if abs(a) < 1e-6:
        fallback = depth_map.astype(np.float32) * 10.0
        return np.clip(fallback, 0.5, 50.0)

    metric = a / (depth_map.astype(np.float32) + b + 1e-6)
    return np.clip(metric, 0.5, 50.0).astype(np.float32)
