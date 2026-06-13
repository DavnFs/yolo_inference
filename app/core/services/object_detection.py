# File: app/core/services/object_detection.py

import os
import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import onnxruntime as ort


# =============================================================================
# MODEL MANAGEMENT AND CONFIGURATION
# =============================================================================
CURRENT_MODEL_NAME = "FASTER-RCNN"
ONNX_SESSIONS = {}
ONNX_SESSION_ERRORS = {}
LAST_INFERENCE_ERROR = None
GEOMETRY_CACHE = {}

# Konstanta Geometri
ROAD_POLYGON_POINTS_RELATIVE = np.float32([(0.0, 1.0), (0.4, 0.55), (0.6, 0.55), (1.0, 1.0)])
BEV_WIDTH, BEV_HEIGHT = 200, 400
BEV_DESTINATION_POINTS = np.float32([[0, BEV_HEIGHT], [0, 0], [BEV_WIDTH, 0], [BEV_WIDTH, BEV_HEIGHT]])
PIXELS_PER_METER = 8.0  # Menyesuaikan rasio kedalaman yang lebih realistis (1 meter = 8 piksel BEV)
HAZARD_COLORS = {"danger": (0, 0, 255), "warning": (0, 255, 255), "safe": (0, 255, 0), "out_of_roi": (128, 128, 128)}

# Konstanta Model
FASTER_RCNN_CLASSES = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck',
                       'traffic light', 'fire hydrant', 'stop sign', 'bench', 'large rock', 'small rock',
                       'large trash', 'small trash']
IDX_TO_WEATHER = {0: 'sunny', 1: 'rainy', 2: 'foggy'}
YOLO_INPUT_WIDTH, YOLO_INPUT_HEIGHT = 640, 640
YOLO_CONF_THRESH, YOLO_NMS_THRESH = 0.25, 0.45

# Keywords in filename that identify a model as Faster-RCNN type
FASTER_RCNN_KEYWORDS = ["resnet", "faster", "rcnn", "fasterrcnn"]


# =============================================================================
# AUTO-DISCOVERY
# =============================================================================

def discover_models(models_dir: str) -> Dict:
    """
    Scans models_dir for .onnx files and builds model_configs automatically.
    - Files with 'resnet'/'faster'/'rcnn' in name → FASTER-RCNN type
    - Everything else → YOLO type (paired with labels.txt in same folder)
    """
    configs = {}
    labels_path = os.path.join(models_dir, "labels.txt")

    if not os.path.isdir(models_dir):
        print(f"[WARNING] Models directory not found: {models_dir}")
        return configs

    onnx_files = [
        f for f in os.listdir(models_dir)
        if f.lower().endswith(".onnx") and ".dml_optimized" not in f.lower()
    ]

    if not onnx_files:
        print(f"[WARNING] No .onnx files found in: {models_dir}")
        return configs

    for filename in sorted(onnx_files):
        full_path = os.path.join(models_dir, filename)
        # Generate a clean display name from filename
        # e.g. "yolov8n-aspp-cbam.onnx" → "YOLOV8N-ASPP-CBAM"
        model_key = os.path.splitext(filename)[0].upper()

        name_lower = filename.lower()
        is_faster_rcnn = any(kw in name_lower for kw in FASTER_RCNN_KEYWORDS)

        if is_faster_rcnn:
            configs[model_key] = {
                "path": full_path,
                "type": "FASTER-RCNN"
            }
        else:
            configs[model_key] = {
                "path": full_path,
                "type": "YOLO",
                "labels_path": labels_path
            }

        print(f"[AUTO-DISCOVER] Found model: '{model_key}' | type: {'FASTER-RCNN' if is_faster_rcnn else 'YOLO'} | {filename}")

    return configs


# =============================================================================
# MODEL SWITCHING
# =============================================================================

def set_active_model(model_name: str):
    global CURRENT_MODEL_NAME
    CURRENT_MODEL_NAME = model_name.upper()
    print(f"MODEL SWITCHED: Model aktif sekarang adalah {CURRENT_MODEL_NAME}")
    return True


def get_active_model_name() -> str:
    return CURRENT_MODEL_NAME


class DirectMLSession:
    """Wrap an ONNX session with IO binding to minimize CPU<->DML copies."""

    def __init__(self, session: ort.InferenceSession, device_id: int = 0):
        self.session = session
        self.device_type = "dml"
        self.device_id = device_id
        self.io_binding = session.io_binding()
        self._output_names = [o.name for o in session.get_outputs()]

    def get_inputs(self):
        return self.session.get_inputs()

    def get_outputs(self):
        return self.session.get_outputs()

    def get_providers(self):
        return self.session.get_providers()

    def run(self, feed_dict: Dict[str, np.ndarray], output_names: List[str] | None = None):
        self.io_binding.clear_binding_inputs()
        self.io_binding.clear_binding_outputs()

        # Keep references alive until run_with_iobinding() completes.
        bound_inputs = []
        for name, value in feed_dict.items():
            input_np = np.ascontiguousarray(value)
            ort_value = ort.OrtValue.ortvalue_from_numpy(input_np, self.device_type, self.device_id)
            bound_inputs.append(ort_value)
            self.io_binding.bind_ortvalue_input(name, ort_value)

        target_outputs = output_names if output_names is not None else self._output_names
        for name in target_outputs:
            self.io_binding.bind_output(name, self.device_type, self.device_id)

        self.session.run_with_iobinding(self.io_binding)
        return self.io_binding.copy_outputs_to_cpu()


def _apply_free_dimension_overrides(sess_options: ort.SessionOptions):
    overrides = {
        "batch": 1,
        "batch_size": 1,
        "height": YOLO_INPUT_HEIGHT,
        "width": YOLO_INPUT_WIDTH,
        "image_height": YOLO_INPUT_HEIGHT,
        "image_width": YOLO_INPUT_WIDTH,
    }
    for name, value in overrides.items():
        try:
            sess_options.add_free_dimension_override_by_name(name, value)
        except Exception:
            pass


def _build_session_options(use_gpu: bool, compatibility_mode: bool = False) -> ort.SessionOptions:
    sess_options = ort.SessionOptions()
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    if use_gpu:
        sess_options.enable_cpu_mem_arena = False
        sess_options.enable_mem_pattern = not compatibility_mode

        if compatibility_mode:
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        else:
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            _apply_free_dimension_overrides(sess_options)
            try:
                sess_options.enable_mem_reuse = True
            except Exception:
                pass

        # Let DirectML scheduling dominate; avoid CPU thread contention.
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1

        # Enforce GPU path: if any node cannot run on non-CPU EP, session init should fail.
        try:
            sess_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        except Exception as e:
            print(f"[ONNXRuntime] Warning: cannot disable CPU EP fallback: {e}")
    else:
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Ryzen 7 4800U: 8 physical cores, 16 threads
        cpu_count = os.cpu_count() or 8
        sess_options.intra_op_num_threads = max(1, min(cpu_count, 8))
        sess_options.inter_op_num_threads = 1

    return sess_options


def _build_providers(model_path: str, model_key: str):
    available = ort.get_available_providers()
    print(f"[ONNXRuntime] Available providers: {available}")

    providers = []
    use_gpu = False

    if "DmlExecutionProvider" in available:
        use_gpu = True
        providers.append((
            "DmlExecutionProvider",
            {
                "device_id": 0,
                "performance_preference": 0,
                "disable_metacommands": False,
            }
        ))
        print("[ONNXRuntime] Using DirectML-only mode (no CPU fallback).")
    else:
        print("[ONNXRuntime] DirectML not available. Using CPU only.")
        print("[ONNXRuntime] Make sure you installed: pip install onnxruntime-directml")
        if "AzureExecutionProvider" in available:
            print("[ONNXRuntime] Detected AzureExecutionProvider without DirectML.")
            print("[ONNXRuntime] This usually means 'onnxruntime' CPU package is overriding 'onnxruntime-directml'.")
            print("[ONNXRuntime] Fix: pip uninstall onnxruntime && pip install --force-reinstall onnxruntime-directml")
        providers.append("CPUExecutionProvider")

    return providers, use_gpu


def _build_optimized_model_path(model_path: str) -> str:
    root, ext = os.path.splitext(model_path)
    if root.lower().endswith(".dml_optimized"):
        return model_path
    return f"{root}.dml_optimized{ext}"


def _create_inference_session(model_path: str, sess_options: ort.SessionOptions, providers: list):
    try:
        return ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
    except Exception:
        if providers and isinstance(providers[0], tuple) and providers[0][0] == "DmlExecutionProvider":
            print("[ONNXRuntime] Retrying DirectML init with default provider options...")
            return ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=["DmlExecutionProvider"]
            )
        raise


def _run_session(session: Any, output_names: List[str] | None, inputs: Dict[str, np.ndarray]):
    if isinstance(session, DirectMLSession):
        return session.run(inputs, output_names=output_names)
    return session.run(output_names, inputs)


def _warmup_session(session: Any, model_key: str):
    raw_session = session.session if isinstance(session, DirectMLSession) else session
    if "DmlExecutionProvider" not in raw_session.get_providers():
        return

    input_meta = raw_session.get_inputs()
    if len(input_meta) != 1:
        return

    shape = []
    for i, dim in enumerate(input_meta[0].shape):
        if isinstance(dim, int) and dim > 0:
            shape.append(dim)
        elif i == 0:
            shape.append(1)
        elif i == 1:
            shape.append(3)
        elif i == 2:
            shape.append(YOLO_INPUT_HEIGHT)
        elif i == 3:
            shape.append(YOLO_INPUT_WIDTH)
        else:
            shape.append(1)

    if len(shape) != 4:
        return

    input_name = input_meta[0].name
    dummy = np.random.rand(*shape).astype(np.float32)

    print(f"[WARMUP] Running 3 warmup inferences for '{model_key}' (DirectML shader compile)...")
    for i in range(3):
        t0 = time.time()
        _run_session(session, None, {input_name: dummy})
        dt_ms = round((time.time() - t0) * 1000, 2)
        print(f"[WARMUP] {i + 1}/3: {dt_ms} ms")


def load_onnx_model_once(model_path: str, model_key: str):
    global ONNX_SESSIONS, ONNX_SESSION_ERRORS
    model_key = model_key.upper()
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    if model_key in ONNX_SESSION_ERRORS:
        raise RuntimeError(ONNX_SESSION_ERRORS[model_key])

    if model_key not in ONNX_SESSIONS:
        print(f"ONNX MODEL '{model_key}' NOT IN CACHE. Loading from: {model_path}")

        providers, use_gpu = _build_providers(model_path, model_key)
        init_profiles = [False, True] if use_gpu else [False]
        optimized_model_path = _build_optimized_model_path(model_path) if use_gpu else None
        last_error = None

        for compatibility_mode in init_profiles:
            sess_options = _build_session_options(use_gpu=use_gpu, compatibility_mode=compatibility_mode)
            session_model_path = model_path

            if use_gpu and not compatibility_mode:
                if optimized_model_path and os.path.exists(optimized_model_path):
                    session_model_path = optimized_model_path
                    print(f"[ONNXRuntime] Using cached optimized model: {optimized_model_path}")
                else:
                    try:
                        sess_options.optimized_model_filepath = optimized_model_path
                    except Exception as e:
                        print(f"[ONNXRuntime] Warning: cannot set optimized model output: {e}")

            try:
                raw_session = _create_inference_session(session_model_path, sess_options=sess_options, providers=providers)
                if use_gpu and "DmlExecutionProvider" in raw_session.get_providers():
                    ONNX_SESSIONS[model_key] = DirectMLSession(raw_session)
                    _warmup_session(ONNX_SESSIONS[model_key], model_key)
                else:
                    ONNX_SESSIONS[model_key] = raw_session
                break
            except Exception as e:
                last_error = e
                if use_gpu and not compatibility_mode:
                    print("[ONNXRuntime] Optimized DirectML init failed. Retrying compatibility profile...")
                else:
                    dml_mode = use_gpu
                    if dml_mode:
                        err_msg = (
                            f"DirectML gagal inisialisasi untuk model '{model_key}'. "
                            "Model ini kemungkinan tidak kompatibel dengan DirectML pada kombinasi driver/GPU/ONNX Runtime saat ini. "
                            "Coba model ONNX lain yang kompatibel DirectML atau export ulang model dengan graph yang lebih sederhana."
                        )
                    else:
                        err_msg = f"Gagal memuat model '{model_key}': {e}"

                    ONNX_SESSION_ERRORS[model_key] = err_msg
                    raise RuntimeError(err_msg) from last_error

        print(f"ONNX session for '{model_key}' loaded. Provider: {ONNX_SESSIONS[model_key].get_providers()}")
    return ONNX_SESSIONS[model_key]


# =============================================================================
# INFERENCE
# =============================================================================

def _decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    image_np = np.frombuffer(image_bytes, np.uint8)
    image_bgr = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Gagal decode image bytes.")
    return image_bgr


def get_faster_rcnn_prediction_from_frame(image_bgr: np.ndarray, model_path: str) -> Dict:
    session = load_onnx_model_once(model_path, "FASTER-RCNN")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    detection_input_np = np.ascontiguousarray(
        image_rgb.transpose(2, 0, 1).astype(np.float32, copy=False)
    )
    weather_rgb = cv2.resize(image_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    weather_input_np = np.expand_dims(
        np.ascontiguousarray(weather_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0),
        axis=0,
    )
    inputs = {
        'detection_input': detection_input_np,
        'weather_input': weather_input_np
    }

    outputs = _run_session(session, None, inputs)
    onnx_boxes, onnx_labels, onnx_scores, onnx_weather_logits = outputs

    detections = []
    for i, score in enumerate(onnx_scores):
        if score > 0.5:
            class_id = int(onnx_labels[i])
            label = FASTER_RCNN_CLASSES[class_id] if class_id < len(FASTER_RCNN_CLASSES) else f"class_{class_id}"
            detections.append({
                "label": label,
                "score": round(float(score), 4),
                "bounding_box": [int(coord) for coord in onnx_boxes[i]]
            })

    weather_logits = np.squeeze(onnx_weather_logits)
    if weather_logits.ndim == 1:
        weather_idx = int(np.argmax(weather_logits))
    else:
        weather_idx = int(np.bincount(np.argmax(weather_logits, axis=0).ravel()).argmax())
    weather_label = IDX_TO_WEATHER.get(weather_idx, "unknown")
    return {"detections": detections, "weather_prediction": weather_label}


def get_faster_rcnn_prediction(image_bytes: bytes, model_path: str) -> Dict:
    return get_faster_rcnn_prediction_from_frame(_decode_image_bytes(image_bytes), model_path)


def load_yolo_labels(labels_path: str):
    pass # Deprecated, auto-loading is now used

def preprocess_yolo(image_bgr: np.ndarray):
    # Letterbox seperti Ultralytics, tetapi tetap di BGR sampai blobFromImage.
    shape = image_bgr.shape[:2]  # [height, width]
    new_shape = (YOLO_INPUT_HEIGHT, YOLO_INPUT_WIDTH)
    
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        resized = cv2.resize(image_bgr, new_unpad, interpolation=cv2.INTER_LINEAR)
    else:
        resized = image_bgr
        
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    padded_img = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    input_tensor = cv2.dnn.blobFromImage(
        padded_img,
        scalefactor=1.0 / 255.0,
        size=(YOLO_INPUT_WIDTH, YOLO_INPUT_HEIGHT),
        swapRB=True,
        crop=False,
    )
    return np.ascontiguousarray(input_tensor), r, dw, dh, shape[1], shape[0]


def postprocess_yolo(output, scale, dw, dh, labels, orig_w: int, orig_h: int):
    predictions = np.squeeze(output)
    if predictions.ndim != 2:
        return []
    if predictions.shape[0] < predictions.shape[1] and predictions.shape[0] <= 256:
        predictions = predictions.T

    if predictions.shape[1] < 5:
        return []

    if len(labels) and predictions.shape[1] == len(labels) + 5:
        class_scores = predictions[:, 5:] * predictions[:, 4:5]
    else:
        class_scores = predictions[:, 4:]

    if class_scores.size == 0:
        return []

    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(class_scores.shape[0]), class_ids]
    keep = scores > YOLO_CONF_THRESH
    if not np.any(keep):
        return []

    boxes = predictions[keep, :4]
    scores = scores[keep].astype(np.float32, copy=False)
    class_ids = class_ids[keep].astype(np.int32, copy=False)

    cx, cy, bw, bh = boxes.T
    x1 = np.clip((cx - bw / 2 - dw) / scale, 0, orig_w - 1)
    y1 = np.clip((cy - bh / 2 - dh) / scale, 0, orig_h - 1)
    x2 = np.clip((cx + bw / 2 - dw) / scale, 0, orig_w - 1)
    y2 = np.clip((cy + bh / 2 - dh) / scale, 0, orig_h - 1)

    boxes_orig = np.stack((x1, y1, x2, y2), axis=1).astype(np.int32)
    boxes_nms = np.stack((x1, y1, x2 - x1, y2 - y1), axis=1).astype(np.int32)
    valid = (boxes_nms[:, 2] > 0) & (boxes_nms[:, 3] > 0)
    if not np.any(valid):
        return []

    boxes_orig = boxes_orig[valid]
    boxes_nms = boxes_nms[valid]
    scores = scores[valid]
    class_ids = class_ids[valid]

    indices = cv2.dnn.NMSBoxes(boxes_nms.tolist(), scores.tolist(), YOLO_CONF_THRESH, YOLO_NMS_THRESH)
    final_detections = []
    if len(indices) > 0:
        for i in np.asarray(indices).reshape(-1):
            class_id = class_ids[i]
            label = labels[class_id] if class_id < len(labels) else f"class_{class_id}"
            final_detections.append({
                "label": label,
                "score": round(float(scores[i]), 4),
                "bounding_box": boxes_orig[i].tolist()
            })
    return final_detections


MODEL_LABELS_CACHE = {}

def get_yolo_labels(session, labels_path: str, model_key: str):
    import ast
    if model_key in MODEL_LABELS_CACHE:
        return MODEL_LABELS_CACHE[model_key]
        
    sess = session.session if isinstance(session, DirectMLSession) else session
    try:
        meta_dict = sess.get_modelmeta().custom_metadata_map
        if 'names' in meta_dict:
            names_dict = ast.literal_eval(meta_dict['names'])
            labels = [names_dict[i] for i in range(len(names_dict))]
            MODEL_LABELS_CACHE[model_key] = labels
            print(f"[LABELS] Auto-extracted {len(labels)} classes from ONNX metadata for {model_key}")
            return labels
    except Exception as e:
        print(f"[LABELS] Failed to parse metadata for {model_key}: {e}")

    # Fallback to file reading if no metadata available
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"YOLO labels file not found at: {labels_path}")
    with open(labels_path, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    MODEL_LABELS_CACHE[model_key] = labels
    return labels

def get_yolo_prediction_from_frame(image_bgr: np.ndarray, model_path: str, labels_path: str) -> Dict:
    model_key = get_active_model_name()
    session = load_onnx_model_once(model_path, model_key)
    labels = get_yolo_labels(session, labels_path, model_key)
    input_tensor, scale, dw, dh, orig_w, orig_h = preprocess_yolo(image_bgr)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    outputs = _run_session(session, [output_name], {input_name: input_tensor})
    detections = postprocess_yolo(outputs[0], scale, dw, dh, labels, orig_w, orig_h)
    return {"detections": detections, "weather_prediction": "not_applicable"}


def get_yolo_prediction(image_bytes: bytes, model_path: str, labels_path: str) -> Dict:
    return get_yolo_prediction_from_frame(_decode_image_bytes(image_bytes), model_path, labels_path)


def _run_path_planning(
    results: Dict,
    frame_bgr: np.ndarray,
    external_depth: np.ndarray
) -> Dict:
    try:
        from app.core.services.path_planning import compute_path

        h, w = frame_bgr.shape[:2]

        abs_poly = (
            ROAD_POLYGON_POINTS_RELATIVE *
            np.array([w, h], dtype=np.float32)
        ).astype(np.int32)

        if any("hazard_level" not in det for det in results.get("detections", [])):
            analyzed_dets, _ = perform_road_analysis((h, w), results.get("detections", []))
            results["detections"] = analyzed_dets

        results["path_planning"] = compute_path(
            results["detections"], external_depth, (h, w), abs_poly
        )
    except Exception as e:
        results["path_planning"] = {
            "path_found": False,
            "waypoints": [],
            "error": str(e),
        }
    return results


def get_combined_prediction_from_frame(
    frame_bgr: np.ndarray,
    model_configs: Dict,
    use_mc_dropout: bool = False,
    pt_path: str = None,
    n_samples: int = 5,
    enable_path_planning: bool = False,
    external_depth: np.ndarray = None
) -> Dict:
    global LAST_INFERENCE_ERROR
    start_time = time.perf_counter()
    active_model_key = get_active_model_name()
    config = model_configs.get(active_model_key)
    if not config:
        return {"error": f"Konfigurasi untuk model '{active_model_key}' tidak ditemukan."}

    model_path = config["path"]
    model_type = config["type"]

    if use_mc_dropout and pt_path and model_type == "YOLO":
        try:
            from app.core.services.mc_dropout import load_pt_model, mc_inference

            pt_model, ok = load_pt_model(pt_path)
            if ok:
                mc_result = mc_inference(pt_model, frame_bgr, n_samples=n_samples)
                mc_result["weather_prediction"] = "N/A"
                mc_result["metrics"] = {"processing_latency_ms": 0}
                if enable_path_planning and external_depth is not None:
                    _run_path_planning(mc_result, frame_bgr, external_depth)
                return mc_result
        except Exception as e:
            print(f"[MC Dropout] Falling back to ONNX inference: {e}")

    try:
        if model_type == "FASTER-RCNN":
            results = get_faster_rcnn_prediction_from_frame(frame_bgr, model_path)
        elif model_type == "YOLO":
            results = get_yolo_prediction_from_frame(frame_bgr, model_path, config["labels_path"])
        else:
            return {"error": f"Tipe model '{model_type}' tidak dikenal."}
        LAST_INFERENCE_ERROR = None
    except Exception as e:
        end_time = time.perf_counter()
        err_msg = str(e)
        if err_msg != LAST_INFERENCE_ERROR:
            print(f"[INFERENCE ERROR] {err_msg}")
            LAST_INFERENCE_ERROR = err_msg
        return {
            "error": err_msg,
            "detections": [],
            "weather_prediction": "not_applicable",
            "metrics": {"processing_latency_ms": round((end_time - start_time) * 1000, 2)}
        }

    end_time = time.perf_counter()
    results["metrics"] = {"processing_latency_ms": round((end_time - start_time) * 1000, 2)}
    if enable_path_planning and external_depth is not None:
        _run_path_planning(results, frame_bgr, external_depth)
    return results


def get_combined_prediction(image_bytes: bytes, model_configs: Dict) -> Dict:
    start_time = time.perf_counter()
    try:
        frame_bgr = _decode_image_bytes(image_bytes)
    except Exception as e:
        end_time = time.perf_counter()
        return {
            "error": str(e),
            "detections": [],
            "weather_prediction": "not_applicable",
            "metrics": {"processing_latency_ms": round((end_time - start_time) * 1000, 2)}
        }
    return get_combined_prediction_from_frame(frame_bgr, model_configs)


# =============================================================================
# ANALYSIS AND VISUALIZATION
# =============================================================================

def _get_frame_geometry(frame_shape: tuple) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = frame_shape
    cache_key = (h, w)
    if cache_key not in GEOMETRY_CACHE:
        abs_polygon_points = (ROAD_POLYGON_POINTS_RELATIVE * [w, h]).astype(np.float32)
        perspective_matrix = cv2.getPerspectiveTransform(abs_polygon_points, BEV_DESTINATION_POINTS)
        inv_perspective_matrix = cv2.invert(perspective_matrix)[1]
        GEOMETRY_CACHE[cache_key] = (abs_polygon_points, perspective_matrix, inv_perspective_matrix)
    return GEOMETRY_CACHE[cache_key]


def perform_road_analysis(frame_shape: tuple, detections: list) -> Tuple[list, np.ndarray]:
    abs_polygon_points, perspective_matrix, _ = _get_frame_geometry(frame_shape)
    final_detections = []

    if not detections:
        return final_detections, abs_polygon_points

    ref_points = []
    for det in detections:
        x1, y1, x2, y2 = det['bounding_box']
        ref_points.append([(x1 + x2) * 0.5, y2])

    transformed_points = cv2.perspectiveTransform(
        np.array(ref_points, dtype=np.float32).reshape(-1, 1, 2),
        perspective_matrix,
    ).reshape(-1, 2)

    for det in detections:
        x1, y1, x2, y2 = det['bounding_box']
        hazard_level, distance_meters = "out_of_roi", None

        transformed_point = transformed_points[len(final_detections)]
        distance_pixels = BEV_HEIGHT - transformed_point[1]
        distance_meters = max(0, float(distance_pixels)) / PIXELS_PER_METER

        bbox_points = np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            dtype=np.float32,
        )
        try:
            intersection_area, _ = cv2.intersectConvexConvex(abs_polygon_points, bbox_points)
        except cv2.error:
            intersection_area = 0

        if intersection_area > 0:
            if distance_meters is not None:
                if distance_meters <= 15:
                    hazard_level = "danger"
                elif 15 < distance_meters <= 35:
                    hazard_level = "warning"
                else:
                    hazard_level = "safe"
            else:
                hazard_level = "safe"
                
        det.update({
            "hazard_level": hazard_level,
            "distance_m": round(distance_meters, 2) if distance_meters is not None else None
        })
        final_detections.append(det)
    return final_detections, abs_polygon_points


def draw_main_visualization(
    frame: np.ndarray,
    analysis_results: Dict,
    abs_polygon_points: np.ndarray,
    show_uncertainty: bool = False,
    path_data=None,
    frame_shape: tuple = None
) -> np.ndarray:
    frame_h, frame_w = frame.shape[:2]
    cv2.polylines(frame, [np.int32(abs_polygon_points)], isClosed=True, color=(0, 255, 255), thickness=2)
    for det in analysis_results.get("detections", []):
        x1, y1, x2, y2 = det['bounding_box']
        x1 = max(0, min(frame_w - 1, int(x1)))
        y1 = max(0, min(frame_h - 1, int(y1)))
        x2 = max(0, min(frame_w - 1, int(x2)))
        y2 = max(0, min(frame_h - 1, int(y2)))
        if x2 <= x1 or y2 <= y1:
            continue
        hazard_level = det.get('hazard_level', 'out_of_roi')
        color = HAZARD_COLORS.get(hazard_level, (128, 128, 128))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label_text = f"{det['label']} {det['score']:.2f}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_top = max(0, y1 - th - 10)
        label_right = min(frame_w - 1, x1 + tw)
        label_y = max(th + 2, y1 - 5)
        cv2.rectangle(frame, (x1, label_top), (label_right, y1), color, -1)
        cv2.putText(frame, label_text, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        if show_uncertainty and "uncertainty" in det:
            unc = det["uncertainty"]
            if unc < 0.05:
                unc_color = (0, 255, 0)
            elif unc < 0.15:
                unc_color = (0, 255, 255)
            else:
                unc_color = (0, 0, 255)
            cv2.putText(frame, f"u={unc:.3f}", (x1, min(frame_h - 5, y2 + 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, unc_color, 1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), unc_color, 2)
        if hazard_level != 'out_of_roi' and det.get("distance_m") is not None:
            dist_text = f"{det['distance_m']}m"
            cv2.putText(frame, dist_text, (x1, max(th + 2, label_top - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    if path_data and path_data.get("path_found"):
        waypoints = path_data.get("waypoints", [])
        if len(waypoints) >= 2 and frame_shape is not None:
            # Use inverse perspective matrix to map BEV waypoints back to camera frame
            _, _, M_inv = _get_frame_geometry(frame_shape)
            bev_pts = np.array(waypoints, dtype=np.float32).reshape(-1, 1, 2)
            camera_pts = cv2.perspectiveTransform(bev_pts, M_inv).reshape(-1, 2)

            projected_points = []
            for pt in camera_pts:
                ix, iy = int(round(pt[0])), int(round(pt[1]))
                if 0 <= ix < frame_w and 0 <= iy < frame_h:
                    projected_points.append((ix, iy))

            if len(projected_points) >= 2:
                cv2.polylines(
                    frame,
                    [np.array(projected_points, dtype=np.int32)],
                    isClosed=False,
                    color=(0, 255, 255),
                    thickness=3,
                )
            for point in projected_points:
                cv2.circle(frame, point, 4, (0, 200, 255), -1)
            if projected_points:
                label_x, label_y = projected_points[0]
                cv2.putText(frame, "PATH", (label_x, max(15, label_y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    return frame
