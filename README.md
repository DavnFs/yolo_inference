# YOLO Inference — Autonomous Driving Perception & Path Planning Dashboard

A real-time autonomous driving perception system built in Python/Tkinter that performs multi-model object detection (YOLO / Faster-RCNN), Bird's Eye View (BEV) occupancy grid mapping, A* path planning with C-space inflation, and Monte Carlo Dropout uncertainty estimation. Uses hardware-level stereo depth matrices (DrivingStereo dataset) for accurate metric distance measurement.

---

## Table of Contents

- [System Architecture](#system-architecture)
- [Directory Structure](#directory-structure)
- [Prerequisites & Installation](#prerequisites--installation)
- [How to Run](#how-to-run)
- [Module Deep-Dive](#module-deep-dive)
  - [gui_main.py — Dashboard & Data Ingestion](#gui_mainpy--dashboard--data-ingestion)
  - [object_detection.py — Perception Engine](#object_detectionpy--perception-engine)
  - [path_planning.py — BEV Grid & A* Path Planner](#path_planningpy--bev-grid--a-path-planner)
  - [mc_dropout.py — Uncertainty Estimation](#mc_dropoutpy--uncertainty-estimation)
  - [convert.py — Model Export & DirectML Optimization](#convertpy--model-export--directml-optimization)
- [Key Constants & Configuration](#key-constants--configuration)
- [Data Flow Pipeline](#data-flow-pipeline)
- [Custom Model Architecture](#custom-model-architecture)
- [Stereo Depth Pipeline](#stereo-depth-pipeline)
- [Simulation vs. Hardware](#simulation-vs-hardware)
- [Path Planning Algorithm](#path-planning-algorithm)
- [BEV & OGM Visualization](#bev--ogm-visualization)
- [Model Management & Auto-Discovery](#model-management--auto-discovery)
- [GPU Acceleration (DirectML)](#gpu-acceleration-directml)
- [Benchmarking](#benchmarking)
- [Troubleshooting](#troubleshooting)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      gui_main.py (Tkinter Dashboard)                │
│  ┌──────────────┐  ┌──────────────────┐  ┌───────────────────────┐ │
│  │  Video Panel  │  │  BEV + OGM Panel  │  │   Control Panel      │ │
│  │  (main_canvas)│  │  (bev_canvas +    │  │   - Model selector   │ │
│  │               │  │   ogm_canvas)     │  │   - Mode selector    │ │
│  │               │  │                   │  │   - MC Dropout       │ │
│  │               │  │                   │  │   - Path Planning    │ │
│  └──────┬───────┘  └────────▲──────────┘  │   - Theme toggle     │ │
│         │                   │              └───────────────────────┘ │
└─────────┼───────────────────┼──────────────────────────────────────┘
          │                   │
          ▼                   │
┌─────────────────────────────┼──────────────────────────────────────┐
│           Perception Pipeline (object_detection.py)                │
│                                                                    │
│  ┌────────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │ Model Loading   │───▶│  Inference   │───▶│ Post-Processing  │   │
│  │ (ONNX/DirectML) │    │  (YOLO/RCNN) │    │ (NMS, labels)    │   │
│  └────────────────┘    └──────────────┘    └────────┬─────────┘   │
│                                                      │             │
│  ┌──────────────────────────────────────────────────▼──────────┐  │
│  │ perform_road_analysis() — hazard_level + distance_m         │  │
│  │ _run_path_planning()    — stereo depth override → compute   │  │
│  └─────────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│           Path Planning (path_planning.py)                           │
│                                                                      │
│  build_occupancy_grid()         compute_path()                       │
│  ┌────────────────────┐        ┌─────────────────────────────────┐  │
│  │ 1. Pinhole project  │        │ 1. Emergency stop check         │  │
│  │    detections → BEV │        │ 2. ACC / overtake / cruise      │  │
│  │ 2. C-space inflate  │───┐    │ 3. Goal selection + snapping    │  │
│  │ 3. Clear hood zone  │   │    │ 4. A* search                    │  │
│  └────────────────────┘   │    │ 5. Path smoothing               │  │
│                            └───▶└─────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
offline-monit-app/
├── gui_main.py                     # Main entry point — Tkinter dashboard
├── extract_labels*.py              # Utility scripts for label extraction
│
├── app/
│   ├── core/
│   │   └── services/
│   │       ├── object_detection.py # Perception: model loading, inference, BEV projection
│   │       ├── path_planning.py    # Occupancy grid, A*, goal selection, smoothing
│   │       └── mc_dropout.py       # Monte Carlo Dropout uncertainty estimation
│   │
│   └── models/                     # ONNX model files (auto-discovered)
│       ├── best.onnx               # YOLO model
│       ├── best.dml_optimized.onnx # DirectML-optimized variant
│       ├── best_triplehead.onnx    # Triple-head YOLO (weather + detection)
│       ├── Resnet50.onnx           # Faster-RCNN backbone
│       └── ...                     # Additional model variants
│
├── convert/
│   ├── convert.py                  # PyTorch → ONNX export + DirectML optimization
│   ├── custom_modules.py           # Custom module definitions for export
│   ├── best.pt                     # PyTorch checkpoint
│   └── best_triplehead.pt          # Triple-head PyTorch checkpoint
│
├── stereo_dataset/                 # DrivingStereo dataset (not in git)
│   ├── left_images/                # RGB left camera frames (.png)
│   ├── disparity/                  # 16-bit disparity maps (.png)
│   └── image_R/                    # Right camera frames (unused)
│
├── assets/                         # Sample video (output_640.mp4)
├── output/                         # Generated output directory
└── README.md                       # This file
```

---

## Prerequisites & Installation

### Python Version
- Python 3.12+ (tested with 3.13.14)

### Core Dependencies

```powershell
pip install onnxruntime-directml opencv-python numpy pillow ultralytics torch onnx onnxslim
```

> **Important:** Install `onnxruntime-directml` **instead of** `onnxruntime`. They cannot coexist — the CPU package overrides the DirectML provider. If you already have `onnxruntime` installed:
> ```powershell
> pip uninstall onnxruntime onnxruntime-directml
> pip install onnxruntime-directml
> ```

### Stereo Dataset
Place the DrivingStereo dataset in `stereo_dataset/` with this structure:
```
stereo_dataset/
├── left_images/
│   ├── 000001.png
│   ├── 000002.png
│   └── ...
├── disparity/
│   ├── 000001.png    # 16-bit disparity maps
│   ├── 000002.png
│   └── ...
└── image_R/          # (optional, not used)
```

Stereo camera intrinsics for DrivingStereo:
- `fx = 721.53` (focal length in pixels, native width 1242px)
- `baseline = 0.54` meters

---

## How to Run

```powershell
# From project root
python gui_main.py
```

The dashboard launches in fullscreen. Use **Escape** to exit fullscreen.

### Input Modes

| Mode | Source | Depth Source | Path Planning |
|------|--------|--------------|---------------|
| **Stereo Dataset Sim** | `stereo_dataset/left_images/` | Stereo disparity → metric depth | Full A* enabled |
| **Video File** | Any video file (.mp4, .avi, etc.) | None (detection only) | Disabled |
| **Webcam** | System camera (index 0) | None (detection only) | Disabled |
| **Image File** | Single image | None (detection only) | Disabled |

> **Note:** Path planning requires a stereo depth matrix (`external_depth`). Only **Stereo Dataset Sim** mode provides this. In other modes, the pipeline runs detection and visualization only.

### Workflow
1. Select a **Model** from the dropdown (auto-discovered from `app/models/`)
2. Select **Stereo Dataset Sim** mode
3. Click **Browse** to select the `stereo_dataset/` folder
4. Toggle **MC Dropout** for uncertainty estimation (requires `.pt` checkpoint)
5. Toggle **Path Planning** to enable BEV occupancy grid + A* navigation
6. Click **Start Processing**

---

## Module Deep-Dive

### gui_main.py — Dashboard & Data Ingestion

**Classes:**

**`StereoSimLoader`** — Loads DrivingStereo left_images + disparity pairs.
- Reads 16-bit disparity PNGs, converts to metric depth: `depth_m = (baseline × fx) / disparity`
- Clips depth to `[0.5, 40.0]` meters
- Downsamples both frame and depth to 640px max width for performance
- Frame: `INTER_LINEAR` (visual quality), Depth: `INTER_NEAREST` (preserves discrete distances)

**`DashboardGUI`** — Tkinter application with 3-column layout:
- **Column 0 (weight=5):** Main video canvas — shows annotated detection frame
- **Column 1 (weight=3):** Mapping panel — BEV canvas (top) + OGM canvas (bottom)
- **Column 2 (weight=2):** Control panel — model/mode selectors, toggles, status cards

**Key rendering methods:**
- `_flush_canvas_update()` — Thread-safe Tkinter photo reference update for main video
- `_flush_bev_update()` — BEV dot overlay using pinhole projection (matches `build_occupancy_grid` math)
- `_flush_ogm_update()` — Renders obstacle grid as RGB image with A* waypoints overlaid via `cv2.line`

**Thread safety:** All canvas updates use `master.after(0, ...)` to marshal from the processing thread to the Tkinter main thread. Pending-update flags prevent queue flooding.

---

### object_detection.py — Perception Engine

**Constants (shared across pipeline):**
```python
BEV_WIDTH, BEV_HEIGHT = 200, 400       # BEV grid dimensions (pixels)
PIXELS_PER_METER = 8.0                 # 1 meter = 8 BEV pixels
STEREO_FX = 721.53                     # DrivingStereo focal length (1242px native)
ROAD_POLYGON_POINTS_RELATIVE = [...]   # Trapezoid ROI (fractional coordinates)
```

**Model Auto-Discovery (`discover_models`):**
- Scans `app/models/` for `.onnx` files
- Classifies by filename: `resnet`/`faster`/`rcnn` → FASTER-RCNN type, everything else → YOLO
- YOLO models require a `labels.txt` in the same directory (or auto-extracted from ONNX metadata)

**DirectML GPU Acceleration:**
- `DirectMLSession` wraps ONNX sessions with IO binding to minimize CPU↔GPU copies
- `_build_providers()` detects `DmlExecutionProvider` availability
- `_build_optimized_model_path()` generates `.dml_optimized.onnx` path
- Automatic fallback: DirectML → compatibility mode → CPU
- Warmup inference (3 passes) to compile DirectML shaders

**Inference Pipeline:**
1. `get_combined_prediction_from_frame()` — entry point, branches by model type
2. For YOLO: `preprocess_yolo()` → `session.run()` → `postprocess_yolo()` (NMS)
3. For Faster-RCNN: `get_faster_rcnn_prediction_from_frame()` with raw ONNX output parsing
4. Optional MC Dropout path: `mc_inference()` with stochastic forward passes

**Road Analysis (`perform_road_analysis`):**
- Perspective-transforms detection reference points to BEV space
- Assigns `hazard_level` (danger/warning/safe/out_of_roi) and `distance_m` per detection
- Caches geometry (perspective matrix, inverse) per frame resolution

**Path Planning Integration (`_run_path_planning`):**
- Overrides homography-based distance with true stereo depth (percentile sampling from bounding box patch)
- Re-evaluates hazard levels: danger ≤ 15m, warning ≤ 35m, safe > 35m
- Calls `compute_path()` from `path_planning.py`

**Visualization (`draw_main_visualization`):**
- Draws road polygon overlay, bounding boxes (color-coded by hazard), labels, distance
- Projects A* waypoints back to camera frame using inverse perspective transform (`M_inv`)
- Optional MC Dropout uncertainty circles around detections

---

### path_planning.py — BEV Grid & A* Path Planner

**Module-Level Constants:**
```python
_DRIVINGSTEREO_NATIVE_W = 1242     # Native width for fx calibration
_EGO_WIDTH_M = 1.8                 # Ego vehicle width (meters)
_EGO_LENGTH_M = 2.5                # Obstacle box length (meters)
_OBS_MIN_WIDTH_PX = 4              # Minimum obstacle width (pixels)
_INFLATION_RADIUS_M = 0.6          # C-space Minkowski sum padding
_HOOD_CLEAR_ROWS = 40              # Bottom rows cleared after inflation
_EMERGENCY_STOP_ROWS = 40          # Rows scanned for emergency stop
_EMERGENCY_STOP_THRESHOLD = 0.7    # Fraction blocked to trigger stop
_CRUISE_DIST_M = 20.0              # ACC vs overtake distance threshold
_LANE_CHECK_STRIP_PX = 40         # Adjacent-lane check width
_OVERTAKE_LATERAL_PX = 44          # Overtake goal lateral offset (5.5m)
_OVERTAKE_CLEARANCE_PX = 40        # Clearance past obstacle for overtake
_ACC_BACKSTOP_PX = 80              # ACC goal behind obstacle
_TURN_PENALTY = 0.5                # A* heading-change penalty
_GOAL_SNAP_RADIUS = 5              # Goal snapping search radius
```

**`build_occupancy_grid()`:**
1. Computes dynamic focal length: `fx = STEREO_FX × (orig_w / 1242.0)`
2. Projects each YOLO detection to BEV using pinhole camera model:
   - `X_meters = ((bbox_cx - cx) × dist) / fx`
   - `x_bev = BEV_WIDTH//2 + X_meters × PIXELS_PER_METER`
3. Draws filled rectangles at projected positions
4. Applies C-space inflation (Minkowski sum) with 11×11 kernel (~0.6m padding)
5. Clears bottom 40 rows (ego hood noise zone)

**`astar()`:**
- 8-connected grid search with Euclidean heuristic
- Proximity clearance penalty: `(cell_val / 255)² × 15.0` — exponentially penalizes cells near obstacles
- Heading turn penalty: `0.5` per direction change — encourages smooth curves
- Heap tuple: `(f_score, g_score, (x,y), (dx,dy))` — tracks parent direction for turn penalty
- Max iterations: `BEV_WIDTH × BEV_HEIGHT` (safety cap)

**`compute_path()` — Goal Selection Logic:**

| Condition | Goal Strategy |
|-----------|---------------|
| Emergency stop (front 40 rows ≥ 70% blocked) | Return `path_found: False`, single stop point |
| Ego lane clear | Goal at horizon `(center_x, 15)` with snapping |
| Obstacle far (> 20m) — ACC | Goal between ego and obstacle (`closest_obs_y + 80px`) |
| Obstacle close (≤ 20m) — Overtake | Goal at `(center_x ± 44, overtake_y)` if adjacent lane clear |
| Both lanes blocked | Stop behind obstacle |

**Goal Snapping (`_snap_goal`):**
If a goal lands on an obstacle boundary (inflated cell), searches a 5px radius for the nearest free cell. Prevents A* from failing due to goal-in-obstacle.

**Start Bias:**
Start points are reordered based on goal direction — if goal is right of center, `(center_x + 20, 390)` is tried first. This seeds A* with the correct initial heading for overtaking.

**`smooth_path()`:**
- Moving average with window=5
- Downsampled to max 20 waypoints for clean visualization

---

### mc_dropout.py — Uncertainty Estimation

Enables stochastic forward passes by replacing `BatchNorm2d` with `Dropout2d` at target layers (18, 21, 24) during inference.

**Custom Modules (must match convert.py):**
- `SimAM` — Spatial attention module (energy-based)
- `ASPP` — Atrous Spatial Pyramid Pooling (multi-scale context)
- `FPN` — Feature Pyramid Network (top-down refinement)
- `PANet` — Path Aggregation Network (bottom-up augmentation)

**`mc_inference()`:**
- Loads `.pt` checkpoint, applies dropout modifications
- Runs N forward passes with stochastic dropout active
- Returns mean bounding boxes + variance (uncertainty) per detection

---

### convert.py — Model Export & DirectML Optimization

Exports PyTorch `.pt` models to ONNX format and optionally optimizes for DirectML.

```powershell
cd convert
python convert.py
```

**Steps:**
1. Registers custom modules (SimAM, ASPP, FPN, PANet) into Ultralytics namespace
2. Loads `.pt` checkpoint via `YOLO()`
3. Exports to ONNX with `opset=12`, `simplify=True`, `batch=1`
4. Runs DirectML graph optimization pass (`onnxruntime.transformers.optimizer`)
5. Saves optimized model as `best_triplehead.dml_optimized.onnx`

---

## Key Constants & Configuration

### Shared Geometry Constants (object_detection.py)

| Constant | Value | Description |
|----------|-------|-------------|
| `BEV_WIDTH` | 200 | BEV grid width (pixels) = 25m lateral |
| `BEV_HEIGHT` | 400 | BEV grid height (pixels) = 50m forward |
| `PIXELS_PER_METER` | 8.0 | Spatial resolution: 1m = 8px |
| `STEREO_FX` | 721.53 | DrivingStereo focal length (1242px native) |
| `ROAD_POLYGON_POINTS_RELATIVE` | `[(0.0,1.0), (0.4,0.55), (0.6,0.55), (1.0,1.0)]` | Trapezoid road ROI |

### Focal Length Scaling

All three pipeline stages (stereo loader, occupancy grid, BEV canvas) use the same formula:

```python
fx = STEREO_FX * (orig_w / 1242.0)
```

For 640px downscaled frames: `fx = 721.53 × (640/1242) ≈ 371.6`

### Hazard Classification Thresholds

| Level | Distance | Color |
|-------|----------|-------|
| `danger` | ≤ 15m | Red |
| `warning` | ≤ 35m | Cyan/Yellow |
| `safe` | > 35m | Green |
| `out_of_roi` | Outside polygon | Gray |

---

## Data Flow Pipeline

```
StereoSimLoader
    │ (left_images/*.png + disparity/*.png)
    ▼
depth_m = (baseline × fx) / disparity    ← metric stereo depth
Downscale to 640px width
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ 1. PERCEPTION (object_detection.py)                  │
│    YOLO/Faster-RCNN → bounding boxes + labels        │
│    perform_road_analysis → hazard_level + distance_m  │
│    _run_path_planning → override with stereo depth    │
│    Percentile sampling from bbox patch for true dist  │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│ 2. OCCUPANCY GRID (path_planning.py)                 │
│    Pinhole project detections → BEV rectangles       │
│    C-space inflate (11×11 kernel, ~0.6m padding)     │
│    Clear hood zone (bottom 40 rows)                  │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│ 3. GOAL SELECTION (compute_path)                     │
│    Emergency stop check (front 40 rows)              │
│    ACC / Overtake / Cruise logic                     │
│    Goal snapping if on obstacle boundary             │
│    Start bias toward goal direction                  │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│ 4. A* SEARCH (astar)                                 │
│    8-connected grid, Euclidean heuristic             │
│    Proximity clearance penalty                       │
│    Heading turn penalty (0.5)                        │
│    Max iterations: 200×400 = 80,000                  │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│ 5. SMOOTHING + VISUALIZATION                         │
│    Moving average (window=5), downsample to 20 pts   │
│    Overlay on BEV canvas (pinhole back-projection)   │
│    Overlay on OGM canvas (cv2.line on grid image)    │
└─────────────────────────────────────────────────────┘
```

---

## Custom Model Architecture

Models use YOLOv11 with custom attention and multi-scale modules:

```
Input (640×640×3)
    │
    ├─ Backbone: CSPDarknet + SimAM (spatial attention)
    │
    ├─ Neck: ASPP (atrous rates 1,6,12,18) → FPN (top-down) → PANet (bottom-up)
    │
    └─ Head: Triple-head output
         ├─ Detection head (bounding boxes + classes)
         ├─ Weather classification head
         └─ (Optional) MC Dropout layers at indices 18, 21, 24
```

**SimAM (Simple Attention Module):**
Energy-based spatial attention that computes attention weights from feature statistics without额外 parameters:
```python
y = (x - μ)² / (4 × (σ + ε)) + 0.5
return x × sigmoid(y)
```

**ASPP (Atrous Spatial Pyramid Pooling):**
Parallel dilated convolutions at rates (1, 6, 12, 18) + global average pooling branch. Captures multi-scale context without increasing parameters significantly.

---

## Stereo Depth Pipeline

The sole depth source is the DrivingStereo dataset's 16-bit disparity maps:

```
16-bit Disparity PNG
    │
    ▼
disparity = raw / 256.0          # Convert from fixed-point
disparity = max(disparity, 0.1)  # Guard against division by zero
depth_m = (0.54 × 721.53) / disparity
depth_m = clip(depth_m, 0.5, 40.0)
    │
    ▼
Downscale to 640px width (INTER_NEAREST for depth)
    │
    ▼
Passed as external_depth to _run_path_planning()
    │
    ▼
Per-detection override: sample 20th percentile from bbox patch
→ replaces homography-based distance with true stereo distance
```

**Why stereo depth is mandatory:**
- Monocular depth estimation (Depth Anything V2) produces relative depth, not metric distance
- Stereo disparity directly yields physical distance via the calibrated baseline formula
- The path planning grid requires accurate meter-scale distances for BEV projection, hazard classification, and obstacle sizing

---

## Simulation vs. Hardware

This repository is a **simulation harness for an undergraduate thesis on autonomous driving perception**, not a finished hardware integration. It is important to be explicit about what is real and what is a stand-in:

- **Current implementation**: depth comes from pre-computed `.npy` / 16-bit disparity files (DrivingStereo / KITTI-derived), loaded from disk by `StereoSimLoader`. There is no live camera or stereo matching happening at runtime — the "depth sensor" is a recorded dataset played back frame by frame.
- **Target hardware**: a ZED 2i stereo camera. The ZED SDK computes per-pixel depth directly on-device and additionally provides a **confidence map** for each depth pixel, which this simulation has no equivalent for (every `.npy` pixel is treated as equally trustworthy).
- **Parameter recalibration required for hardware**: the depth tolerance ratios, ground-plane model, and road/vegetation segmentation thresholds in `path_planning.py` are tuned against KITTI-scale depth noise and geometry. Moving to ZED 2i will require re-deriving `camera_height_m`, `fx`, `fy`, `cx`, and `cy` in `app/core/config.py` (`CAMERA_PROFILES["ZED_2I"]`) using the unit's factory calibration, rather than reusing the KITTI values.
- **Known limitation**: road vs. sidewalk/vegetation discrimination degrades when the depth source is estimated (monocular) rather than true stereo, since flat, coplanar surfaces (asphalt, sidewalk, grass) can be geometrically indistinguishable. ZED 2i's native stereo depth plus its per-pixel confidence map is expected to reduce this ambiguity by letting low-confidence, noisy readings near object/surface boundaries be filtered out before they reach the occupancy grid.

---

## Path Planning Algorithm

### C-Space Inflation (Minkowski Sum)

Since A* treats the vehicle as a 1-pixel point, obstacles must be inflated by the vehicle's physical radius + safety margin:

- Inflation radius: `0.6m` → `int(0.6 × 8.0) = 5px` → kernel size `11×11`
- Applied via `cv2.dilate()` after obstacle mapping
- Hood zone cleared AFTER inflation to prevent bleed-back

### Emergency Stop

Checks `grid[BEV_HEIGHT-80 : BEV_HEIGHT-40, ego_left:ego_right]` — 40 rows (~5m) of the ego corridor just above the hood zone. If ≥70% of cells are blocked (> 200), triggers immediate stop.

### Adaptive Cruise Control (ACC)

When obstacle is > 20m ahead: goal placed between ego and obstacle at `closest_obs_y + 80px` (10m behind obstacle). Keeps safe following distance.

### Overtaking

When obstacle is ≤ 20m ahead:
1. Check adjacent lanes at overtake depth (±10px window, 40px strip width)
2. If right lane free: goal at `(center_x + 44, overtake_y)` — 5.5m lateral offset
3. If left lane free: goal at `(center_x - 44, overtake_y)`
4. Overtake clearance: `closest_obs_y - 40 - int(2.5 × 8)` — past the full inflated obstacle

### A* Cost Function

```
f(n) = g(n) + h(n)

g(n) = parent_g + move_cost + proximity_penalty + turn_penalty

move_cost       = 1.0 (cardinal) or 1.4142 (diagonal)
proximity_penalty = (grid[y,x] / 255)² × 15.0   # exponential near obstacles
turn_penalty    = 0.5 if direction changed        # smooth curves
h(n)            = Euclidean distance to goal
```

---

## BEV & OGM Visualization

### BEV Canvas (bev_canvas)
- Detection dots positioned via pinhole projection (same formula as occupancy grid)
- Color-coded by `hazard_level`: red (danger), cyan (warning), green (safe)
- Uncertainty rings (red outline) for MC Dropout detections with variance ≥ 0.1
- A* waypoints overlaid via inverse perspective transform (`M_inv`)
- Ego vehicle marker at bottom-center

### OGM Canvas (ogm_canvas)
- Direct rendering of the `obstacle_grid` NumPy array as RGB image
- Obstacles: cyan (> 127) / red (> 200)
- A* waypoints drawn via `cv2.line` directly on the grid image (guaranteed alignment)
- Resized to canvas with `INTER_NEAREST` for crisp boundaries

---

## Model Management & Auto-Discovery

Models in `app/models/` are automatically discovered at startup:

| Filename Pattern | Type | Example |
|-----------------|------|---------|
| Contains `resnet`, `faster`, `rcnn` | FASTER-RCNN | `Resnet50.onnx` |
| Everything else | YOLO | `best.onnx`, `best_triplehead.onnx` |
| Contains `.dml_optimized` | DirectML-optimized | `best.dml_optimized.onnx` |

**YOLO labels:** Auto-extracted from ONNX metadata (`names` field) or read from `labels.txt` in the same directory.

**Model switching:** Use the dropdown in the control panel. Models are loaded lazily on first inference and cached in `ONNX_SESSIONS`.

---

## GPU Acceleration (DirectML)

The application uses **ONNX Runtime DirectML** for GPU inference on Windows (AMD/Intel/NVIDIA via DirectX 12).

### Provider Priority
1. `DmlExecutionProvider` (GPU via DirectML)
2. `CPUExecutionProvider` (fallback)

### Optimized Models
When a model is first loaded on GPU, ONNX Runtime applies graph optimizations. The optimized model is cached as `*.dml_optimized.onnx` and reused on subsequent loads.

### Verifying GPU Usage
Check the console output at startup:
```
[ONNXRuntime] Available providers: ['DmlExecutionProvider', 'CPUExecutionProvider']
ONNX MODEL 'BEST' NOT IN CACHE. Loading from: app/models/best.onnx
ONNX session for 'BEST' loaded. Provider: ['DmlExecutionProvider']
[WARMUP] Running 3 warmup inferences for 'BEST' (DirectML shader compile)...
```

If you see `CPUExecutionProvider` only, reinstall:
```powershell
pip uninstall onnxruntime onnxruntime-directml
pip install onnxruntime-directml
```

---

## Benchmarking

When processing a **Video File**, a benchmark report is shown at completion:

```
===== BENCHMARK RESULTS =====
Model Used: BEST
Input Video FPS: 30.00
Total Time: 45.23 s
Total Frames Read: 1357
Frames Processed (AI): 678
Frames Skipped (Sync): 679

--- FPS STATS ---
Average FPS: 15.2
Max FPS: 18.4
Min FPS: 12.1

--- LATENCY STATS ---
Average Latency: 66 ms
Max Latency: 89 ms
Min Latency: 54 ms
```

**Frame skipping:** If inference is slower than the source video FPS, unread frames are dropped to keep preview timing aligned to the original video clock.

---

## Troubleshooting

### `AttributeError: module 'onnxruntime' has no attribute 'InferenceSession'`
**Cause:** CPU `onnxruntime` is overriding `onnxruntime-directml`.
**Fix:**
```powershell
pip uninstall onnxruntime onnxruntime-directml
pip install onnxruntime-directml
```

### "No Models Found" on startup
**Cause:** No `.onnx` files in `app/models/`.
**Fix:** Place ONNX model files in `app/models/` and restart.

### BEV/OGM shows "NO PATH" with empty road
**Cause:** Obstacles are over-inflated or focal length is wrong.
**Check:**
- `STEREO_FX` matches your dataset (721.53 for DrivingStereo)
- `_INFLATION_RADIUS_M` is reasonable (0.6m default)
- `_EGO_LENGTH_M` obstacle box isn't too long (2.5m default)

### Depth values look wrong (all 0 or all 40)
**Cause:** Disparity file isn't 16-bit or baseline/fx are wrong.
**Check:** Disparity PNGs should be 16-bit (`cv2.IMREAD_UNCHANGED`). Baseline=0.54m, fx=721.53 for DrivingStereo.

### DirectML warmup fails
**Cause:** GPU driver doesn't support required DirectX 12 features.
**Fallback:** The app automatically retries with CPU. Check GPU driver updates.

### Stereo Dataset Sim mode: "No matching pairs"
**Cause:** Folder structure doesn't match `left_images/` + `disparity/` convention.
**Fix:** Ensure both subdirectories exist with matching `.png` filenames.

### Path planning disabled in Video/Webcam mode
**Cause:** Path planning requires stereo depth. Non-stereo modes pass `None` for `external_depth`.
**Solution:** Use **Stereo Dataset Sim** mode for full path planning functionality.

---

## License

Academic project — Universitas Kuliah. Not for commercial use without permission.

---

## Credits

- **YOLOv11** architecture with custom SimAM + ASPP + FPN + PANet modules
- **DrivingStereo** dataset for stereo camera calibration and testing
- **ONNX Runtime DirectML** for cross-vendor GPU acceleration on Windows
