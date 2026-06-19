# Technical Implementation Plan
## Autonomous Vehicle Perception System: KITTI → ROS 2 Humble → YOLO BEV
**Referensi Codebase:** `DavnFs/yolo_inference` | **Target Platform:** NVIDIA Jetson (JetPack 6, aarch64) | **Python:** 3.10+

---

> **Catatan Kritis Arsitektur (Baca Dulu)**
>
> Dari analisis `gui_main.py`, sistem kamu sudah punya `StereoSimLoader` yang membaca `left_images/` + `disparity/` secara static dari disk, `BEV canvas` yang sudah render via `cv2.circle`, dan `OGM canvas` berbasis `obstacle_grid`. Semua Fase di bawah ini adalah **upgrade** dari arsitektur yang sudah ada, bukan penggantian total. Goal akhirnya: ganti `StereoSimLoader._frame_list` dengan ROS 2 topic subscriber, dan upgrade `_flush_bev_update()` dengan proyeksi 3D yang benar secara matematis.

---

## Fase 1: Akuisisi Data KITTI Dataset

### 1.1 Dataset yang Dibutuhkan

Untuk testing sistem persepsi (bukan training), gunakan **KITTI Raw Data** bukan KITTI-360. Alasannya:

- KITTI Raw Data sudah menyediakan dense disparity via `data_scene_flow` (lebih ringkas dari KITTI-360 yang puluhan GB).
- Struktur folder KITTI Raw Data langsung kompatibel dengan `StereoSimLoader` existing kamu (tinggal remap direktori).
- KITTI-360 hanya relevan jika kamu butuh 360° coverage — untuk persepsi forward-facing YOLO, ini overkill dan buang storage.

**⚠️ Arsitektur Warning:** Jangan download KITTI Training Set (12 GB) untuk testing. Gunakan hanya `2011_09_26_drive_0001` sampai `0005` — total ~800 MB, sudah lebih dari cukup untuk validasi pipeline.

### 1.2 Autentikasi KITTI via CLI

KITTI menggunakan sistem autentikasi cookie-based (bukan token/API key). Cara handle di CLI:

**Step 1 — Daftar akun di https://www.cvlibs.net/datasets/kitti/user_register.php**

**Step 2 — Dapatkan session cookie:**
```bash
# Login dan simpan cookie ke file
curl -c kitti_cookies.txt \
     -d "email=YOUR_EMAIL&password=YOUR_PASSWORD" \
     -X POST "https://www.cvlibs.net/datasets/kitti/login.php" \
     -L --silent -o /dev/null

# Verifikasi cookie berhasil
cat kitti_cookies.txt | grep -v "^#" | grep "cvlibs"
```

**Step 3 — Download menggunakan cookie:**
```bash
wget --load-cookies kitti_cookies.txt \
     --no-check-certificate \
     -P /data/kitti/raw \
     "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0001/2011_09_26_drive_0001_sync.zip"
```

**Catatan Penting:** KITTI sejak 2023 mulai hosting file besar di AWS S3, yang aksesnya **tidak memerlukan login** — langsung bisa `wget` URL S3. URL login hanya diperlukan untuk file yang masih di server cvlibs.net.

### 1.3 Bash Script Download Otomatis (Subset Testing)

```bash
#!/usr/bin/env bash
# =============================================================
# kitti_download_subset.sh
# Download minimal subset KITTI untuk testing pipeline YOLO
# Target: ~800 MB total (5 drive sequences)
# Kompatibel: aarch64, wget >= 1.20
# =============================================================
set -euo pipefail

KITTI_BASE="https://s3.eu-central-1.amazonaws.com/avg-kitti"
OUTPUT_DIR="${1:-/data/kitti}"
DRIVES=(
    "2011_09_26_drive_0001_sync"
    "2011_09_26_drive_0002_sync"
    "2011_09_26_drive_0005_sync"
)

CALIB_URL="${KITTI_BASE}/raw_data/2011_09_26_calib.zip"
SCENE_FLOW_URL="${KITTI_BASE}/data_scene_flow.zip"  # Disparity GT (194 MB)

mkdir -p "${OUTPUT_DIR}/raw" "${OUTPUT_DIR}/disparity_gt" "${OUTPUT_DIR}/calib"

echo "[1/3] Downloading calibration files..."
wget -q --show-progress -P "${OUTPUT_DIR}/calib" "${CALIB_URL}"
unzip -q "${OUTPUT_DIR}/calib/2011_09_26_calib.zip" -d "${OUTPUT_DIR}/calib/"

echo "[2/3] Downloading RGB sequences..."
for DRIVE in "${DRIVES[@]}"; do
    DATE="${DRIVE:0:10}"
    URL="${KITTI_BASE}/raw_data/${DATE}_drive_${DRIVE:14:4}/${DRIVE}.zip"
    echo "  -> ${DRIVE}"
    wget -q --show-progress -c -P "${OUTPUT_DIR}/raw" "${URL}"
    unzip -q "${OUTPUT_DIR}/raw/${DRIVE}.zip" -d "${OUTPUT_DIR}/raw/"
    rm "${OUTPUT_DIR}/raw/${DRIVE}.zip"
done

echo "[3/3] Downloading Scene Flow disparity (dense depth GT)..."
wget -q --show-progress -c -P "${OUTPUT_DIR}/disparity_gt" "${SCENE_FLOW_URL}"
unzip -q "${OUTPUT_DIR}/disparity_gt/data_scene_flow.zip" -d "${OUTPUT_DIR}/disparity_gt/"

echo "Done. Struktur tersedia di: ${OUTPUT_DIR}"
```

**Jalankan:**
```bash
chmod +x kitti_download_subset.sh
./kitti_download_subset.sh /data/kitti
```

### 1.4 Struktur Folder Rekomendasi (Post-Extract)

Struktur ini dirancang agar **langsung compatible** dengan `StereoSimLoader` existing dan pipeline ROS 2 di Fase 2:

```
/data/kitti/
├── calib/
│   └── 2011_09_26/
│       ├── calib_cam_to_cam.txt      ← fx, fy, cx, cy ada di sini (P_rect_02)
│       ├── calib_imu_to_velo.txt
│       └── calib_velo_to_cam.txt
│
├── sequences/                        ← renamed dari raw/ untuk clarity
│   ├── 0001/
│   │   ├── left_images/              ← rename dari image_02/data/ (langsung cocok StereoSimLoader)
│   │   │   ├── 0000000000.png
│   │   │   ├── 0000000001.png
│   │   │   └── ...
│   │   ├── disparity/                ← dari data_scene_flow/training/disp_noc_0/
│   │   │   ├── 0000000000.png        ← 16-bit PNG, format KITTI = disp * 256
│   │   │   └── ...
│   │   └── timestamps.txt            ← dari image_02/timestamps.txt
│   ├── 0002/
│   └── 0005/
│
└── rosbags/                          ← output Fase 2
    ├── kitti_0001.mcap
    ├── kitti_0002.mcap
    └── kitti_0005.mcap
```

**Script remap folder (post-extract):**
```bash
#!/usr/bin/env bash
# Remap KITTI raw structure → StereoSimLoader-compatible structure
KITTI_RAW="/data/kitti/raw/2011_09_26"
TARGET="/data/kitti/sequences"

for DRIVE_DIR in "${KITTI_RAW}"/2011_09_26_drive_*/; do
    DRIVE_NUM=$(echo "${DRIVE_DIR}" | grep -oP 'drive_\K\d+')
    SEQ_DIR="${TARGET}/${DRIVE_NUM}"
    mkdir -p "${SEQ_DIR}/left_images" "${SEQ_DIR}/disparity"

    # Symlink images (hemat storage, tidak copy)
    ln -sfn "${DRIVE_DIR}image_02/data" "${SEQ_DIR}/left_images"
    cp "${DRIVE_DIR}image_02/timestamps.txt" "${SEQ_DIR}/timestamps.txt"

    echo "Mapped drive ${DRIVE_NUM} → ${SEQ_DIR}"
done

# Disparity dari scene flow harus dicopy manual per sequence
# (format penamaan file berbeda, perlu mapping index)
echo "NOTE: Copy disparity files manually dari data_scene_flow/training/disp_noc_0/"
```

---

## Fase 2: Data Pipeline — KITTI ke ROS 2 Bag (.mcap)

### 2.1 Dependensi

```bash
# Install di Jetson (JetPack 6 = Ubuntu 22.04 + ROS 2 Humble)
sudo apt install -y \
    python3-rclpy \
    ros-humble-rosbag2-py \
    ros-humble-cv-bridge \
    ros-humble-sensor-msgs \
    ros-humble-image-transport

# MCAP writer support
pip install mcap-ros2-support --break-system-packages
```

### 2.2 Script: KITTI Folder → ROS 2 .mcap Bag

```python
#!/usr/bin/env python3
"""
kitti_to_rosbag.py
Konversi KITTI sequence folder ke ROS 2 .mcap bag.
Kompatibel: Python 3.10, aarch64, ROS 2 Humble, JetPack 6.

Peringatan Kritis:
- Jangan jalankan ini di Jetson saat inference berjalan. Proses serialisasi
  rosbag2 sangat I/O-bound dan akan bersaing dengan NVME bandwidth.
- Gunakan format MCAP, bukan DB3. DB3 (SQLite) punya write amplification
  yang buruk untuk data besar (image sequences). MCAP jauh lebih efisien.
"""

import os
import sys
import struct
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.serialization import serialize_message
import rosbag2_py
from sensor_msgs.msg import Image, CameraInfo
from builtin_interfaces.msg import Time


# ──────────────────────────────────────────────
# 1. Kalibrasi Parser
# ──────────────────────────────────────────────

def parse_kitti_calib(calib_path: str) -> dict:
    """
    Parse calib_cam_to_cam.txt KITTI.
    Return dict dengan fx, fy, cx, cy, baseline, P_rect_02.
    
    P_rect_02 adalah 3x4 projection matrix untuk kamera kiri setelah rektifikasi.
    Baris format: P_rect_02: fx 0 cx tx 0 fy cy ty 0 0 1 0
    """
    calib = {}
    with open(calib_path, 'r') as f:
        for line in f:
            if line.startswith('P_rect_02:'):
                vals = list(map(float, line.split()[1:]))
                P = np.array(vals).reshape(3, 4)
                calib['fx'] = P[0, 0]
                calib['fy'] = P[1, 1]
                calib['cx'] = P[0, 2]
                calib['cy'] = P[1, 2]
                calib['P_rect_02'] = P
            elif line.startswith('P_rect_03:'):
                # Kamera kanan — untuk hitung baseline
                vals = list(map(float, line.split()[1:]))
                P3 = np.array(vals).reshape(3, 4)
                # baseline = -T_x / fx (T_x adalah translasi stereo dalam P_rect_03)
                calib['baseline'] = abs(P3[0, 3]) / P3[0, 0]
    return calib


def build_camera_info_msg(calib: dict, img_shape: tuple, frame_id: str) -> CameraInfo:
    """Bangun sensor_msgs/CameraInfo dari dict kalibrasi KITTI."""
    msg = CameraInfo()
    msg.header.frame_id = frame_id
    h, w = img_shape[:2]
    msg.height = h
    msg.width = w
    msg.distortion_model = 'plumb_bob'
    msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]  # KITTI sudah rectified, distorsi = 0
    msg.k = [
        calib['fx'], 0.0,          calib['cx'],
        0.0,         calib['fy'],  calib['cy'],
        0.0,         0.0,          1.0
    ]
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]  # Identity (rectified)
    P = calib['P_rect_02'].flatten().tolist()
    msg.p = P
    return msg


# ──────────────────────────────────────────────
# 2. Timestamp Parser
# ──────────────────────────────────────────────

def parse_kitti_timestamps(ts_path: str) -> list:
    """
    Parse KITTI timestamps.txt → list of nanoseconds (int).
    Format baris: 2011-09-26 13:02:25.978919000
    """
    timestamps_ns = []
    ref_dt = None
    with open(ts_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            dt = datetime.strptime(line, '%Y-%m-%d %H:%M:%S.%f')
            if ref_dt is None:
                ref_dt = dt
            delta_ns = int((dt - ref_dt).total_seconds() * 1e9)
            timestamps_ns.append(delta_ns)
    return timestamps_ns


# ──────────────────────────────────────────────
# 3. Image → sensor_msgs/Image
# ──────────────────────────────────────────────

def bgr_to_ros_image(img_bgr: np.ndarray, stamp: Time, frame_id: str) -> Image:
    """
    Konversi BGR image ke sensor_msgs/Image (encoding: bgr8).
    TIDAK menggunakan cv_bridge untuk menghindari dependensi ROS runtime
    di skrip converter yang jalan standalone.
    """
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = img_bgr.shape[0]
    msg.width = img_bgr.shape[1]
    msg.encoding = 'bgr8'
    msg.is_bigendian = False
    msg.step = img_bgr.shape[1] * 3
    msg.data = img_bgr.tobytes()
    return msg


def disparity_to_ros_depth(disp_raw: np.ndarray, fx: float, baseline: float,
                            stamp: Time, frame_id: str) -> Image:
    """
    Konversi 16-bit KITTI disparity PNG → depth_map dalam meter → sensor_msgs/Image (32FC1).

    KITTI 16-bit disparity convention: actual_disparity = pixel_value / 256.0
    Depth [m] = (fx * baseline) / disparity

    Encoding 32FC1 dipilih karena:
    - cv_bridge di sisi subscriber langsung bisa convert ke float32 numpy
    - Tidak ada information loss (vs 16UC1 yang perlu scaling)
    - Kompatibel dengan semua depth-aware tool di ROS 2 ecosystem
    """
    disp_f = disp_raw.astype(np.float32) / 256.0
    # Vectorized — tidak ada loop piksel
    valid_mask = disp_f > 0.1
    depth_m = np.where(valid_mask, (fx * baseline) / disp_f, 0.0).astype(np.float32)
    depth_m = np.clip(depth_m, 0.0, 80.0)  # KITTI max range ~50m, clip 80m aman

    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = depth_m.shape[0]
    msg.width = depth_m.shape[1]
    msg.encoding = '32FC1'
    msg.is_bigendian = False
    msg.step = depth_m.shape[1] * 4  # float32 = 4 bytes
    msg.data = depth_m.tobytes()
    return msg


# ──────────────────────────────────────────────
# 4. Main Converter
# ──────────────────────────────────────────────

def ns_to_ros_stamp(ns: int) -> Time:
    t = Time()
    t.sec = ns // 1_000_000_000
    t.nanosec = ns % 1_000_000_000
    return t


def convert_sequence(seq_dir: str, calib_path: str, output_bag: str):
    """
    Konversi satu KITTI sequence folder ke .mcap bag.

    Topics yang ditulis:
    - /kitti/camera/left/image_raw    (sensor_msgs/Image, bgr8)
    - /kitti/camera/left/depth        (sensor_msgs/Image, 32FC1, meter)
    - /kitti/camera/left/camera_info  (sensor_msgs/CameraInfo)
    """
    seq_path = Path(seq_dir)
    left_dir = seq_path / 'left_images'
    disp_dir = seq_path / 'disparity'
    ts_file  = seq_path / 'timestamps.txt'

    # Validasi
    for p in [left_dir, disp_dir, ts_file]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    calib = parse_kitti_calib(calib_path)
    timestamps_ns = parse_kitti_timestamps(str(ts_file))

    frame_files = sorted(f for f in left_dir.iterdir() if f.suffix == '.png')
    n_frames = min(len(frame_files), len(timestamps_ns))

    print(f"Converting {n_frames} frames from {seq_dir}")
    print(f"  fx={calib['fx']:.2f}, fy={calib['fy']:.2f}, "
          f"cx={calib['cx']:.2f}, cy={calib['cy']:.2f}, "
          f"baseline={calib.get('baseline', 0.54):.4f}m")

    # Setup rosbag2 writer
    storage_opts = rosbag2_py.StorageOptions(uri=output_bag, storage_id='mcap')
    converter_opts = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    writer = rosbag2_py.SequentialWriter()
    writer.open(storage_opts, converter_opts)

    # Register topics
    TOPICS = {
        '/kitti/camera/left/image_raw':   'sensor_msgs/msg/Image',
        '/kitti/camera/left/depth':       'sensor_msgs/msg/Image',
        '/kitti/camera/left/camera_info': 'sensor_msgs/msg/CameraInfo',
    }
    for topic, type_str in TOPICS.items():
        writer.create_topic(rosbag2_py.TopicMetadata(
            name=topic,
            type=type_str,
            serialization_format='cdr'
        ))

    # Baca satu frame dulu untuk dapatkan image shape
    sample_img = cv2.imread(str(frame_files[0]))
    cam_info_msg = build_camera_info_msg(calib, sample_img.shape,
                                          frame_id='camera_left')

    baseline = calib.get('baseline', 0.54)
    fx = calib['fx']

    for i, frame_path in enumerate(frame_files[:n_frames]):
        if i % 50 == 0:
            print(f"  Frame {i}/{n_frames}...")

        ts_ns = timestamps_ns[i]
        stamp = ns_to_ros_stamp(ts_ns)

        # Load RGB
        img_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"  WARN: Cannot read {frame_path}, skipping.")
            continue

        # Load disparity
        disp_path = disp_dir / frame_path.name
        if not disp_path.exists():
            print(f"  WARN: No disparity for {frame_path.name}, skipping.")
            continue
        disp_raw = cv2.imread(str(disp_path), cv2.IMREAD_UNCHANGED)
        if disp_raw is None:
            continue

        # Build messages
        img_msg   = bgr_to_ros_image(img_bgr, stamp, 'camera_left')
        depth_msg = disparity_to_ros_depth(disp_raw, fx, baseline, stamp, 'camera_left')

        # Update CameraInfo stamp
        cam_info_msg.header.stamp = stamp

        # Write ke bag
        writer.write('/kitti/camera/left/image_raw',
                     serialize_message(img_msg), ts_ns)
        writer.write('/kitti/camera/left/depth',
                     serialize_message(depth_msg), ts_ns)
        writer.write('/kitti/camera/left/camera_info',
                     serialize_message(cam_info_msg), ts_ns)

    del writer
    print(f"Bag saved: {output_bag}")


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python3 kitti_to_rosbag.py <seq_dir> <calib_path> <output.mcap>")
        sys.exit(1)

    convert_sequence(
        seq_dir=sys.argv[1],
        calib_path=sys.argv[2],
        output_bag=sys.argv[3]
    )
```

**Jalankan:**
```bash
source /opt/ros/humble/setup.bash

python3 kitti_to_rosbag.py \
    /data/kitti/sequences/0001 \
    /data/kitti/calib/2011_09_26/calib_cam_to_cam.txt \
    /data/kitti/rosbags/kitti_0001.mcap
```

**Verifikasi bag:**
```bash
ros2 bag info /data/kitti/rosbags/kitti_0001.mcap
```

---

## Fase 3: Refactoring Codebase — Static Folder → ROS 2 Subscriber

### 3.1 Strategi Refactoring

Dari analisis `gui_main.py`, komponen yang harus direfactor:

| Komponen Existing | Status | Aksi |
|---|---|---|
| `StereoSimLoader` | Baca file statis | **Ganti** dengan ROS subscriber node |
| `_processing_loop()` — branch `is_stereo_sim` | Block thread, polling | **Ganti** dengan async callback |
| `process_and_display_frame()` | Sudah bagus | **Pertahankan**, ganti input source |
| `_flush_bev_update()` | BEV via cv2.circle | **Upgrade** di Fase 4 |
| `gui_main.py` Tkinter loop | Main thread | **Pertahankan**, tambah ROS spin thread |

**Pola refactoring yang direkomendasikan:** Jalankan ROS 2 node di thread terpisah (`daemon=True`), komunikasi ke GUI via `queue.Queue` yang sudah thread-safe. Jangan gunakan `rclpy.spin()` di main thread — itu akan memblokir Tkinter event loop.

### 3.2 ROS 2 Subscriber Node dengan Message Filters

```python
#!/usr/bin/env python3
"""
ros2_perception_node.py
ROS 2 subscriber node untuk yolo_inference.
Replace StereoSimLoader dengan synchronized subscriber.

Arsitektur:
- Node berjalan di background thread
- Frame + depth dikirim ke GUI via thread-safe Queue
- GUI tetap di main thread (Tkinter requirement)

Peringatan:
- JANGAN gunakan rclpy.spin() di main thread jika Tkinter aktif.
- message_filters.ApproximateTimeSynchronizer: slop 0.05s (50ms).
  Nilai ini cukup longgar untuk data offline bag. Untuk real sensor,
  turunkan ke 0.02s atau gunakan ExactTimeSynchronizer.
- cv_bridge di Jetson: INSTALL via apt, bukan pip.
  pip install cv_bridge akan build dari source dan kemungkinan gagal
  di aarch64 karena missing ROS build dependencies.
"""

import threading
import queue
from typing import Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import Image, CameraInfo
import message_filters
from cv_bridge import CvBridge


# ──────────────────────────────────────────────
# Konfigurasi Topic (sesuaikan jika beda)
# ──────────────────────────────────────────────
TOPIC_RGB   = '/kitti/camera/left/image_raw'
TOPIC_DEPTH = '/kitti/camera/left/depth'
TOPIC_INFO  = '/kitti/camera/left/camera_info'


class PerceptionSubscriberNode(Node):
    """
    Node yang subscribe RGB + Depth secara tersinkronisasi.
    
    Desain keputusan:
    - Queue maxsize=2: jika GUI processing lambat, frame baru
      akan menimpa yang lama. Ini BENAR untuk real-time — lebih
      baik drop frame daripada terus accumulate lag.
    - CameraInfo: subscribe sekali, cache intrinsik.
    """

    def __init__(self, frame_queue: queue.Queue):
        super().__init__('yolo_perception_subscriber')
        self._bridge = CvBridge()
        self._frame_queue = frame_queue
        self._camera_info: Optional[CameraInfo] = None
        self._calib_cache: Optional[dict] = None

        # Subscribe CameraInfo (latched, cukup sekali)
        self._info_sub = self.create_subscription(
            CameraInfo,
            TOPIC_INFO,
            self._camera_info_callback,
            qos_profile=rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value
        )

        # Synchronized RGB + Depth
        self._rgb_sub = message_filters.Subscriber(
            self, Image, TOPIC_RGB,
            qos_profile=rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value
        )
        self._depth_sub = message_filters.Subscriber(
            self, Image, TOPIC_DEPTH,
            qos_profile=rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value
        )

        # ApproximateTimeSynchronizer — toleransi 50ms untuk bag offline
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self._rgb_sub, self._depth_sub],
            queue_size=10,
            slop=0.05
        )
        self._sync.registerCallback(self._synced_callback)
        self.get_logger().info('PerceptionSubscriberNode ready.')

    def _camera_info_callback(self, msg: CameraInfo):
        """Cache camera intrinsics. Di-call sekali di awal bag playback."""
        if self._calib_cache is not None:
            return  # Sudah cached, skip

        K = msg.k  # flat list 9 elemen
        self._calib_cache = {
            'fx': K[0],
            'fy': K[4],
            'cx': K[2],
            'cy': K[5],
            'width': msg.width,
            'height': msg.height,
        }
        self.get_logger().info(
            f'CameraInfo cached: fx={K[0]:.2f}, fy={K[4]:.2f}, '
            f'cx={K[2]:.2f}, cy={K[5]:.2f}'
        )

    def _synced_callback(self, rgb_msg: Image, depth_msg: Image):
        """
        Callback terpanggil ketika pasangan RGB+Depth tersinkronisasi tersedia.
        
        Konversi dilakukan di sini (background thread) agar main thread (GUI)
        tidak dibebani deserialisasi. Queue.put_nowait() agar tidak blocking.
        """
        try:
            # RGB: bgr8 → numpy (H, W, 3) uint8
            frame_bgr = self._bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')

            # Depth: 32FC1 → numpy (H, W) float32 [meter]
            # KRITIS: gunakan passthrough, jangan konversi encoding
            # '32FC1' sudah float32 dalam meter dari Fase 2 converter kita
            depth_m = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

            # Validasi shape
            if depth_m.dtype != np.float32:
                depth_m = depth_m.astype(np.float32)

            # Kirim ke GUI queue — non-blocking (drop jika penuh)
            payload = {
                'frame_bgr': frame_bgr,
                'depth_m': depth_m,
                'calib': self._calib_cache,
                'timestamp': rgb_msg.header.stamp,
            }
            try:
                self._frame_queue.put_nowait(payload)
            except queue.Full:
                # GUI lebih lambat dari bag playback — drop frame, log sekali
                pass

        except Exception as e:
            self.get_logger().error(f'Callback error: {e}')

    def get_calib(self) -> Optional[dict]:
        return self._calib_cache


# ──────────────────────────────────────────────
# Runner — Jalankan di Thread Terpisah
# ──────────────────────────────────────────────

class ROSBagRunner:
    """
    Wrapper untuk menjalankan ROS 2 node di background thread.
    Integrasikan ke DashboardGUI sebagai pengganti StereoSimLoader.
    """

    def __init__(self):
        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self._node: Optional[PerceptionSubscriberNode] = None
        self._executor: Optional[SingleThreadedExecutor] = None
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if not rclpy.ok():
            rclpy.init()
        self._node = PerceptionSubscriberNode(self._frame_queue)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(
            target=self._spin, daemon=True, name='ros2_spin'
        )
        self._thread.start()

    def _spin(self):
        try:
            self._executor.spin()
        except Exception as e:
            print(f'[ROSBagRunner] spin error: {e}')

    def get_frame(self, timeout: float = 0.1) -> Optional[dict]:
        """Blocking get dengan timeout. Return None jika kosong."""
        try:
            return self._frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        if self._executor:
            self._executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()


# ──────────────────────────────────────────────
# Cara Integrasi ke gui_main.py
# ──────────────────────────────────────────────
"""
Di DashboardGUI.__init__(), tambahkan:
    self._ros_runner = None

Di start_processing(), untuk mode "ROS 2 Bag":
    self._ros_runner = ROSBagRunner()
    self._ros_runner.start()

Di _processing_loop(), ganti branch stereo_sim:
    while not self.stop_processing_flag.is_set():
        payload = self._ros_runner.get_frame(timeout=0.1)
        if payload is None:
            continue
        frame_bgr = payload['frame_bgr']
        depth_m = payload['depth_m']
        self.process_and_display_frame(frame_bgr, external_depth=depth_m)

Di stop_processing():
    if self._ros_runner:
        self._ros_runner.stop()
        self._ros_runner = None
"""
```

### 3.3 Cara Konversi Depth ke Meter untuk Path Planning

Depth yang keluar dari `cv_bridge` dengan encoding `passthrough` sudah float32 dalam meter (karena kita write `32FC1` di Fase 2). Tidak perlu konversi tambahan — langsung feed ke `external_depth` parameter yang sudah ada di `process_and_display_frame()`:

```python
# Di _processing_loop() setelah get_frame()
payload = self._ros_runner.get_frame()
if payload:
    frame_bgr = payload['frame_bgr']   # shape: (H, W, 3), uint8
    depth_m   = payload['depth_m']     # shape: (H, W), float32, satuan METER

    # Validasi sebelum feed ke path_planning
    assert depth_m.dtype == np.float32, "Depth harus float32"
    assert depth_m.ndim == 2,           "Depth harus 2D array"
    
    # Langsung bisa dipakai sebagai matriks jarak absolut
    # depth_m[v, u] = jarak piksel (u,v) ke kamera dalam METER
    self.process_and_display_frame(frame_bgr, external_depth=depth_m)
```

---

## Fase 4: Bird's Eye View (BEV) — Proyeksi 3D Matematis

### 4.1 Derivasi Linear Algebra: Pixel → World 3D

Sistem kamera pinhole menggunakan model proyeksi perspektif. Untuk merekonstruksi koordinat 3D dari piksel 2D + depth scalar:

**Model Proyeksi (Forward — 3D ke 2D):**

```
u = fx * (X/Z) + cx
v = fy * (Y/Z) + cy
```

**Invers (Back-projection — 2D + Z ke 3D):**

```
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
Z = Z  (langsung dari depth map)
```

Dalam notasi matriks, untuk kumpulan N titik, gunakan **homogeneous coordinates**:

```
Untuk setiap bounding box detection dengan centroid (u_c, v_c) dan depth Z:

[X]   [1/fx    0    -cx/fx] [u_c]
[Y] = [  0   1/fy  -cy/fy] [v_c] * Z
[Z]   [  0     0      1  ] [ 1 ]

Atau lebih simpel (scalar form):
X = (u_c - cx) * Z / fx
Y = (v_c - cy) * Z / fy
```

Untuk BEV (Top-Down View), kita **buang sumbu Y** (vertikal) dan plot di bidang XZ:
- `X` = posisi lateral (kiri-kanan dari kamera)
- `Z` = jarak ke depan (depth)

**Pengambilan depth untuk bounding box:** Ambil **median** dari region `depth_map[y1:y2, x1:x2]` (bukan mean). Median robust terhadap noise border dan reflektif surface yang sering terjadi di hood kendaraan.

### 4.2 Implementasi Vectorized NumPy

```python
"""
bev_projection.py
Proyeksi BEV yang benar secara matematis, fully vectorized.
Ini adalah upgrade dari _flush_bev_update() di gui_main.py existing.

KRITIK KERAS terhadap implementasi existing di gui_main.py:
- Baris ini di _flush_bev_update() menggunakan cx_img = orig_w // 2
  sebagai aproksimasi cx. Ini SALAH untuk KITTI karena cx KITTI tidak
  tepat di tengah (umumnya 609.5 bukan 621). Error ini akumulatif
  dan akan membuat semua objek tampak slightly off-center di BEV.
- Perbaikan: gunakan nilai cx dari CameraInfo yang sudah kita parse.
"""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class Detection3D:
    label: str
    x_m: float       # lateral position [meter]
    y_m: float       # vertical position [meter] (biasanya diabaikan di BEV)
    z_m: float       # depth / forward distance [meter]
    confidence: float
    hazard_level: str
    bbox_2d: Tuple[int, int, int, int]  # x1, y1, x2, y2 dalam piksel


def backproject_detections(
    detections: List[dict],
    depth_map: np.ndarray,   # (H, W) float32, meter
    intrinsics: Intrinsics
) -> List[Detection3D]:
    """
    Konversi list detections (format existing yolo_inference) ke koordinat 3D.
    
    VECTORIZED: semua operasi dilakukan dalam satu pass NumPy per batch.
    Tidak ada for-loop per piksel — hanya for-loop per detection (jumlahnya kecil, O(10)).
    
    Kenapa median depth, bukan point depth?
    - Ambil titik tengah bbox saja berisiko: titik itu bisa jatuh di kaca
      mobil (depth jauh) atau di area oklusi (depth noise).
    - Median dari seluruh region bbox lebih robust, komputasi O(n) dengan
      n = area bbox piksel, masih sangat cepat di NumPy.
    """
    results = []
    H, W = depth_map.shape

    # Batch: kumpulkan semua bbox centroid dulu untuk vectorized ops
    u_centers = []
    v_centers = []
    depths = []
    valid_dets = []

    for det in detections:
        bbox = det.get('bounding_box', [0, 0, 0, 0])  # format: [x1, y1, x2, y2]
        x1, y1, x2, y2 = bbox

        # Clamp ke bounds
        x1c = max(0, int(x1)); x2c = min(W, int(x2))
        y1c = max(0, int(y1)); y2c = min(H, int(y2))

        if x2c <= x1c or y2c <= y1c:
            continue

        # Ambil region depth dan compute median (vectorized internal NumPy)
        region = depth_map[y1c:y2c, x1c:x2c]
        valid_pixels = region[region > 0.1]

        if valid_pixels.size < 10:  # Terlalu sedikit piksel valid, skip
            continue

        z = float(np.median(valid_pixels))

        # Centroid piksel bounding box
        u_c = (x1 + x2) * 0.5
        v_c = (y1 + y2) * 0.5

        u_centers.append(u_c)
        v_centers.append(v_c)
        depths.append(z)
        valid_dets.append(det)

    if not valid_dets:
        return []

    # ── VECTORIZED BACKPROJECTION ──────────────────────────────────────
    u_arr = np.array(u_centers, dtype=np.float32)
    v_arr = np.array(v_centers, dtype=np.float32)
    z_arr = np.array(depths,    dtype=np.float32)

    # X = (u - cx) * Z / fx  [lateral, kiri negatif, kanan positif]
    # Y = (v - cy) * Z / fy  [vertikal, atas negatif]
    # Z = depth               [forward, positif]
    X_arr = (u_arr - intrinsics.cx) * z_arr / intrinsics.fx
    Y_arr = (v_arr - intrinsics.cy) * z_arr / intrinsics.fy

    for i, det in enumerate(valid_dets):
        bbox = det.get('bounding_box', [0, 0, 0, 0])
        results.append(Detection3D(
            label=det.get('label', 'unknown'),
            x_m=float(X_arr[i]),
            y_m=float(Y_arr[i]),
            z_m=float(z_arr[i]),
            confidence=det.get('confidence', 0.0),
            hazard_level=det.get('hazard_level', 'safe'),
            bbox_2d=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        ))

    return results


# ──────────────────────────────────────────────
# BEV Grid Renderer (OpenCV)
# ──────────────────────────────────────────────

HAZARD_COLORS = {
    'critical': (0,   0,   255),   # Red
    'warning':  (0,   165, 255),   # Orange
    'caution':  (0,   255, 255),   # Yellow
    'safe':     (0,   200, 0),     # Green
}


def render_bev_opencv(
    detections_3d: List[Detection3D],
    canvas_wh: Tuple[int, int] = (400, 600),
    max_range_m: float = 40.0,
    lateral_range_m: float = 15.0,
    path_waypoints: Optional[List[Tuple]] = None
) -> np.ndarray:
    """
    Render BEV grid ke OpenCV image.

    Koordinat grid:
    - Ego vehicle di BAWAH-TENGAH canvas
    - Z (depth) = arah ke ATAS canvas
    - X (lateral) = kiri-kanan

    UPGRADE dari _flush_bev_update() existing:
    - Posisi X sekarang dihitung dari backprojection yang benar, bukan
      aproksimasi berbasis bbox_cx dan cx_img = orig_w // 2
    - Radius circle proporsional terhadap confidence
    - Grid lines sebagai visual reference
    """
    W, H = canvas_wh
    bev = np.zeros((H, W, 3), dtype=np.uint8)

    # Pixel per meter untuk setiap sumbu
    ppm_z = H / max_range_m       # Vertical (depth)
    ppm_x = W / (2 * lateral_range_m)  # Horizontal (lateral)

    # Origin ego vehicle (piksel) — bottom center
    ego_px = W // 2
    ego_py = H - 10

    # ── Grid lines ──────────────────────────────────────────────────
    # Distance rings setiap 10m
    for d_m in range(10, int(max_range_m) + 1, 10):
        y_grid = int(ego_py - d_m * ppm_z)
        if 0 <= y_grid < H:
            cv2.line(bev, (0, y_grid), (W, y_grid), (40, 40, 40), 1)
            cv2.putText(bev, f'{d_m}m', (5, y_grid - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 80), 1)

    # Center line (ego forward)
    cv2.line(bev, (ego_px, 0), (ego_px, H), (40, 40, 40), 1)

    # Ego vehicle marker
    cv2.rectangle(bev,
                  (ego_px - 8, ego_py - 15),
                  (ego_px + 8, ego_py),
                  (0, 200, 200), -1)

    # ── VECTORIZED: konversi semua 3D coords ke pixel sekaligus ──────
    if detections_3d:
        x_m_arr = np.array([d.x_m for d in detections_3d], dtype=np.float32)
        z_m_arr = np.array([d.z_m for d in detections_3d], dtype=np.float32)

        # Pixel coordinates
        px_arr = (ego_px + x_m_arr * ppm_x).astype(np.int32)
        py_arr = (ego_py - z_m_arr * ppm_z).astype(np.int32)

        # Filter yang dalam canvas bounds
        valid = (px_arr >= 0) & (px_arr < W) & (py_arr >= 0) & (py_arr < H)

        for i, det in enumerate(detections_3d):
            if not valid[i]:
                continue

            px, py = int(px_arr[i]), int(py_arr[i])
            color = HAZARD_COLORS.get(det.hazard_level, (128, 128, 128))
            radius = max(4, int(det.confidence * 10))

            cv2.circle(bev, (px, py), radius, color, -1)
            cv2.circle(bev, (px, py), radius + 2, color, 1)  # Outline

            # Label: class + distance
            label_str = f"{det.label[:4]} {det.z_m:.1f}m"
            cv2.putText(bev, label_str, (px + 6, py + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1)

    # ── Path waypoints ──────────────────────────────────────────────
    if path_waypoints and len(path_waypoints) > 1:
        # Konversi waypoints dari BEV grid space ke canvas space
        # (asumsi waypoints sudah dalam format (x_px_bev, y_px_bev))
        for i in range(len(path_waypoints) - 1):
            pt1 = (int(path_waypoints[i][0] * W / 400),
                   int(path_waypoints[i][1] * H / 600))
            pt2 = (int(path_waypoints[i+1][0] * W / 400),
                   int(path_waypoints[i+1][1] * H / 600))
            cv2.line(bev, pt1, pt2, (0, 200, 255), 2)

    return bev


# ──────────────────────────────────────────────
# RViz2 MarkerArray Publisher (Opsional)
# ──────────────────────────────────────────────

def detections_to_marker_array(
    detections_3d: List[Detection3D],
    node: 'rclpy.node.Node',
    frame_id: str = 'camera_left'
):
    """
    Publish deteksi ke RViz2 sebagai visualization_msgs/MarkerArray.
    Berguna untuk debug pipeline di development stage.
    
    Catatan untuk Jetson:
    - MarkerArray publish overhead kecil (CPU side), aman dipakai.
    - Jangan publish di setiap frame jika RViz2 tidak aktif — cek
      subscriber count dulu untuk hemat bandwidth.
    """
    from visualization_msgs.msg import MarkerArray, Marker
    from geometry_msgs.msg import Point
    from std_msgs.msg import ColorRGBA
    from builtin_interfaces.msg import Duration

    marker_arr = MarkerArray()
    now = node.get_clock().now().to_msg()

    # Clear semua marker lama dulu
    clear_marker = Marker()
    clear_marker.action = Marker.DELETEALL
    marker_arr.markers.append(clear_marker)

    COLORS_RGBA = {
        'critical': (1.0, 0.0, 0.0, 0.8),
        'warning':  (1.0, 0.6, 0.0, 0.8),
        'caution':  (1.0, 1.0, 0.0, 0.8),
        'safe':     (0.0, 0.8, 0.0, 0.8),
    }

    for i, det in enumerate(detections_3d):
        # Sphere marker di posisi 3D
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = now
        m.ns = 'detections'
        m.id = i
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = float(det.z_m)   # RViz: X = forward
        m.pose.position.y = float(-det.x_m)  # RViz: Y = left (flip sign)
        m.pose.position.z = float(-det.y_m)  # RViz: Z = up
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.5  # 50cm sphere
        r, g, b, a = COLORS_RGBA.get(det.hazard_level, (0.5, 0.5, 0.5, 0.8))
        m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = a
        m.lifetime = Duration(sec=0, nanosec=200_000_000)  # 200ms TTL
        marker_arr.markers.append(m)

        # Text marker dengan label
        t = Marker()
        t.header.frame_id = frame_id
        t.header.stamp = now
        t.ns = 'labels'
        t.id = i + 1000
        t.type = Marker.TEXT_VIEW_FACING
        t.action = Marker.ADD
        t.pose.position.x = float(det.z_m)
        t.pose.position.y = float(-det.x_m)
        t.pose.position.z = float(-det.y_m) + 0.6
        t.scale.z = 0.3
        t.color.r = 1.0; t.color.g = 1.0; t.color.b = 1.0; t.color.a = 1.0
        t.text = f"{det.label}\n{det.z_m:.1f}m"
        t.lifetime = Duration(sec=0, nanosec=200_000_000)
        marker_arr.markers.append(t)

    return marker_arr
```

### 4.3 Cara Integrasi ke gui_main.py

Upgrade `_flush_bev_update()` dengan backprojection yang benar:

```python
# Di DashboardGUI._flush_bev_update() — REPLACE existing logic
def _flush_bev_update(self):
    from app.core.services.bev_projection import (
        backproject_detections, render_bev_opencv, Intrinsics
    )

    dets = self._latest_bev_detections
    path_data = self._latest_path_data
    self._latest_bev_detections = None
    self._latest_path_data = None
    self._bev_update_pending = False

    if dets is None:
        return

    # Ambil intrinsik dari ROS CameraInfo (bukan hardcode)
    calib = self._latest_calib  # set dari ROSBagRunner payload
    if calib is None:
        return  # Tunggu sampai CameraInfo tersedia

    intrinsics = Intrinsics(
        fx=calib['fx'], fy=calib['fy'],
        cx=calib['cx'], cy=calib['cy']
    )

    # Ambil depth map terbaru (dari last frame payload)
    depth_m = self._latest_depth_map  # simpan di process_and_display_frame()

    # Backproject ke 3D
    dets_3d = backproject_detections(dets, depth_m, intrinsics)

    # Render BEV
    canvas_w = self.bev_canvas.winfo_width() or 400
    canvas_h = self.bev_canvas.winfo_height() or 600
    bev_img = render_bev_opencv(
        dets_3d,
        canvas_wh=(canvas_w, canvas_h),
        path_waypoints=path_data.get('waypoints') if path_data else None
    )

    # Update canvas (pattern existing tetap sama)
    img_rgb = cv2.cvtColor(bev_img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(img_pil)
    if self._bev_image_id is None:
        self._bev_image_id = self.bev_canvas.create_image(
            canvas_w // 2, canvas_h // 2, image=img_tk, anchor=tk.CENTER)
    else:
        self.bev_canvas.itemconfig(self._bev_image_id, image=img_tk)
    self.bev_canvas.image = img_tk
```

---

## Kritik Arsitektur & Rekomendasi Jetson

### ⚠️ Bottleneck #1: `rosbag2_py` Sequential Writer di SSD eMMC

Jetson Orin menggunakan eMMC yang write speed-nya terbatas (~200 MB/s). KITTI RGB frame 1242×375 = ~1.4 MB per frame. Pada 10 FPS, itu **14 MB/s** — aman. Tapi jika kamu menambahkan stereo kanan + LIDAR nanti, total bisa melampaui batas eMMC. **Solusi:** tulis bag ke NVMe M.2 via PCIe, bukan eMMC internal.

### ⚠️ Bottleneck #2: `cv2.cvtColor` + `ImageTk.PhotoImage` di Tkinter Loop

Di `_update_canvas()` existing, setiap frame ada:
1. `cv2.cvtColor` (BGR→RGB)
2. `Image.fromarray` (PIL)
3. `ImageTk.PhotoImage` (PIL→Tk)

Ini **3 full-frame copy** di CPU RAM setiap frame. Pada resolusi 1242×375 dengan 3 channel, satu frame = ~1.4 MB × 3 copy = **4.2 MB per frame di RAM**. Pada 30 FPS itu **126 MB/s RAM bandwidth hanya untuk display**. Di Jetson Orin yang share memory CPU-GPU, ini signifikan.

**Rekomendasi:** Resize frame ke canvas size **sebelum** konversi color, bukan setelahnya. Kamu sudah melakukannya (`cv2.resize` di `_update_canvas`) tapi `cvtColor` masih dilakukan pada full-res frame. Swap urutannya:

```python
# SEKARANG (suboptimal):
frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # Full res → color convert
resized = cv2.resize(frame_rgb, (new_w, new_h))          # Lalu resize

# SEHARUSNYA (optimal — ~4x lebih cepat):
resized = cv2.resize(frame_bgr, (new_w, new_h))          # Resize dulu (BGR, kecil)
frame_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)     # Lalu convert (kecil)
```

### ⚠️ Bottleneck #3: `cv_bridge` dan VRAM

**Jangan** load depth map ke GPU/VRAM hanya untuk konversi format. `cv_bridge.imgmsg_to_cv2()` mengembalikan CPU numpy array — ini sudah benar. Yang sering salah: developer menaruh depth map ke `torch.Tensor().cuda()` sebelum dikirim ke path planning, padahal path planning berbasis `scipy.ndimage` atau A* yang CPU-only. Ini buang waktu copy CPU↔GPU sia-sia.

**Rule:** Data naik ke GPU (VRAM) **hanya** untuk ONNX inference. Depth map tetap di CPU numpy untuk semua operasi geometri.

### ⚠️ Bottleneck #4: `message_filters.ApproximateTimeSynchronizer` Drop Rate

Jika timestamp di bag tidak benar-benar sinkron (misalnya karena konverter Fase 2 menulis timestamp yang sama persis untuk RGB dan depth), `ApproximateTimeSynchronizer` bisa drop semua pasangan jika `slop` terlalu ketat. **Test dulu** dengan:

```bash
ros2 bag play /data/kitti/rosbags/kitti_0001.mcap --rate 0.5
ros2 topic echo /kitti/camera/left/image_raw --field header.stamp | head -20
ros2 topic echo /kitti/camera/left/depth --field header.stamp | head -20
```

Jika timestamp identik (karena kita set stamp yang sama di konverter), ganti ke `ExactTimeSynchronizer` yang lebih efisien:

```python
self._sync = message_filters.TimeSynchronizer(
    [self._rgb_sub, self._depth_sub],
    queue_size=10
)
```

---

## Ringkasan Alur End-to-End

```
KITTI Raw Data (disk)
        │
        ▼ [Fase 1: kitti_download_subset.sh + remap]
/data/kitti/sequences/0001/
├── left_images/*.png     (RGB, uint8)
├── disparity/*.png       (16-bit, KITTI convention)
└── timestamps.txt
        │
        ▼ [Fase 2: kitti_to_rosbag.py]
/data/kitti/rosbags/kitti_0001.mcap
├── /kitti/camera/left/image_raw   (bgr8)
├── /kitti/camera/left/depth       (32FC1, meter)
└── /kitti/camera/left/camera_info
        │
        ▼ [ros2 bag play + Fase 3: ROSBagRunner]
PerceptionSubscriberNode
├── RGB frame  → frame_bgr (H,W,3) uint8
└── Depth map  → depth_m   (H,W)   float32 [meter]
        │
        ▼ [gui_main.py: process_and_display_frame()]
object_detection (ONNX inference)
+ path_planning (A*, OGM)
+ road_analysis
        │
        ▼ [Fase 4: backproject_detections() + render_bev_opencv()]
BEV Canvas (pixel) + RViz2 MarkerArray (3D)
```

---

*Generated for DavnFs/yolo_inference | Target: NVIDIA Jetson JetPack 6 (aarch64) | ROS 2 Humble*