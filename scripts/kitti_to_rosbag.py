#!/usr/bin/env python3
"""
kitti_to_rosbag.py
Konversi KITTI sequence folder ke ROS 2 .mcap bag.
Kompatibel: Python 3.10, aarch64, ROS 2 Humble, JetPack 6.

Usage:
    source /opt/ros/humble/setup.bash
    python3 scripts/kitti_to_rosbag.py \\
        /data/kitti/sequences/0001 \\
        /data/kitti/calib/2011_09_26/calib_cam_to_cam.txt \\
        /data/kitti/rosbags/kitti_0001.mcap

Topics yang ditulis:
    /kitti/camera/left/image_raw    (sensor_msgs/Image, bgr8)
    /kitti/camera/left/depth        (sensor_msgs/Image, 32FC1, meter)
    /kitti/camera/left/camera_info  (sensor_msgs/CameraInfo)

Peringatan:
- Jangan jalankan ini di Jetson saat inference berjalan. Proses serialisasi
  rosbag2 sangat I/O-bound dan akan bersaing dengan NVME bandwidth.
- Gunakan format MCAP, bukan DB3. DB3 (SQLite) punya write amplification
  yang buruk untuk data besar. MCAP jauh lebih efisien.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

try:
    import rclpy
    from rclpy.serialization import serialize_message
    import rosbag2_py
    from sensor_msgs.msg import Image, CameraInfo
    from builtin_interfaces.msg import Time
except ImportError as e:
    print(f"[ERROR] ROS 2 Python packages tidak ditemukan: {e}")
    print("Jalankan: source /opt/ros/humble/setup.bash")
    sys.exit(1)


# ──────────────────────────────────────────────
# 1. Kalibrasi Parser
# ──────────────────────────────────────────────

def parse_kitti_calib(calib_path: str) -> dict:
    """
    Parse calib_cam_to_cam.txt KITTI.
    Return dict dengan fx, fy, cx, cy, baseline, P_rect_02.
    """
    calib = {}
    P2 = None
    P3 = None
    with open(calib_path, 'r') as f:
        for line in f:
            if line.startswith('P_rect_02:'):
                vals = list(map(float, line.split()[1:]))
                P2 = np.array(vals).reshape(3, 4)
                calib['fx'] = P2[0, 0]
                calib['fy'] = P2[1, 1]
                calib['cx'] = P2[0, 2]
                calib['cy'] = P2[1, 2]
                calib['P_rect_02'] = P2
            elif line.startswith('P_rect_03:'):
                vals = list(map(float, line.split()[1:]))
                P3 = np.array(vals).reshape(3, 4)
                calib['P_rect_03'] = P3

    if 'fx' not in calib:
        raise ValueError(f"P_rect_02 tidak ditemukan di {calib_path}")

    # Baseline benar = selisih translasi lateral cam_02 dan cam_03
    # P[0,3] = -fx * baseline_offset_dari_origin
    # baseline_02_03 = (P3[0,3] - P2[0,3]) / (-fx)
    if P2 is not None and P3 is not None:
        fx = calib['fx']
        tx2 = P2[0, 3] / (-fx)   # offset lateral cam_02 dari origin [m]
        tx3 = P3[0, 3] / (-fx)   # offset lateral cam_03 dari origin [m]
        calib['baseline'] = abs(tx3 - tx2)
    else:
        calib['baseline'] = 0.54

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
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
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
            try:
                # KITTI timestamps: nanosecond precision (9 digits setelah titik)
                # Python %f hanya handle 6 digit — truncate ke microseconds
                if '.' in line:
                    base, frac = line.split('.', 1)
                    line = base + '.' + frac[:6]
                dt = datetime.strptime(line, '%Y-%m-%d %H:%M:%S.%f')
            except ValueError:
                continue
            if ref_dt is None:
                ref_dt = dt
            delta_ns = int((dt - ref_dt).total_seconds() * 1e9)
            timestamps_ns.append(delta_ns)
    return timestamps_ns


# ──────────────────────────────────────────────
# 3. Image Converters
# ──────────────────────────────────────────────

def bgr_to_ros_image(img_bgr: np.ndarray, stamp: Time, frame_id: str) -> Image:
    """Konversi BGR image ke sensor_msgs/Image (encoding: bgr8)."""
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
    Konversi 16-bit KITTI disparity PNG → depth map dalam meter → sensor_msgs/Image (32FC1).

    KITTI 16-bit disparity convention: actual_disparity = pixel_value / 256.0
    Depth [m] = (fx * baseline) / disparity
    """
    disp_f = disp_raw.astype(np.float32) / 256.0
    valid_mask = disp_f > 0.1
    depth_m = np.where(valid_mask, (fx * baseline) / disp_f, 0.0).astype(np.float32)
    depth_m = np.clip(depth_m, 0.0, 80.0)

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
    """
    seq_path = Path(seq_dir)
    left_dir = seq_path / 'left_images'
    disp_dir = seq_path / 'disparity'
    ts_file  = seq_path / 'timestamps.txt'

    for p in [left_dir, disp_dir, ts_file]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required path: {p}")

    calib = parse_kitti_calib(calib_path)
    timestamps_ns = parse_kitti_timestamps(str(ts_file))

    frame_files = sorted(f for f in left_dir.iterdir() if f.suffix == '.png')
    n_frames = min(len(frame_files), len(timestamps_ns))

    if n_frames == 0:
        raise RuntimeError(f"No matching frames/timestamps found in {seq_dir}")

    print(f"Converting {n_frames} frames from {seq_dir}")
    print(f"  Calib: fx={calib['fx']:.2f}, fy={calib['fy']:.2f}, "
          f"cx={calib['cx']:.2f}, cy={calib['cy']:.2f}, "
          f"baseline={calib['baseline']:.4f}m")
    print(f"  Output: {output_bag}")

    os.makedirs(os.path.dirname(output_bag) or '.', exist_ok=True)

    converter_opts = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )

    # Coba mcap dulu (efisien). Fallback ke sqlite3 (db3) jika plugin mcap
    # belum terinstall (ros-humble-rosbag2-storage-mcap).
    writer = rosbag2_py.SequentialWriter()
    try:
        storage_opts = rosbag2_py.StorageOptions(uri=output_bag, storage_id='mcap')
        writer.open(storage_opts, converter_opts)
        print("  Storage: mcap")
    except RuntimeError:
        print("  WARN: plugin 'mcap' tidak tersedia. Fallback ke 'sqlite3' (db3).")
        print("  Untuk mcap, install: sudo apt install ros-humble-rosbag2-storage-mcap")
        db3_uri = output_bag
        if db3_uri.endswith('.mcap'):
            db3_uri = db3_uri[:-len('.mcap')]
        writer = rosbag2_py.SequentialWriter()
        storage_opts = rosbag2_py.StorageOptions(uri=db3_uri, storage_id='sqlite3')
        writer.open(storage_opts, converter_opts)
        print(f"  Storage: sqlite3 → {db3_uri}")

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

    sample_img = cv2.imread(str(frame_files[0]))
    if sample_img is None:
        raise RuntimeError(f"Cannot read sample frame: {frame_files[0]}")
    cam_info_msg = build_camera_info_msg(calib, sample_img.shape, frame_id='camera_left')

    baseline = calib['baseline']
    fx = calib['fx']

    print(f"  Writing frames...")
    for i, frame_path in enumerate(frame_files[:n_frames]):
        if i % 50 == 0:
            print(f"  Frame {i}/{n_frames} ({100*i//n_frames}%)...")

        ts_ns = timestamps_ns[i]
        stamp = ns_to_ros_stamp(ts_ns)

        img_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"  WARN: Cannot read {frame_path.name}, skipping.")
            continue

        disp_path = disp_dir / frame_path.name
        if not disp_path.exists():
            print(f"  WARN: No disparity for {frame_path.name}, skipping.")
            continue
        disp_raw = cv2.imread(str(disp_path), cv2.IMREAD_UNCHANGED)
        if disp_raw is None:
            print(f"  WARN: Cannot read disparity {disp_path.name}, skipping.")
            continue

        img_msg   = bgr_to_ros_image(img_bgr, stamp, 'camera_left')
        depth_msg = disparity_to_ros_depth(disp_raw, fx, baseline, stamp, 'camera_left')
        cam_info_msg.header.stamp = stamp

        writer.write('/kitti/camera/left/image_raw',
                     serialize_message(img_msg), ts_ns)
        writer.write('/kitti/camera/left/depth',
                     serialize_message(depth_msg), ts_ns)
        writer.write('/kitti/camera/left/camera_info',
                     serialize_message(cam_info_msg), ts_ns)

    del writer
    print(f"Bag saved: {output_bag}")
    print(f"Verifikasi: ros2 bag info {output_bag}")


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python3 kitti_to_rosbag.py <seq_dir> <calib_path> <output.mcap>")
        print("")
        print("Example:")
        print("  python3 scripts/kitti_to_rosbag.py \\")
        print("      /data/kitti/sequences/0001 \\")
        print("      /data/kitti/calib/2011_09_26/calib_cam_to_cam.txt \\")
        print("      /data/kitti/rosbags/kitti_0001.mcap")
        sys.exit(1)

    convert_sequence(
        seq_dir=sys.argv[1],
        calib_path=sys.argv[2],
        output_bag=sys.argv[3]
    )
