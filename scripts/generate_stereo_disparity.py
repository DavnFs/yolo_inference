#!/usr/bin/env python3
"""
generate_stereo_disparity.py
Hitung disparity dari stereo pair KITTI (image_02 kiri + image_03 kanan)
menggunakan StereoSGBM. Hasilnya dijamin SINKRON dengan RGB karena dihitung
dari frame yang sama.

Output: 16-bit PNG di format KITTI (pixel_value = disparity * 256)
Kompatibel langsung dengan disparity_to_ros_depth() di kitti_to_rosbag.py.

Usage:
    source .venv/bin/activate
    python3 scripts/generate_stereo_disparity.py \\
        /data/kitti/raw/2011_09_26/2011_09_26_drive_0001_sync \\
        /data/kitti/sequences/0001/disparity

    # Atau batch semua drive:
    python3 scripts/generate_stereo_disparity.py \\
        /data/kitti/raw/2011_09_26/2011_09_26_drive_0001_sync \\
        /data/kitti/sequences/0001/disparity --preview
"""

import sys
import os
import argparse
from pathlib import Path

import cv2
import numpy as np


# ── StereoSGBM Parameters ───────────────────────────────────────────────────
# Tuned untuk KITTI 1242x375, kemudian di-resize ke 640px
# numDisparities harus kelipatan 16; 128 = max depth ~5m pada KITTI baseline
_NUM_DISP    = 128   # max disparity (piksel)
_BLOCK_SIZE  = 9     # SAD window, harus ganjil
_P1          = 8  * 3 * _BLOCK_SIZE ** 2   # smoothness penalty kecil
_P2          = 32 * 3 * _BLOCK_SIZE ** 2   # smoothness penalty besar

def build_stereo_matcher() -> cv2.StereoSGBM:
    matcher = cv2.StereoSGBM.create(
        minDisparity=0,
        numDisparities=_NUM_DISP,
        blockSize=_BLOCK_SIZE,
        P1=_P1,
        P2=_P2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    return matcher


def compute_disparity_16bit(left_bgr: np.ndarray, right_bgr: np.ndarray,
                              matcher: cv2.StereoSGBM) -> np.ndarray:
    """
    Return 16-bit disparity PNG kompatibel KITTI.
    KITTI convention: stored_value = disparity_float * 256
    StereoSGBM output sudah dalam fixed-point Q4.4 (value / 16 = disparity float),
    jadi konversi: kitti_16bit = sgbm_output * 16  (= disp_float * 256)
    """
    left_gray  = cv2.cvtColor(left_bgr,  cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)

    disp_sgbm = matcher.compute(left_gray, right_gray)  # int16, fixed-point /16
    # Clip nilai negatif (invalid) ke 0
    disp_sgbm = np.clip(disp_sgbm, 0, None)
    # Konversi ke KITTI 16-bit format: *16 untuk fixed-point→float, *256 untuk KITTI
    # Tapi SGBM sudah /16 fixed-point, jadi: kitti = sgbm_raw * (256/16) = sgbm_raw * 16
    disp_16bit = (disp_sgbm * 16).astype(np.uint16)
    return disp_16bit


def generate_disparity_for_sequence(drive_dir: str, output_dir: str,
                                     preview: bool = False):
    drive_path  = Path(drive_dir)
    left_dir    = drive_path / "image_02" / "data"
    right_dir   = drive_path / "image_03" / "data"
    output_path = Path(output_dir)

    if not left_dir.exists():
        raise FileNotFoundError(f"Left camera dir tidak ditemukan: {left_dir}")
    if not right_dir.exists():
        raise FileNotFoundError(f"Right camera dir tidak ditemukan: {right_dir}")

    output_path.mkdir(parents=True, exist_ok=True)

    left_files = sorted(left_dir.glob("*.png"))
    if not left_files:
        raise RuntimeError(f"Tidak ada file PNG di {left_dir}")

    matcher = build_stereo_matcher()
    print(f"Drive : {drive_path.name}")
    print(f"Left  : {left_dir}")
    print(f"Right : {right_dir}")
    print(f"Output: {output_path}")
    print(f"Frames: {len(left_files)}")
    print()

    for i, left_path in enumerate(left_files):
        right_path = right_dir / left_path.name
        if not right_path.exists():
            print(f"  WARN [{i:04d}] right frame tidak ada: {right_path.name}, skip.")
            continue

        left_bgr  = cv2.imread(str(left_path),  cv2.IMREAD_COLOR)
        right_bgr = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
        if left_bgr is None or right_bgr is None:
            print(f"  WARN [{i:04d}] gagal baca frame, skip.")
            continue

        disp_16bit = compute_disparity_16bit(left_bgr, right_bgr, matcher)

        out_path = output_path / left_path.name
        cv2.imwrite(str(out_path), disp_16bit)

        if i % 20 == 0 or i == len(left_files) - 1:
            print(f"  [{i+1:4d}/{len(left_files)}] {left_path.name} → {out_path.name}")

        if preview and i == 0:
            # Visualisasi frame pertama
            disp_vis = (disp_16bit / 256.0).astype(np.float32)
            disp_vis_norm = cv2.normalize(disp_vis, None, 0, 255,
                                           cv2.NORM_MINMAX).astype(np.uint8)
            disp_color = cv2.applyColorMap(disp_vis_norm, cv2.COLORMAP_PLASMA)
            stacked = np.vstack([
                cv2.resize(left_bgr, (640, 192)),
                cv2.resize(disp_color, (640, 192)),
            ])
            cv2.imshow("Left | Disparity (frame 0)", stacked)
            print("\n  Preview: tekan sembarang tombol untuk lanjut...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    print(f"\nSelesai. {len(list(output_path.glob('*.png')))} disparity files di {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate sinkron stereo disparity dari KITTI drive"
    )
    parser.add_argument("drive_dir",
        help="Path ke folder drive, e.g. /data/kitti/raw/2011_09_26/2011_09_26_drive_0001_sync")
    parser.add_argument("output_dir",
        help="Path output disparity, e.g. /data/kitti/sequences/0001/disparity")
    parser.add_argument("--preview", action="store_true",
        help="Tampilkan preview frame 0 (butuh display)")
    args = parser.parse_args()

    generate_disparity_for_sequence(args.drive_dir, args.output_dir, args.preview)


if __name__ == "__main__":
    main()
