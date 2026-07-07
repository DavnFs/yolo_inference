#!/usr/bin/env python3
"""
setup_kitti_dataset.py
Automates downloading, unzipping, and generating ZED-compatible metric depth
.npy arrays for a subset of the KITTI dataset.
"""

import argparse
import os
import shutil
import sys
import urllib.request
import zipfile

import cv2
import numpy as np

KITTI_BASE_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti"
DRIVES_METADATA = {
    "0001": {"drive_name": "2011_09_26_drive_0001_sync", "date": "2011_09_26"},
    "0002": {"drive_name": "2011_09_26_drive_0002_sync", "date": "2011_09_26"},
    "0005": {"drive_name": "2011_09_26_drive_0005_sync", "date": "2011_09_26"},
    "0009": {"drive_name": "2011_09_26_drive_0009_sync", "date": "2011_09_26"},
    "0011": {"drive_name": "2011_09_26_drive_0011_sync", "date": "2011_09_26"},
    "0013": {"drive_name": "2011_09_26_drive_0013_sync", "date": "2011_09_26"},
    "0014": {"drive_name": "2011_09_26_drive_0014_sync", "date": "2011_09_26"},
    "0015": {"drive_name": "2011_09_26_drive_0015_sync", "date": "2011_09_26"},
    "0017": {"drive_name": "2011_09_26_drive_0017_sync", "date": "2011_09_26"},
    "0018": {"drive_name": "2011_09_26_drive_0018_sync", "date": "2011_09_26"},
    "0019": {"drive_name": "2011_09_26_drive_0019_sync", "date": "2011_09_26"},
    "0020": {"drive_name": "2011_09_26_drive_0020_sync", "date": "2011_09_26"},
    "0022": {"drive_name": "2011_09_26_drive_0022_sync", "date": "2011_09_26"},
    "0023": {"drive_name": "2011_09_26_drive_0023_sync", "date": "2011_09_26"},
}


def download_progress_callback(block_num, block_size, total_size):
    downloaded = block_num * block_size
    percent = min(100, int(downloaded * 100 / total_size)) if total_size > 0 else 0
    mb_downloaded = downloaded / (1024 * 1024)
    mb_total = total_size / (1024 * 1024) if total_size > 0 else 0
    sys.stdout.write(
        f"\r  Downloading: {percent}% [{mb_downloaded:.1f}/{mb_total:.1f} MB]"
    )
    sys.stdout.flush()


def download_file(url, target_path):
    print(f"Fetching: {url}")
    try:
        urllib.request.urlretrieve(url, target_path, download_progress_callback)
        print("\n  Download complete.")
    except Exception as e:
        print(f"\n  Error downloading file: {e}")
        raise e


def unzip_file(zip_path, extract_to):
    print(f"Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print("  Extraction complete.")


def parse_kitti_calib(calib_file_path):
    """Parses calib_cam_to_cam.txt to get focal length (fx) and baseline."""
    fx = baseline = cx = cy = None

    with open(calib_file_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        # Color left camera (image_02)
        if line.startswith("P_rect_02:"):
            parts = line.strip().split()[1:]
            p2 = np.array(parts, dtype=np.float32).reshape(3, 4)
            fx = p2[0, 0]
            cx = p2[0, 2]
            cy = p2[1, 2]

        # Color right camera (image_03)
        if line.startswith("P_rect_03:"):
            parts = line.strip().split()[1:]
            p3 = np.array(parts, dtype=np.float32).reshape(3, 4)
            # Baseline = |tx / fx| where tx is P[0,3]
            baseline = abs(p3[0, 3] / p3[0, 0])

    if fx is None or baseline is None:
        print(f"\nError: Could not parse P_rect_02/P_rect_03 from {calib_file_path}")
        sys.exit(1)

    return fx, baseline, cx, cy


def build_stereo_matcher() -> cv2.StereoSGBM:
    block_size = 9
    matcher = cv2.StereoSGBM.create(
        minDisparity=0,
        numDisparities=128,
        blockSize=block_size,
        P1=8 * 3 * block_size**2,
        P2=32 * 3 * block_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    return matcher


def main():
    parser = argparse.ArgumentParser(
        description="Download KITTI and generate ZED-compatible depth .npy"
    )
    parser.add_argument(
        "--seq", choices=["0001", "0002", "0005", "0009", "0011", "0013", "0014", "0015", "0017", "0018", "0019", "0020", "0022", "0023"], default="0001"
    )
    parser.add_argument("--dest", default="stereo_dataset/kitti")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_seq_dir = os.path.join(base_dir, args.dest + "_" + args.seq)
    temp_dir = os.path.join(base_dir, "tmp_kitti_setup")

    os.makedirs(target_seq_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    drive_info = DRIVES_METADATA[args.seq]
    drive_name = drive_info["drive_name"]
    drive_date = drive_info["date"]

    print("=================================================================")
    print("KITTI DATASET SETUP SCRIPT (ZED 2i Simulation Mode)")
    print(f"Target Sequence: {args.seq} ({drive_name})")
    print("=================================================================\n")

    # 1. Download Calibration
    calib_zip = os.path.join(temp_dir, "calib.zip")
    calib_url = f"{KITTI_BASE_URL}/raw_data/{drive_date}_calib.zip"
    if not os.path.exists(calib_zip):
        print("[1/5] Downloading Calibration Zip...")
        download_file(calib_url, calib_zip)
    else:
        print("[1/5] Calibration zip already downloaded.")

    calib_extract_path = os.path.join(temp_dir, "calib")
    if not os.path.exists(calib_extract_path):
        unzip_file(calib_zip, calib_extract_path)

    # Parse calibration
    calib_file = os.path.join(calib_extract_path, drive_date, "calib_cam_to_cam.txt")
    if not os.path.exists(calib_file):
        print("Error: calib_cam_to_cam.txt not found!")
        sys.exit(1)

    fx, baseline, cx, cy = parse_kitti_calib(calib_file)
    print(f"  Calibration parsed: fx={fx}, baseline={baseline}m, cx={cx}, cy={cy}")

    # 2. Download RGB sequence
    seq_zip = os.path.join(temp_dir, f"{drive_name}.zip")
    seq_url = (
        f"{KITTI_BASE_URL}/raw_data/{drive_date}_drive_{args.seq}/{drive_name}.zip"
    )
    if not os.path.exists(seq_zip):
        print(f"\n[2/5] Downloading Sequence {args.seq} Zip...")
        download_file(seq_url, seq_zip)
    else:
        print(f"\n[2/5] Sequence zip already downloaded.")

    seq_extract_path = os.path.join(temp_dir, "sequence")
    if not os.path.exists(os.path.join(seq_extract_path, drive_date, drive_name)):
        unzip_file(seq_zip, seq_extract_path)

    raw_left_dir = os.path.join(
        seq_extract_path, drive_date, drive_name, "image_02", "data"
    )
    raw_right_dir = os.path.join(
        seq_extract_path, drive_date, drive_name, "image_03", "data"
    )

    # 3. Structure folders & copy Left images
    dest_left_dir = os.path.join(target_seq_dir, "left_images")
    dest_depth_dir = os.path.join(
        target_seq_dir, "depth_npy"
    )  # Changed from disparity to depth_npy
    os.makedirs(dest_left_dir, exist_ok=True)
    os.makedirs(dest_depth_dir, exist_ok=True)

    print("\n[3/5] Copying left camera images...")
    left_files = sorted(
        f for f in os.listdir(raw_left_dir) if f.lower().endswith(".png")
    )
    for f in left_files:
        shutil.copy2(os.path.join(raw_left_dir, f), os.path.join(dest_left_dir, f))
    print(f"  Copied {len(left_files)} files.")

    # 4. Save intrinsics for later use in the pipeline
    print("\n[4/5] Saving camera intrinsics for pipeline...")
    intrinsics = {
        "fx": float(fx),
        "fy": float(fx),
        "cx": float(cx),
        "cy": float(cy),
        "baseline": float(baseline),
    }
    np.save(os.path.join(target_seq_dir, "intrinsics.npy"), intrinsics)

    # 5. Generate metric depth .npy arrays
    print("\n[5/5] Generating ZED-compatible metric depth maps (.npy)...")
    matcher = build_stereo_matcher()
    generated_count = 0

    for f in left_files:
        left_bgr = cv2.imread(os.path.join(raw_left_dir, f), cv2.IMREAD_COLOR)
        right_bgr = cv2.imread(os.path.join(raw_right_dir, f), cv2.IMREAD_COLOR)

        if left_bgr is None or right_bgr is None:
            continue

        # Compute disparity
        left_gray = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)
        disp_sgbm = matcher.compute(left_gray, right_gray)
        disp_sgbm = np.clip(disp_sgbm, 0, None)

        # Convert SGBM raw output to actual float disparity (divide by 16)
        disp_float = disp_sgbm.astype(np.float32) / 16.0

        # Convert Disparity to Metric Depth (ZED format)
        # Depth = (Focal_Length * Baseline) / Disparity
        depth_meters = np.zeros_like(disp_float, dtype=np.float32)
        valid_disp = disp_float > 0.1  # Avoid division by zero
        depth_meters[valid_disp] = (fx * baseline) / disp_float[valid_disp]

        # Clip depth to 80 meters (ZED 2i max range is ~20-40m, but KITTI can see further)
        depth_meters = np.clip(depth_meters, 0, 80.0)

        # Save as .npy float32
        frame_id = os.path.splitext(f)[0]
        np.save(os.path.join(dest_depth_dir, f"{frame_id}.npy"), depth_meters)

        generated_count += 1
        if generated_count % 20 == 0 or generated_count == len(left_files):
            print(f"  Processed {generated_count}/{len(left_files)} frames...")

    # Cleanup temp dir
    # shutil.rmtree(temp_dir, ignore_errors=True)

    print("\n=================================================================")
    print("SUCCESS: KITTI Dataset is ready for ZED 2i simulation!")
    print(f"Folder: {target_seq_dir}")
    print(
        "Contains: left_images/ (RGB), depth_npy/ (32-bit float meters), intrinsics.npy"
    )
    print("=================================================================")


if __name__ == "__main__":
    main()
