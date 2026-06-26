#!/usr/bin/env python3
"""
analyze_sequence_detections.py
Runs YOLO detection and stereo depth extraction on the downloaded KITTI sequence
to analyze what obstacles are present and if there are any avoidance/warning scenes.
"""

import os
import sys
import numpy as np
import cv2

# Set path relative to project root
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from app.core.services.object_detection import (
    discover_models,
    set_active_model,
    get_combined_prediction_from_frame,
    perform_road_analysis
)

# Rebuild a simple loader to avoid GUI import overhead
class LocalKittiLoader:
    def __init__(self, root_dir: str, fx: float = 721.53, baseline: float = 0.54):
        self.root_dir = root_dir
        self.fx = fx
        self.baseline = baseline
        self.left_dir = os.path.join(root_dir, "left_images")
        self.disp_dir = os.path.join(root_dir, "disparity")
        self.frames = sorted(f for f in os.listdir(self.left_dir) if f.lower().endswith(".png"))

    def read_frame(self, idx):
        if idx < 0 or idx >= len(self.frames):
            return None, None
        fname = self.frames[idx]
        left_path = os.path.join(self.left_dir, fname)
        disp_path = os.path.join(self.disp_dir, fname)

        img = cv2.imread(left_path)
        disp_raw = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED)
        if img is None or disp_raw is None:
            return None, None

        # Calculate disparity
        disp = disp_raw.astype(np.float32) / 256.0
        disp = np.where(disp <= 0.1, 0.1, disp)
        depth = (self.baseline * self.fx) / disp
        depth = np.clip(depth, 0.5, 40.0)

        # Downscale like in simulation loader
        target_w = 640
        ratio = target_w / img.shape[1]
        target_h = int(img.shape[0] * ratio)
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        return img, depth


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze KITTI sequence detections.")
    parser.add_argument("--seq", default="0001", choices=["0001", "0002", "0005"], help="Sequence name (default: 0001)")
    args = parser.parse_args()

    models_dir = os.path.join(base_dir, "app", "models")
    configs = discover_models(models_dir)
    if not configs:
        print("No models found in app/models.")
        return

    # Choose first available model (e.g. YOLOV11_SIMAM_ASPP or BEST_COMBINE)
    model_name = "YOLOV11_SIMAM_ASPP" if "YOLOV11_SIMAM_ASPP" in configs else list(configs.keys())[0]
    print(f"Using model: {model_name}")
    set_active_model(model_name)

    seq_folder = f"kitti_{args.seq}"
    sequence_dir = os.path.join(base_dir, "stereo_dataset", seq_folder)
    if not os.path.isdir(sequence_dir):
        print(f"Sequence directory not found: {sequence_dir}")
        return

    loader = LocalKittiLoader(sequence_dir)
    n_frames = len(loader.frames)
    print(f"Loaded KITTI Sequence {args.seq} with {n_frames} frames.")

    print("\nScanning frames for dynamic obstacles (hazard_level = warning/danger)...")
    print("-" * 80)
    print(f"{'Frame':<8} | {'Object Label':<15} | {'Score':<6} | {'Distance (m)':<12} | {'Hazard Level':<12}")
    print("-" * 80)

    danger_count = 0
    warning_count = 0
    hazard_events = []

    for i in range(n_frames):
        res = loader.read_frame(i)
        if res[0] is None:
            continue
        img, depth = res

        results = get_combined_prediction_from_frame(
            img,
            configs,
            use_mc_dropout=False,
            pt_path=None,
            enable_path_planning=True,
            external_depth=depth
        )

        detections = results.get("detections", [])
        # Extract distances from depth map
        h, w = img.shape[:2]
        for det in detections:
            bbox = det.get("bounding_box", [0, 0, 0, 0])
            x1, y1, x2, y2 = [int(v) for v in bbox]
            patch = depth[y1:y2, x1:x2]
            valid_mask = (patch > 0.5) & (patch < 100.0)
            if np.any(valid_mask):
                true_dist = float(np.percentile(patch[valid_mask], 20))
            else:
                true_dist = 0.0

            if 0.5 < true_dist < 40.0:
                det["distance_m"] = round(true_dist, 2)
                if true_dist <= 15.0:
                    det["hazard_level"] = "danger"
                    danger_count += 1
                elif true_dist <= 35.0:
                    det["hazard_level"] = "warning"
                    warning_count += 1
                else:
                    det["hazard_level"] = "safe"

                if det["hazard_level"] in ["warning", "danger"]:
                    print(f"{i:<8} | {det['label']:<15} | {det['score']:<6.2f} | {det['distance_m']:<12} | {det['hazard_level']:<12}")
                    hazard_events.append((i, det['label'], det['distance_m'], det['hazard_level']))

    print("-" * 80)
    print(f"Scan complete. Total danger events: {danger_count}, total warning events: {warning_count}")

    if hazard_events:
        # Group by object to describe the trajectory
        from collections import defaultdict
        grouped = defaultdict(list)
        for frame, label, dist, haz in hazard_events:
            grouped[label].append((frame, dist, haz))

        print("\nSCENE ANALYSIS SUMMARY:")
        for label, events in grouped.items():
            first_frame, first_dist, _ = events[0]
            last_frame, last_dist, _ = events[-1]
            min_dist = min(e[1] for e in events)
            print(f"- A {label} is tracked from frame {first_frame} ({first_dist}m) to frame {last_frame} ({last_dist}m).")
            print(f"  It reaches a minimum distance of {min_dist}m.")
    else:
        print("\nNo close obstacles (danger/warning) detected in KITTI sequence 0001.")


if __name__ == "__main__":
    main()
