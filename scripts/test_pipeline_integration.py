import os
import cv2
import numpy as np

from app.core.services.object_detection import get_combined_prediction_from_frame, set_active_model
from gui_main import StereoSimLoader

def test_integration():
    print("Testing data integration pipeline with real dataset...")
    
    stereo_loader = StereoSimLoader("stereo_dataset/kitti_0005")
    if not stereo_loader.is_opened():
        print("Failed to open stereo dataset.")
        return
        
    ret, frame_bgr, depth_m = stereo_loader.read_frame(0)
    if not ret:
        print("Failed to read frame 0.")
        return
        
    print(f"Loaded frame shape: {frame_bgr.shape}, depth shape: {depth_m.shape}")
    
    model_path = os.path.join("app", "models", "best_combine.onnx")
    labels_path = os.path.join("app", "models", "labels.txt")
    
    if not os.path.exists(labels_path):
        with open(labels_path, "w") as f:
            f.write("car\npedestrian\nbicycle\n")
            
    model_configs = {
        "BEST_COMBINE": {
            "path": model_path,
            "type": "YOLO",
            "labels_path": labels_path
        }
    }
    
    set_active_model("BEST_COMBINE")
    
    camera_intrinsics = {
        "fx": 721.53 * 640/1242,
        "fy": 721.53 * 193/375,
        "cx": 609.55 * 640/1242,
        "cy": 172.85 * 193/375
    }
    
    results = get_combined_prediction_from_frame(
        frame_bgr=frame_bgr,
        model_configs=model_configs,
        enable_path_planning=True,
        external_depth=depth_m,
        camera_intrinsics=camera_intrinsics
    )
    
    print("Integration test complete.")
    print("Detections count:", len(results.get("detections", [])))
    print("Path Planning Output:", "Yes" if "path_planning" in results else "No")
    if "error" in results:
        print("Error:", results["error"])

if __name__ == "__main__":
    test_integration()
