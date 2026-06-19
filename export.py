"""
Download Depth Anything V2 Small ONNX and verify it with ONNX Runtime.

Output:
    app/models/depth_anything_v2_vits.onnx

Why this downloads an ONNX file directly:
    The official Depth Anything V2 Small repo provides a PyTorch checkpoint.
    A matching ONNX export is already published by onnx-community and avoids
    handwritten architecture drift such as pos_embed/reg_token mismatches.

Requirements:
    pip install huggingface_hub onnx onnxruntime  # or onnxruntime-gpu for GPU

Usage:
    python export.py
"""

import os
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from huggingface_hub import hf_hub_download


REPO_ID = "onnx-community/Depth-Anything-V2-Small"
REPO_FILENAME = "onnx/model.onnx"
OUTPUT_PATH = Path("app") / "models" / "depth_anything_v2_vits.onnx"
DEPTH_INPUT_SIZE = 518


def download_onnx_model(output_path: Path = OUTPUT_PATH) -> Path:
    print("[1/3] Downloading Depth Anything V2 Small ONNX from HuggingFace...")
    cached_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=REPO_FILENAME,
        local_dir="./tmp_depth_anything_v2_small",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached_path, output_path)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"      Source: {cached_path}")
    print(f"      Saved : {output_path} ({size_mb:.1f} MB)")
    return output_path


def validate_onnx_model(model_path: Path):
    print("[2/3] Validating ONNX graph...")
    model = onnx.load(str(model_path))
    onnx.checker.check_model(model)
    print("      ONNX checker PASSED.")


def _select_providers():
    available = ort.get_available_providers()
    print(f"      Available providers: {available}")
    for ep in ("TensorrtExecutionProvider", "CUDAExecutionProvider", "DmlExecutionProvider"):
        if ep in available:
            return [ep, "CPUExecutionProvider"]
    print("      WARNING: No GPU provider available. Falling back to CPU.")
    return ["CPUExecutionProvider"]


def verify_onnx_inference(model_path: Path):
    print("[3/3] Verifying inference with ONNX Runtime...")
    providers = _select_providers()
    session = ort.InferenceSession(str(model_path), providers=providers)
    active_providers = session.get_providers()
    print(f"      Active providers: {active_providers}")

    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    input_name = input_meta.name
    output_name = output_meta.name

    print(f"      Input : {input_name} {input_meta.shape} {input_meta.type}")
    print(f"      Output: {output_name} {output_meta.shape} {output_meta.type}")

    dummy = np.random.rand(1, 3, DEPTH_INPUT_SIZE, DEPTH_INPUT_SIZE).astype(np.float32)
    outputs = session.run(None, {input_name: dummy})
    depth = np.asarray(outputs[0])

    print(f"      Output shape: {depth.shape}")
    print(f"      Output range: [{depth.min():.4f}, {depth.max():.4f}]")

    gpu_eps = ("TensorrtExecutionProvider", "CUDAExecutionProvider", "DmlExecutionProvider")
    if any(ep in active_providers for ep in gpu_eps):
        print("      GPU inference verification PASSED.")
    else:
        print("      CPU inference verification PASSED.")


def main():
    model_path = download_onnx_model()
    validate_onnx_model(model_path)
    verify_onnx_inference(model_path)

    print("\nDone. Use this path in the dashboard:")
    print(f"  {os.path.abspath(model_path)}")


if __name__ == "__main__":
    main()
