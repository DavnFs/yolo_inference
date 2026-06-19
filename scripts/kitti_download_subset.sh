#!/usr/bin/env bash
# =============================================================
# kitti_download_subset.sh
# Download minimal subset KITTI untuk testing pipeline YOLO
# Target: ~800 MB total (3 drive sequences)
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
SCENE_FLOW_URL="${KITTI_BASE}/data_scene_flow.zip"

mkdir -p "${OUTPUT_DIR}/raw" "${OUTPUT_DIR}/disparity_gt" "${OUTPUT_DIR}/calib"

echo "[1/3] Downloading calibration files..."
wget -q --show-progress -c -P "${OUTPUT_DIR}/calib" "${CALIB_URL}"
unzip -q -o "${OUTPUT_DIR}/calib/2011_09_26_calib.zip" -d "${OUTPUT_DIR}/calib/"
echo "  Calibration extracted."

echo "[2/3] Downloading RGB sequences..."
for DRIVE in "${DRIVES[@]}"; do
    DATE="${DRIVE:0:10}"
    DRIVE_ID="${DRIVE:17:4}"   # "2011_09_26_drive_XXXX_sync" → index 17, 4 chars = "XXXX"
    URL="${KITTI_BASE}/raw_data/${DATE}_drive_${DRIVE_ID}/${DRIVE}.zip"
    echo "  -> Downloading ${DRIVE}..."
    wget -q --show-progress -c -P "${OUTPUT_DIR}/raw" "${URL}" || {
        echo "  WARN: Failed to download ${DRIVE}, skipping."
        continue
    }
    unzip -q -o "${OUTPUT_DIR}/raw/${DRIVE}.zip" -d "${OUTPUT_DIR}/raw/"
    rm -f "${OUTPUT_DIR}/raw/${DRIVE}.zip"
    echo "  -> ${DRIVE} extracted."
done

echo "[3/3] Downloading Scene Flow disparity (dense depth GT, ~194 MB)..."
echo "  NOTE: This contains disparity for KITTI training sequences (000-200)."
echo "  Only a subset overlaps with the drives above."
wget -q --show-progress -c -P "${OUTPUT_DIR}/disparity_gt" "${SCENE_FLOW_URL}" || {
    echo "  WARN: Scene Flow download failed. Disparity will need manual download."
    echo "  URL: ${SCENE_FLOW_URL}"
}
if [ -f "${OUTPUT_DIR}/disparity_gt/data_scene_flow.zip" ]; then
    unzip -q -o "${OUTPUT_DIR}/disparity_gt/data_scene_flow.zip" \
        -d "${OUTPUT_DIR}/disparity_gt/"
    echo "  Scene Flow extracted."
fi

echo ""
echo "Done. Dataset tersedia di: ${OUTPUT_DIR}"
echo "Jalankan remap_kitti_structure.sh untuk menyesuaikan struktur folder."
