#!/usr/bin/env bash
# =============================================================
# remap_kitti_structure.sh
# Remap KITTI raw folder structure → StereoSimLoader-compatible
# dan siap untuk kitti_to_rosbag.py
#
# Input:  /data/kitti/raw/2011_09_26/2011_09_26_drive_XXXX_sync/
# Output: /data/kitti/sequences/XXXX/
#           left_images/ → symlink ke image_02/data/
#           disparity/   → copy dari data_scene_flow/training/disp_noc_0/
#           timestamps.txt
# =============================================================
set -euo pipefail

KITTI_ROOT="${1:-/data/kitti}"
KITTI_RAW="${KITTI_ROOT}/raw/2011_09_26"
SCENE_FLOW_DIR="${KITTI_ROOT}/disparity_gt/training/disp_noc_0"
TARGET="${KITTI_ROOT}/sequences"

if [ ! -d "${KITTI_RAW}" ]; then
    echo "ERROR: KITTI raw directory not found: ${KITTI_RAW}"
    echo "Jalankan kitti_download_subset.sh terlebih dahulu."
    exit 1
fi

mkdir -p "${TARGET}"

echo "Mapping KITTI raw → sequences..."
for DRIVE_DIR in "${KITTI_RAW}"/2011_09_26_drive_*/; do
    if [ ! -d "${DRIVE_DIR}" ]; then
        continue
    fi

    DRIVE_NUM=$(basename "${DRIVE_DIR}" | grep -oP 'drive_\K\d+')
    SEQ_DIR="${TARGET}/${DRIVE_NUM}"
    mkdir -p "${SEQ_DIR}/disparity"

    # Symlink image_02/data/ sebagai left_images/ (hemat storage)
    if [ -d "${DRIVE_DIR}image_02/data" ]; then
        ln -sfn "${DRIVE_DIR}image_02/data" "${SEQ_DIR}/left_images"
        echo "  [${DRIVE_NUM}] left_images -> ${DRIVE_DIR}image_02/data"
    else
        echo "  WARN: No image_02/data in ${DRIVE_DIR}"
    fi

    # Copy timestamps
    if [ -f "${DRIVE_DIR}image_02/timestamps.txt" ]; then
        cp "${DRIVE_DIR}image_02/timestamps.txt" "${SEQ_DIR}/timestamps.txt"
        echo "  [${DRIVE_NUM}] timestamps.txt copied."
    fi

    # Count frames
    N_FRAMES=$(ls "${SEQ_DIR}/left_images/"*.png 2>/dev/null | wc -l || echo 0)
    echo "  [${DRIVE_NUM}] ${N_FRAMES} RGB frames available."

    # Copy disparity dari scene flow (naming adalah 000000.png, 000001.png, ...)
    # CATATAN: Scene Flow berisi disparity untuk KITTI Training Set (000-200), bukan
    # raw drives yang kita download. Frame tidak berkorespondensi secara semantik.
    # Ini hanya untuk testing pipeline — depth VALID secara numerik, tapi bukan
    # depth dari scene yang sama dengan RGB-nya.
    # Hanya berlaku jika scene flow tersedia
    if [ -d "${SCENE_FLOW_DIR}" ]; then
        echo "  [${DRIVE_NUM}] Copying disparity from scene_flow (first ${N_FRAMES} files)..."
        DISP_FILES=($(ls "${SCENE_FLOW_DIR}"/*.png 2>/dev/null | head -n "${N_FRAMES}"))
        IDX=0
        for DISP_FILE in "${DISP_FILES[@]}"; do
            # Rename ke match KITTI naming: 0000000000.png
            TARGET_FNAME=$(printf '%010d.png' ${IDX})
            cp "${DISP_FILE}" "${SEQ_DIR}/disparity/${TARGET_FNAME}"
            IDX=$((IDX + 1))
        done
        echo "  [${DRIVE_NUM}] ${IDX} disparity files copied."
    else
        echo "  WARN: Scene Flow dir not found: ${SCENE_FLOW_DIR}"
        echo "  INFO: Run kitti_download_subset.sh to get disparity data."
    fi
done

echo ""
echo "Remap selesai. Struktur tersedia di: ${TARGET}"
echo ""
echo "Untuk konversi ke rosbag, jalankan:"
echo "  python3 scripts/kitti_to_rosbag.py \\"
echo "      ${TARGET}/0001 \\"
echo "      ${KITTI_ROOT}/calib/2011_09_26/calib_cam_to_cam.txt \\"
echo "      ${KITTI_ROOT}/rosbags/kitti_0001.mcap"
