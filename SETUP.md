# Setup Guide — Road Analysis Dashboard

Panduan ini mencakup dua platform: **NVIDIA Jetson (JetPack 6)** dan **Windows 10/11**.
Keduanya menjalankan kode yang sama. Perbedaan hanya pada GPU backend, ROS 2, dan cara install dependensi.

---

## Daftar Isi

- [Arsitektur Singkat](#arsitektur-singkat)
- [Jetson Setup](#jetson-setup)
- [Windows Setup](#windows-setup)
- [Dataset KITTI](#dataset-kitti)
- [Menjalankan Aplikasi](#menjalankan-aplikasi)
- [Perbedaan Jetson vs Windows (Ringkasan)](#perbedaan-jetson-vs-windows-ringkasan)
- [Troubleshooting](#troubleshooting)

---

## Arsitektur Singkat

```
gui_main.py              ← Entry point (Tkinter dashboard)
app/
  core/services/
    object_detection.py  ← Inference YOLO/Faster-RCNN via ONNX Runtime
    bev_projection.py    ← Dense BEV point cloud (GPU CuPy / CPU numpy)
    path_planning.py     ← Occupancy grid + A* path planner
    ros2_perception_node.py  ← ROS 2 subscriber (graceful degradasi jika tidak ada ROS)
  models/
    baseline.onnx        ← Model YOLO11n (auto-discovered)
    labels.txt           ← 28 class labels
scripts/
  generate_stereo_disparity.py  ← Hitung disparity dari stereo pair KITTI
  kitti_to_rosbag.py    ← Konversi KITTI sequence → ROS 2 bag
  remap_kitti_structure.sh      ← Remap folder KITTI raw → sequences/
  kitti_download_subset.sh      ← Download subset KITTI dari AWS S3
```

**Input modes yang tersedia:**

| Mode | Membutuhkan ROS 2 | Depth Source | Path Planning |
|------|:-:|---|:-:|
| Stereo Dataset Sim | Tidak | Disparity PNG → depth meter | ✅ |
| ROS 2 Bag | **Ya** | Topik `/kitti/camera/left/depth` | ✅ |
| Video File | Tidak | Tidak ada (deteksi saja) | ❌ |
| Webcam | Tidak | Tidak ada (deteksi saja) | ❌ |
| Image File | Tidak | Tidak ada (deteksi saja) | ❌ |

---

## Jetson Setup

### Persyaratan Sistem

| Komponen | Versi |
|---|---|
| Hardware | NVIDIA Jetson AGX Orin / Orin NX / Orin Nano |
| JetPack | 6.x (diuji pada 6.0 / 6.1) |
| OS | Ubuntu 22.04 aarch64 |
| CUDA | 12.2 (bundled dengan JetPack) |
| TensorRT | 8.6.x (bundled dengan JetPack) |
| Python | 3.10 (system default JetPack 6) |
| ROS 2 | Humble |

### 1. Install ROS 2 Humble

```bash
# Jika belum terinstall
sudo apt update && sudo apt install -y ros-humble-desktop

# Tambah ke .bashrc (opsional, bisa juga source manual per sesi)
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

### 2. Install sistem dependencies

```bash
sudo apt update
sudo apt install -y \
    ros-humble-cv-bridge \
    ros-humble-message-filters \
    ros-humble-sensor-msgs \
    python3-pip \
    python3-venv \
    libgl1-mesa-glx \
    libglib2.0-0
```

### 3. Clone repository

```bash
git clone <repo-url> ~/Documents/TA-APP/YOLO-INFERENCE
cd ~/Documents/TA-APP/YOLO-INFERENCE
```

### 4. Buat virtual environment

> **Penting:** Gunakan `--system-site-packages` agar `rclpy` dan `cv_bridge` dari ROS 2 bisa
> diakses dari dalam venv.

```bash
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
```

### 5. Install Python dependencies

```bash
pip install --upgrade pip

# Core dependencies
pip install \
    numpy"<2" \
    opencv-python \
    Pillow \
    onnx

# ONNX Runtime GPU (TensorRT + CUDA EP) untuk Jetson JetPack 6
# File .whl biasanya tersedia di folder project atau bisa download dari NVIDIA
# Cek folder Downloads atau teman yang sudah mempunyainya:
pip install onnxruntime_gpu-*.whl   # ganti * dengan nama file yang sesuai
# Atau jika tersedia di PyPI untuk aarch64:
# pip install onnxruntime-gpu

# CuPy untuk GPU-accelerated BEV rendering (CUDA 12.x)
pip install cupy-cuda12x

# Ultralytics (untuk export model, opsional untuk inference)
pip install ultralytics
```

> **Verifikasi ONNX Runtime:**
> ```bash
> python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"
> # Harus muncul: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
> ```

### 6. Fix CuPy untuk Jetson Orin (SM_87)

Jetson Orin menggunakan GPU compute capability SM_87. Setelah install `cupy-cuda12x`,
kernel cache perlu dibuild untuk arch ini:

```bash
# Bersihkan cache lama jika ada
rm -rf ~/.cupy/

# Build kernel cache untuk SM_87 (hanya sekali, ~30 detik)
CUPY_CUDA_ARCH_LIST="8.7" python3 -c "
import cupy as cp
a = cp.zeros((100, 100), dtype=cp.float32)
print('CuPy GPU OK:', cp.__version__)
"
```

> Kalau Jetson kamu bukan Orin (Xavier = SM_72, Nano original = SM_53), sesuaikan nilai arch:
> - Xavier AGX/NX: `CUPY_CUDA_ARCH_LIST="7.2"`
> - Jetson Nano (Maxwell): `CUPY_CUDA_ARCH_LIST="5.3"`

> **Untuk sesi selanjutnya:** `CUPY_CUDA_ARCH_LIST` tidak perlu di-set ulang — cache sudah tersimpan.

### 7. Source ROS 2 sebelum menjalankan aplikasi

```bash
source /opt/ros/humble/setup.bash
source .venv/bin/activate
```

Tambahkan ke `~/.bashrc` kalau ingin otomatis:
```bash
echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc
```

### 8. Verifikasi setup lengkap

```bash
source /opt/ros/humble/setup.bash
source .venv/bin/activate

python3 -c "
import onnxruntime as ort
print('ONNX providers:', ort.get_available_providers())

import cupy as cp
a = cp.zeros((10,10), dtype=cp.float32)
print('CuPy GPU:', cp.__version__)

import rclpy
print('ROS 2 rclpy: OK')

from app.core.services.bev_projection import _CUPY_OK
print('BEV GPU flag:', _CUPY_OK)
"
```

Output yang diharapkan:
```
ONNX providers: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
[BEV] CuPy 14.1.1 — GPU BEV rendering aktif (CUDA device 0)
CuPy GPU: 14.1.1
ROS 2 rclpy: OK
BEV GPU flag: True
```

---

## Windows Setup

### Persyaratan Sistem

| Komponen | Versi |
|---|---|
| OS | Windows 10 / 11 (64-bit) |
| Python | 3.11 atau 3.12 |
| GPU (opsional) | NVIDIA dengan driver ≥ 525 untuk CUDA EP |

> **ROS 2 di Windows:** Opsional. Tanpa ROS 2, hanya mode **Stereo Dataset Sim**,
> Video File, Webcam, dan Image File yang tersedia. Mode **ROS 2 Bag** membutuhkan ROS 2 Humble for Windows.

### 1. Install Python

Download Python 3.11 atau 3.12 dari [python.org](https://www.python.org/downloads/).
Pastikan centang **"Add Python to PATH"** saat instalasi.

```powershell
python --version  # verifikasi
```

### 2. Clone repository

```powershell
git clone <repo-url> C:\Projects\YOLO-INFERENCE
cd C:\Projects\YOLO-INFERENCE
```

### 3. Buat virtual environment

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 4. Install Python dependencies

```powershell
pip install --upgrade pip

# Core dependencies (sama dengan Jetson kecuali ONNX Runtime)
pip install numpy opencv-python Pillow onnx

# ONNX Runtime — pilih salah satu sesuai GPU:
# A) Punya NVIDIA GPU + CUDA Toolkit 12.x → GPU inference
pip install onnxruntime-gpu

# B) Tidak punya GPU atau AMD/Intel GPU → CPU inference (lebih lambat, tetap jalan)
pip install onnxruntime

# C) AMD / Intel / NVIDIA via DirectX 12 (Windows GPU universal) — lihat catatan di bawah
pip install onnxruntime-directml
```

> **CuPy di Windows (opsional):**
> BEV rendering akan otomatis fallback ke CPU kalau CuPy tidak terinstall.
> Kalau punya NVIDIA GPU + CUDA 12.x:
> ```powershell
> pip install cupy-cuda12x
> ```

```powershell
# Ultralytics (untuk export model, opsional untuk inference)
pip install ultralytics
```

### 5. Perubahan kode yang diperlukan untuk Windows

> **Untuk AI agent:** bagian ini daftar file dan baris yang perlu diadaptasi untuk Windows.

#### 5a. `gui_main.py` — Mode Fullscreen

**File:** [gui_main.py](gui_main.py) baris 113

```python
# JETSON (current):
master.attributes('-fullscreen', True)

# WINDOWS — ubah ke windowed mode saat development, atau biarkan fullscreen:
# master.attributes('-fullscreen', True)   # tetap fullscreen, tekan Escape untuk keluar
# ATAU ganti ke ukuran window:
master.geometry("1280x800")
```

#### 5b. `object_detection.py` — Tambah DirectML Provider (opsional, untuk GPU AMD/Intel)

**File:** [app/core/services/object_detection.py](app/core/services/object_detection.py) fungsi `_build_providers()` baris ~258

Kode saat ini hanya mencoba `TensorrtExecutionProvider` dan `CUDAExecutionProvider`.
Untuk Windows dengan GPU AMD atau Intel (via DirectX 12), tambah `DmlExecutionProvider`:

```python
# Tambahkan ke list gpu_providers, SEBELUM TensorrtExecutionProvider:
gpu_providers = [
    (
        "TensorrtExecutionProvider",   # Jetson / Linux NVIDIA only
        { ... }
    ),
    (
        "CUDAExecutionProvider",       # NVIDIA GPU + CUDA Toolkit
        { ... }
    ),
    # TAMBAHKAN INI untuk Windows GPU universal (AMD/Intel/NVIDIA via DirectX 12):
    "DmlExecutionProvider",
]
```

Atau, agar lebih aman dan tidak merusak setup Jetson, tambahkan pengecekan OS:

```python
import platform
if platform.system() == "Windows" and "DmlExecutionProvider" in available:
    providers.append("DmlExecutionProvider")
    use_gpu = True
    print("[ONNXRuntime] Using DmlExecutionProvider (DirectML, Windows GPU).")
```

#### 5c. `object_detection.py` — TRT Cache Path

**File:** [app/core/services/object_detection.py](app/core/services/object_detection.py) baris ~255

```python
TRT_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "app", "trt_cache")
```

Path ini menggunakan `os.path.join` sehingga **otomatis kompatibel Windows** (`\` vs `/`).
Tidak perlu diubah.

#### 5d. `bev_projection.py` — CuPy Fallback

**File:** [app/core/services/bev_projection.py](app/core/services/bev_projection.py) baris ~27

Kode sudah memiliki fallback otomatis:
```python
try:
    import cupy as cp
    _test = cp.zeros(1, dtype=cp.float32)
    _CUPY_OK = True   # GPU BEV aktif
except Exception:
    _CUPY_OK = False  # CPU BEV fallback (lebih lambat, tetap jalan)
```

Tidak perlu diubah. Kalau CuPy tidak terinstall, BEV otomatis pakai CPU dengan `downsample=3`.

#### 5e. `ros2_perception_node.py` — Sudah Graceful Degradasi

**File:** [app/core/services/ros2_perception_node.py](app/core/services/ros2_perception_node.py) baris ~37

```python
_ROS2_AVAILABLE = False
try:
    import rclpy
    ...
    _ROS2_AVAILABLE = True
except ImportError:
    pass  # GUI tetap jalan, mode ROS 2 Bag nonaktif di UI
```

Tidak perlu diubah. Tanpa ROS 2, tombol "ROS 2 Bag" di GUI tidak akan berfungsi
(akan muncul pesan error di status bar).

#### 5f. Script Bash — Perlu WSL atau Adaptasi Manual

File-file ini hanya berjalan di Linux/WSL:
- `scripts/kitti_download_subset.sh` → download dataset KITTI
- `scripts/remap_kitti_structure.sh` → remap folder structure

**Alternatif di Windows:**
1. Gunakan **WSL2** (Windows Subsystem for Linux) untuk menjalankan script bash
2. Atau download dan atur folder KITTI secara manual (lihat [Dataset KITTI](#dataset-kitti))

Script Python berikut **sudah berjalan di Windows** tanpa perubahan:
- `scripts/generate_stereo_disparity.py`
- `scripts/export_baseline.py`

### 6. Verifikasi setup Windows

```powershell
.venv\Scripts\activate

python -c "
import onnxruntime as ort
print('ONNX providers:', ort.get_available_providers())

try:
    import cupy as cp
    a = cp.zeros((10,10), dtype=cp.float32)
    print('CuPy GPU:', cp.__version__)
except Exception as e:
    print('CuPy: tidak tersedia, BEV akan pakai CPU —', e)

from app.core.services.bev_projection import _CUPY_OK
print('BEV GPU flag:', _CUPY_OK)
"
```

---

## Dataset KITTI

### Struktur Folder yang Dibutuhkan

Mode **Stereo Dataset Sim** membutuhkan folder dengan struktur ini:

```
<sequence-folder>/
├── left_images/
│   ├── 0000000000.png
│   ├── 0000000001.png
│   └── ...
├── disparity/
│   ├── 0000000000.png   ← 16-bit disparity (pixel = disparity × 256)
│   ├── 0000000001.png
│   └── ...
└── timestamps.txt       ← (hanya diperlukan untuk konversi ke ROS 2 bag)
```

### Setup di Jetson (lengkap dengan ROS 2 bag)

```bash
# 1. Buat direktori data
sudo mkdir -p /data/kitti/{raw,sequences,calib,rosbags}
sudo chown -R $USER:$USER /data/kitti/

# 2. Download subset KITTI (3 drive, ~800MB)
bash scripts/kitti_download_subset.sh

# 3. Remap struktur folder
bash scripts/remap_kitti_structure.sh

# 4. Generate disparity sinkron dari stereo pair (image_02 + image_03)
# Ganti path sesuai drive yang sudah didownload
source .venv/bin/activate
python3 scripts/generate_stereo_disparity.py \
    /data/kitti/raw/2011_09_26/2011_09_26_drive_0001_sync \
    /data/kitti/sequences/0001/disparity

# 5. Konversi ke ROS 2 bag (opsional, untuk mode ROS 2 Bag di GUI)
source /opt/ros/humble/setup.bash
python3 scripts/kitti_to_rosbag.py \
    /data/kitti/sequences/0001 \
    /data/kitti/calib/2011_09_26/calib_cam_to_cam.txt \
    /data/kitti/rosbags/kitti_0001
```

### Setup di Windows (Stereo Dataset Sim saja)

Tanpa ROS 2, langkah 5 (konversi bag) tidak diperlukan.

**Opsi A — Download otomatis via WSL2:**
```bash
# Di WSL2 terminal
bash scripts/kitti_download_subset.sh
bash scripts/remap_kitti_structure.sh
python3 scripts/generate_stereo_disparity.py \
    /data/kitti/raw/2011_09_26/2011_09_26_drive_0001_sync \
    /mnt/c/Projects/kitti_sequences/0001/disparity
```

**Opsi B — Download manual:**
1. Download KITTI raw data dari [www.cvlibs.net/datasets/kitti](http://www.cvlibs.net/datasets/kitti/raw_data.php)
2. Ekstrak `image_02/data/` sebagai `left_images/`
3. Jalankan `generate_stereo_disparity.py` dari PowerShell:
   ```powershell
   .venv\Scripts\activate
   python scripts/generate_stereo_disparity.py `
       C:\data\kitti\raw\2011_09_26\2011_09_26_drive_0001_sync `
       C:\data\kitti\sequences\0001\disparity
   ```

**Opsi C — Pakai DrivingStereo dataset:**
Dataset [DrivingStereo](https://drivingstereo-dataset.github.io/) sudah dalam format
`left_images/` + `disparity/` dan kompatibel langsung dengan `StereoSimLoader`.
Intrinsics: `fx=721.53`, `baseline=0.54m`.

---

## Menjalankan Aplikasi

### Jetson

```bash
# Setiap sesi baru:
source /opt/ros/humble/setup.bash
source .venv/bin/activate

# Jalankan
python3 gui_main.py
```

Di GUI, tekan **Escape** untuk keluar fullscreen.

**Mode Stereo Dataset Sim:**
1. Pilih model di dropdown (auto-discovered dari `app/models/`)
2. Pilih mode **Stereo Dataset Sim**
3. Klik **Browse** → pilih folder sequence (misal `/data/kitti/sequences/0001`)
4. Aktifkan **Path Planning** checkbox
5. Klik **Start**
6. Gunakan tombol navigasi **[RGB Video] [Depth Map] [BEV] [OGM]** untuk ganti view

**Mode ROS 2 Bag (Jetson only):**
1. Pilih model
2. Pilih mode **ROS 2 Bag**
3. Klik **Browse Bag** → pilih folder bag (misal `/data/kitti/rosbags/kitti_0001`)
4. Klik **Start** → GUI otomatis menjalankan `ros2 bag play --loop`

### Windows

```powershell
.venv\Scripts\activate
python gui_main.py
```

Gunakan mode **Stereo Dataset Sim** atau **Video File**. Mode **ROS 2 Bag** membutuhkan
ROS 2 Humble for Windows (opsional, setup lebih kompleks).

---

## Perbedaan Jetson vs Windows (Ringkasan)

| Aspek | Jetson (JetPack 6) | Windows |
|---|---|---|
| **Python** | 3.10 (system) | 3.11 / 3.12 |
| **ONNX Runtime** | `onnxruntime-gpu` (TRT + CUDA EP) | `onnxruntime-gpu` (CUDA EP) atau `onnxruntime` (CPU) |
| **GPU Backend Inferensi** | TensorrtExecutionProvider → CUDAExecutionProvider | CUDAExecutionProvider → DmlExecutionProvider → CPU |
| **BEV GPU** | CuPy 14.1.1 (SM_87, CUDA 12.2) | CuPy (jika NVIDIA GPU) atau CPU fallback |
| **ROS 2** | Humble (full support) | Opsional (Humble for Windows) |
| **Mode Bag** | ✅ Tersedia | Perlu ROS 2 for Windows |
| **CuPy CUDA arch** | `CUPY_CUDA_ARCH_LIST="8.7"` (Orin) | Otomatis (jika terinstall) |
| **venv flag** | `--system-site-packages` (untuk rclpy) | Standar (tidak perlu flag) |
| **Script bash** | Langsung jalan | Perlu WSL2 |
| **Fullscreen** | Default fullscreen | Bisa ganti ke windowed |

### File yang Tidak Perlu Diubah

Kode berikut sudah cross-platform karena menggunakan `os.path.join` dan graceful degradasi:
- `object_detection.py` — auto-detect GPU provider
- `bev_projection.py` — CuPy optional, CPU fallback otomatis
- `ros2_perception_node.py` — ROS 2 optional, `_ROS2_AVAILABLE = False` jika tidak ada
- `path_planning.py` — pure numpy, tidak ada dependency platform
- `generate_stereo_disparity.py` — pure OpenCV + numpy

### File yang Perlu Diperhatikan di Windows

| File | Baris | Perubahan |
|---|---|---|
| `gui_main.py` | 113 | Fullscreen → windowed kalau perlu |
| `object_detection.py` | ~258 | Tambah `DmlExecutionProvider` jika AMD/Intel GPU |
| `scripts/*.sh` | — | Jalankan via WSL2 atau lakukan manual |

---

## Troubleshooting

### [Jetson] `CUDA_ERROR_INVALID_IMAGE` saat pertama import CuPy

CuPy cache dari arch yang salah. Fix:
```bash
rm -rf ~/.cupy/
CUPY_CUDA_ARCH_LIST="8.7" python3 -c "import cupy as cp; print(cp.zeros(1))"
```

### [Jetson] `ROS 2 tidak tersedia` di GUI padahal sudah source

Venv dibuat tanpa `--system-site-packages`. Buat ulang:
```bash
deactivate
rm -rf .venv
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
pip install ...  # install ulang dependencies
```

### [Jetson] TRT compile 20+ menit

Normal hanya saat **pertama kali** model diload — TRT mem-build engine dan menyimpan cache
di `app/trt_cache/`. Setelah itu load <5 detik. Pastikan folder `app/trt_cache/` ada dan writable.

### [Windows] `No module named 'cv2'`

```powershell
pip install opencv-python
```

### [Windows] `No module named 'onnxruntime'`

```powershell
pip install onnxruntime          # CPU
# atau
pip install onnxruntime-gpu      # NVIDIA GPU + CUDA
```

### [Kedua platform] `No .onnx files found in app/models/`

Pastikan file `app/models/baseline.onnx` ada. Kalau belum ada, export dari `.pt`:
```bash
source .venv/bin/activate
python3 scripts/export_baseline.py
```

### [Kedua platform] Disparity folder kosong

Jalankan `generate_stereo_disparity.py` untuk menghasilkan file disparity dari stereo pair:
```bash
python3 scripts/generate_stereo_disparity.py \
    /path/to/kitti/drive_sync \
    /path/to/output/disparity
```

### [Kedua platform] BEV menampilkan layar hitam

Pastikan:
1. Depth map berisi nilai valid (bukan semua 0). Cek dengan:
   ```python
   import cv2, numpy as np
   d = cv2.imread("disparity/0000000000.png", cv2.IMREAD_UNCHANGED)
   print("Max disparity:", d.max(), "| Non-zero:", np.count_nonzero(d))
   ```
2. Mode yang dipilih menyediakan depth (Stereo Dataset Sim atau ROS 2 Bag).
3. `intrinsics` ter-set dengan benar (`fx > 0`).
