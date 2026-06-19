```markdown
# Panduan uv & Virtual Environment (Jetson)

Panduan cepat untuk manajemen Python dan *virtual environment* menggunakan `uv` di project ini.

## 1. Persiapan Terminal
Jika command `uv` belum dikenali saat membuka terminal baru, jalankan perintah ini untuk mendaftarkan path:
```bash
export PATH="$HOME/.local/bin:$PATH"

```

## 2. Membuat Virtual Environment (.venv)

Masuk ke direktori project dan buat *environment* terisolasi dengan Python 3.12:

```bash
cd /home/skripsi2025/Documents/TA-APP/YOLO-INFERENCE
uv venv --python 3.12 .venv

```

## 3. Keluar-Masuk Virtual Environment

**Masuk / Aktifkan Venv:**
Gunakan command ini setiap kali akan mulai bekerja. Jika berhasil, akan muncul indikator `(.venv)` di awal baris terminal.

```bash
source .venv/bin/activate

```

**Keluar / Nonaktifkan Venv:**
Gunakan command ini untuk keluar dari venv dan kembali menggunakan Python global bawaan OS.

```bash
deactivate

```

## 4. Instalasi Dependencies (Aman untuk Jetson ARM64)

⚠️ **Penting:** Pastikan venv sudah dalam keadaan **aktif** (`.venv` muncul di terminal). Paket `onnxruntime-directml` bawaan repositori Windows harus dihapus karena tidak mendukung Linux ARM64. Gunakan `uv pip` untuk instalasi super cepat.

```bash
uv pip install opencv-python numpy pillow ultralytics onnx onnxslim onnxruntime

```

## 5. Command Penting uv pip

* **Install satu atau beberapa package:**
```bash
uv pip install nama_package

```


* **Install package dari file requirements:**
```bash
uv pip install -r requirements.txt

```


* **Melihat daftar package yang sudah terinstal:**
```bash
uv pip list

```


* **Menyimpan daftar package saat ini ke file (Export):**
```bash
uv pip freeze > requirements.txt

```


* **Menghapus package:**
```bash
uv pip uninstall nama_package

```



## 6. Menjalankan Aplikasi

Setelah semua dependensi terinstal, jalankan skrip utama program:

```bash
python gui_main.py

```

```

```