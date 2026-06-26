# File: gui_main.py

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import subprocess
import cv2
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import random
import numpy as np

from app.core.services.object_detection import (
    get_combined_prediction_from_frame,
    perform_road_analysis,
    draw_main_visualization,
    set_active_model,
    discover_models
)
from app.core.services.bev_projection import (
    backproject_detections,
    render_bev_opencv,
    render_dense_bev,
    intrinsics_from_calib,
    intrinsics_from_frame_width,
)


class StereoSimLoader:
    """Ingest stereo dataset pairs (left RGB + 32-bit float metric depth .npy) and load depth maps."""

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.left_dir = os.path.join(root_dir, "left_images")
        self.disp_dir = os.path.join(root_dir, "depth_npy")
        self._frame_list = []
        if os.path.isdir(self.left_dir) and os.path.isdir(self.disp_dir):
            left_files = sorted(f for f in os.listdir(self.left_dir) if f.lower().endswith(".png"))
            for fname in left_files:
                base_name = os.path.splitext(fname)[0]
                npy_name = f"{base_name}.npy"
                if os.path.exists(os.path.join(self.disp_dir, npy_name)):
                    self._frame_list.append(fname)

    def is_opened(self) -> bool:
        return len(self._frame_list) > 0

    def read_frame(self, index: int):
        """Read the RGB frame and load metric depth for the given sequential index.

        Returns:
            (success: bool, frame_bgr: np.ndarray | None, depth_map_m: np.ndarray | None)
        """
        if index < 0 or index >= len(self._frame_list):
            return False, None, None

        fname = self._frame_list[index]
        left_path = os.path.join(self.left_dir, fname)
        base_name = os.path.splitext(fname)[0]
        disp_path = os.path.join(self.disp_dir, f"{base_name}.npy")

        frame_bgr = cv2.imread(left_path, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            return False, None, None

        try:
            # Load pre-calculated 32-bit float metric depth map (ZED 2i simulation)
            depth_m = np.load(disp_path).astype(np.float32)
        except Exception as e:
            print(f"[StereoSimLoader Error] Failed to load depth map: {e}")
            return False, None, None

        # Clip to operational range
        depth_m = np.clip(depth_m, 0.5, 40.0)

        # Downscale to max 640px width — reduces array payload ~85%
        # Dynamic fx in path_planning.py auto-adapts to new orig_w
        target_w = 640
        ratio = target_w / frame_bgr.shape[1]
        target_h = int(frame_bgr.shape[0] * ratio)
        frame_bgr = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        depth_m = cv2.resize(depth_m, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        return True, frame_bgr, depth_m

THEMES = {
    "dark":  {"BG": "#212121", "FRAME": "#2E2E2E", "TEXT": "#FFFFFF", "ACCENT": "#00A0A0", "CANVAS": "black",   "BTN_FG": "white", "STATUS": "#2E2E2E", "PLACEHOLDER": "#757575"},
    "light": {"BG": "#F5F5F5", "FRAME": "#FFFFFF", "TEXT": "#000000", "ACCENT": "#00796B", "CANVAS": "#BDBDBD", "BTN_FG": "white", "STATUS": "#E0E0E0", "PLACEHOLDER": "#616161"}
}
FONTS = {
    "normal":      ("Segoe UI", 9),
    "bold":        ("Segoe UI", 9, "bold"),
    "title":       ("Segoe UI", 10, "bold"),
    "value":       ("Segoe UI", 14, "bold"),
    "placeholder": ("Segoe UI", 11, "italic")
}


class DashboardGUI:
    def __init__(self, master):
        self.master = master
        master.title("Road Analysis Dashboard")
        master.minsize(800, 480)
        master.attributes('-fullscreen', True)

        self.is_dark_mode = True
        self.processing_thread = None
        self.stop_processing_flag = threading.Event()
        self.fps_queue = deque(maxlen=30)
        self.show_polygon = tk.BooleanVar(value=True)
        self.show_polygon_enabled = True
        self.source_path = None
        self._active_mode = None
        self._active_model_name = None
        self._canvas_image_id = None
        self._canvas_update_pending = False
        self._latest_canvas_frame = None
        self.mc_dropout_var = tk.BooleanVar(value=False)
        self.show_uncertainty_var = tk.BooleanVar(value=True)
        self.mc_n_samples_var = tk.IntVar(value=5)
        self.mc_pt_path = None
        self.path_planning_var = tk.BooleanVar(value=False)
        self._frame_img_w = None

        # ROS 2 Bag state
        self._ros_runner = None
        self._latest_calib = None
        self._latest_depth_map = None
        self._bag_process = None
        self._bag_path = None

        # View state — satu canvas besar, 4 view yang bisa diswitch
        self._active_view = "rgb"          # "rgb" | "depth" | "bev" | "ogm"
        self._nav_buttons = {}

        # Stored latest frames untuk setiap view (BGR numpy array)
        self._latest_rgb_frame   = None
        self._latest_depth_frame = None
        self._latest_bev_frame   = None
        self._latest_ogm_frame   = None

        # Pending flags per view
        self._canvas_update_pending = False
        self._depth_update_pending  = False
        self._bev_update_pending    = False
        self._ogm_update_pending    = False

        # BEV/OGM source data
        self._latest_bev_detections = None
        self._latest_path_data      = None
        self._latest_ogm_grid       = None
        self._latest_ogm_path       = None

        # Optimasi A: Frame skip — BEV + path planning tidak diupdate setiap frame
        self._frame_counter       = 0
        self._bev_skip_interval   = 2   # GPU: setiap frame OK; CPU: skip 1 dari 2
        self._path_skip_interval  = 2   # path planning (A*) setiap 2 frame

        # Optimasi B: Thread pool untuk render BEV dan OGM
        # max_workers=2: satu untuk BEV render, satu untuk OGM render
        self._render_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="render")

        # --- AUTO-DISCOVERY ---
        base_path = os.path.dirname(__file__)
        models_dir = os.path.join(base_path, "app", "models")
        self.model_configs = discover_models(models_dir)

        if not self.model_configs:
            messagebox.showerror(
                "No Models Found",
                f"No .onnx files found in:\n{models_dir}\n\nPlease add model files and restart."
            )
            master.destroy()
            return

        self._setup_style()
        self._create_widgets()
        self._apply_theme()
        self._update_dummy_data()
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.master.bind("<Escape>", lambda e: self.master.attributes("-fullscreen", False))

    # -------------------------------------------------------------------------

    def _setup_style(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')

    def _apply_theme(self):
        colors = THEMES["dark"] if self.is_dark_mode else THEMES["light"]
        self.master.configure(bg=colors["BG"])
        self.style.configure("TFrame",            background=colors["BG"])
        self.style.configure("TLabel",            background=colors["BG"],    foreground=colors["TEXT"],       font=FONTS["normal"])
        self.style.configure("Card.TFrame",       background=colors["FRAME"])
        self.style.configure("Header.TLabel",     background=colors["FRAME"], foreground=colors["ACCENT"],     font=FONTS["bold"])
        self.style.configure("Value.TLabel",      background=colors["FRAME"], foreground=colors["TEXT"],       font=FONTS["value"])
        self.style.configure("TButton",           background=colors["ACCENT"],foreground=colors["BTN_FG"],     font=FONTS["bold"], borderwidth=0)
        self.style.map("TButton",                 background=[('active', colors["ACCENT"])])
        self.style.configure("TCombobox",         fieldbackground=colors["FRAME"], background=colors["FRAME"],
                                                  foreground=colors["TEXT"], arrowcolor=colors["TEXT"],
                                                  selectbackground=colors["FRAME"], selectforeground=colors["TEXT"],
                                                  font=FONTS["normal"])
        self.style.configure("TCheckbutton",      background=colors["BG"],    foreground=colors["TEXT"],       font=FONTS["normal"])
        self.style.map("TCheckbutton",            indicatorcolor=[('selected', colors["ACCENT"])])
        self.style.configure("Status.TLabel",     background=colors["STATUS"],foreground=colors["TEXT"],       font=FONTS["normal"], padding=5)
        self.style.configure("Placeholder.TLabel",background=colors["FRAME"], foreground=colors["PLACEHOLDER"],font=FONTS["placeholder"], anchor="center")
        if hasattr(self, 'main_canvas'):
            self.main_canvas.config(bg=colors["CANVAS"])
        for btn_key, btn in self._nav_buttons.items():
            is_active = (btn_key == self._active_view)
            bg = colors["ACCENT"] if is_active else colors["FRAME"]
            btn.config(bg=bg, fg=colors["BTN_FG"] if is_active else colors["TEXT"])

    # -------------------------------------------------------------------------

    def _create_widgets(self):
        self.master.columnconfigure(0, weight=7)   # display besar
        self.master.columnconfigure(1, weight=2, minsize=240)  # control panel
        self.master.rowconfigure(0, weight=1)

        # Column 0: Display utama (RGB / Depth / BEV / OGM)
        main_display_frame = ttk.Frame(self.master)
        main_display_frame.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        self._create_main_display(main_display_frame)

        # Column 1: Control Panel
        control_panel_frame = ttk.Frame(self.master)
        control_panel_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
        self._create_control_panel(control_panel_frame)

    def _create_main_display(self, parent):
        parent.rowconfigure(0, weight=0)  # nav bar
        parent.rowconfigure(1, weight=1)  # canvas
        parent.rowconfigure(2, weight=0)  # footer
        parent.columnconfigure(0, weight=1)

        # ── Nav bar ──────────────────────────────────────────────────────────
        nav_frame = ttk.Frame(parent, style="Card.TFrame")
        nav_frame.grid(row=0, column=0, sticky="ew", pady=(0, 4))

        views = [
            ("rgb",   "RGB Video"),
            ("depth", "Depth Map"),
            ("bev",   "BEV"),
            ("ogm",   "OGM Grid"),
        ]
        for view_key, label in views:
            btn = tk.Button(
                nav_frame, text=label, font=FONTS["bold"],
                relief="flat", padx=12, pady=6, cursor="hand2",
                command=lambda k=view_key: self._switch_view(k)
            )
            btn.pack(side=tk.LEFT, padx=2, pady=4)
            self._nav_buttons[view_key] = btn

        # ── Canvas utama ─────────────────────────────────────────────────────
        self.main_canvas = tk.Canvas(parent, highlightthickness=0)
        self.main_canvas.grid(row=1, column=0, sticky="nsew")

        # ── Footer ───────────────────────────────────────────────────────────
        footer_frame = ttk.Frame(parent, style="Status.TLabel")
        footer_frame.grid(row=2, column=0, sticky="ew")
        footer_frame.columnconfigure(0, weight=1)

        self.show_polygon_check = ttk.Checkbutton(
            footer_frame, text="Show Polygon",
            variable=self.show_polygon, style="TCheckbutton",
            command=self._on_show_polygon_changed
        )
        self.show_polygon_check.pack(side=tk.LEFT, padx=10)

        self.status_label = ttk.Label(footer_frame, text="FPS: - | Latency: - ms",
                                       style="Status.TLabel")
        self.status_label.pack(side=tk.RIGHT, padx=10)

        # Highlight tombol aktif
        self._refresh_nav_buttons()

    def _refresh_nav_buttons(self):
        """Update warna tombol nav sesuai active view."""
        colors = THEMES["dark"] if self.is_dark_mode else THEMES["light"]
        for key, btn in self._nav_buttons.items():
            if key == self._active_view:
                btn.config(bg=colors["ACCENT"], fg=colors["BTN_FG"])
            else:
                btn.config(bg=colors["FRAME"], fg=colors["TEXT"])

    def _switch_view(self, view_key: str):
        """Ganti view aktif dan tampilkan frame terakhir yang tersimpan."""
        self._active_view = view_key
        self._refresh_nav_buttons()
        frame_map = {
            "rgb":   self._latest_rgb_frame,
            "depth": self._latest_depth_frame,
            "bev":   self._latest_bev_frame,
            "ogm":   self._latest_ogm_frame,
        }
        frame = frame_map.get(view_key)
        if frame is not None:
            self._update_canvas(self.main_canvas, frame)

    def _create_control_panel(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=0)
        parent.rowconfigure(1, weight=0)
        parent.rowconfigure(2, weight=1)

        # --- Controls card ---
        controls_frame = ttk.Frame(parent, style="Card.TFrame", padding=10)
        controls_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        model_names = list(self.model_configs.keys())
        self.model_combo = ttk.Combobox(controls_frame, values=model_names, state="readonly", width=20)
        self.model_combo.set(model_names[0])
        self.model_combo.pack(pady=4, fill=tk.X)

        # Refresh button — rescans models folder without restarting
        ttk.Button(controls_frame, text="↻ Refresh Models", command=self._refresh_models).pack(pady=2, fill=tk.X)

        self.mode_combo = ttk.Combobox(
            controls_frame,
            values=["Select Source...", "Image File", "Video File", "Webcam",
                    "Stereo Dataset Sim", "ROS 2 Bag"],
            state="readonly", width=20
        )
        self.mode_combo.set("Select Source...")
        self.mode_combo.pack(pady=4, fill=tk.X)
        self.mode_combo.bind("<<ComboboxSelected>>", self.on_mode_change)

        # Label nama bag yang dipilih (hanya tampil saat mode ROS 2 Bag)
        self._bag_label_var = tk.StringVar(value="")
        self.bag_label = ttk.Label(controls_frame, textvariable=self._bag_label_var,
                                   style="Header.TLabel", wraplength=200, anchor="w")
        self.bag_label.pack(fill=tk.X)
        self.bag_label.pack_forget()  # sembunyikan dulu

        self.process_button = ttk.Button(controls_frame, text="Start Processing", command=self.toggle_processing)
        self.process_button.pack(pady=4, fill=tk.X, ipady=5)

        ttk.Button(controls_frame, text="Toggle Theme", command=self._toggle_theme).pack(pady=4, fill=tk.X, ipady=5)

        ttk.Separator(controls_frame, orient="horizontal").pack(fill=tk.X, pady=8)
        ttk.Label(controls_frame, text="MC Dropout", style="Header.TLabel").pack(anchor="w")

        self.mc_pt_label = ttk.Label(
            controls_frame, text="No .pt loaded",
            style="Header.TLabel", foreground="gray"
        )
        self.mc_pt_label.pack(fill=tk.X)
        ttk.Button(controls_frame, text="Load .pt Model",
                   command=self._load_pt_model).pack(fill=tk.X, pady=2)

        ttk.Checkbutton(controls_frame, text="Enable MC Dropout",
                        variable=self.mc_dropout_var,
                        style="TCheckbutton").pack(anchor="w")

        ttk.Checkbutton(controls_frame, text="Show Uncertainty Overlay",
                        variable=self.show_uncertainty_var,
                        style="TCheckbutton").pack(anchor="w")

        self.mc_samples_label = ttk.Label(
            controls_frame, text="N Samples: 5",
            style="Header.TLabel"
        )
        self.mc_samples_label.pack(anchor="w")
        ttk.Scale(controls_frame, from_=1, to=10,
                  variable=self.mc_n_samples_var, orient="horizontal",
                  command=self._on_n_samples_change).pack(fill=tk.X)

        ttk.Separator(controls_frame, orient="horizontal").pack(fill=tk.X, pady=8)
        ttk.Label(controls_frame, text="Path Planning", style="Header.TLabel").pack(anchor="w")

        ttk.Checkbutton(controls_frame, text="Enable Path Planning (Stereo)",
                        variable=self.path_planning_var,
                        style="TCheckbutton").pack(anchor="w")

        # --- Status cards ---
        status_cards_frame = ttk.Frame(parent)
        status_cards_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        status_cards_frame.columnconfigure(0, weight=1)
        self._create_status_cards_panel(status_cards_frame)



    def _create_status_cards_panel(self, parent):
        self.speed_label    = self._create_status_card(parent, 0, "SPEED",    "0 Km/h")
        self.battery_label  = self._create_status_card(parent, 1, "BATTERY",  "0 %")
        self.location_label = self._create_status_card(parent, 2, "LOCATION", "0.0, 0.0")
        self.weather_label  = self._create_status_card(parent, 3, "WEATHER",  "N/A")

    def _create_status_card(self, parent, row_index, title, initial_value):
        card = ttk.Frame(parent, style="Card.TFrame", padding=(10, 5))
        card.grid(row=row_index, column=0, sticky="ew", pady=2)
        ttk.Label(card, text=title, style="Header.TLabel").pack(anchor="w")
        value_label = ttk.Label(card, text=initial_value, style="Value.TLabel")
        value_label.pack(anchor="w")
        return value_label

    # -------------------------------------------------------------------------

    def _toggle_theme(self):
        self.is_dark_mode = not self.is_dark_mode
        self._apply_theme()

    def _on_show_polygon_changed(self):
        self.show_polygon_enabled = bool(self.show_polygon.get())

    def _load_pt_model(self):
        path = filedialog.askopenfilename(
            filetypes=[("PyTorch Model", "*.pt")]
        )
        if path:
            self.mc_pt_path = path
            name = os.path.basename(path)
            self.mc_pt_label.config(text=name, foreground="green")

    def _on_n_samples_change(self, val):
        self.mc_samples_label.config(text=f"N Samples: {int(float(val))}")

    def _refresh_models(self):
        """Rescan models folder and update the dropdown — no restart needed."""
        base_path = os.path.dirname(__file__)
        models_dir = os.path.join(base_path, "app", "models")
        self.model_configs = discover_models(models_dir)

        model_names = list(self.model_configs.keys())
        self.model_combo.config(values=model_names)

        if model_names:
            self.model_combo.set(model_names[0])
            messagebox.showinfo("Models Refreshed", f"Found {len(model_names)} model(s):\n" + "\n".join(model_names))
        else:
            self.model_combo.set("")
            messagebox.showwarning("No Models", f"No .onnx files found in:\n{models_dir}")

    # -------------------------------------------------------------------------

    def on_mode_change(self, event=None):
        mode = self.mode_combo.get()
        if mode in ["Image File", "Video File"]:
            self.source_path = filedialog.askopenfilename()
            if not self.source_path:
                self.mode_combo.set("Select Source...")
        elif mode == "Webcam":
            self.source_path = 0
        elif mode == "Stereo Dataset Sim":
            self.source_path = filedialog.askdirectory(title="Select Stereo Dataset Root (with left_images/ and disparity/ subfolders)")
            if not self.source_path:
                self.mode_combo.set("Select Source...")
        elif mode == "ROS 2 Bag":
            bag_path = filedialog.askdirectory(
                title="Pilih folder ROS 2 Bag (berisi .db3 atau .mcap)"
            )
            if not bag_path:
                self.mode_combo.set("Select Source...")
                self.bag_label.pack_forget()
                return
            self._bag_path = bag_path
            self.source_path = "ros2_bag"
            bag_name = os.path.basename(bag_path.rstrip("/"))
            self._bag_label_var.set(f"Bag: {bag_name}")
            self.bag_label.pack(fill=tk.X)
        else:
            self.bag_label.pack_forget()

    def toggle_processing(self):
        if self.processing_thread and self.processing_thread.is_alive():
            self.stop_processing()
        else:
            self.start_processing()

    def start_processing(self):
        selected_mode = self.mode_combo.get()
        if selected_mode == "Select Source...":
            messagebox.showwarning("Source Not Selected", "Please select an input source.")
            return
        if not self.model_configs:
            messagebox.showerror("No Models", "No models loaded. Drop .onnx files into app/models/ and click ↻ Refresh Models.")
            return
        self._active_mode = selected_mode
        self._active_model_name = self.model_combo.get()
        self._frame_start = None
        self.fps_queue.clear()
        self.stop_processing_flag.clear()
        self.process_button.config(text="Stop Processing")
        self.model_combo.config(state="disabled")
        self.mode_combo.config(state="disabled")
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()

    def _start_bag_play(self, bag_path: str, rate: float = 1.0):
        """Launch ros2 bag play sebagai subprocess. Non-blocking."""
        self._stop_bag_play()  # pastikan tidak ada yang sedang jalan
        cmd = ["ros2", "bag", "play", bag_path, "--loop",
               "--rate", str(rate)]
        try:
            self._bag_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"[BagPlay] Started: {' '.join(cmd)}")
        except FileNotFoundError:
            print("[BagPlay] ERROR: 'ros2' tidak ditemukan di PATH. "
                  "Jalankan: source /opt/ros/humble/setup.bash")
            self._bag_process = None

    def _stop_bag_play(self):
        """Terminate subprocess ros2 bag play jika sedang berjalan."""
        if self._bag_process is not None:
            if self._bag_process.poll() is None:  # masih jalan
                self._bag_process.terminate()
                try:
                    self._bag_process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self._bag_process.kill()
            self._bag_process = None
            print("[BagPlay] Stopped.")

    def stop_processing(self):
        self.stop_processing_flag.set()
        self._stop_bag_play()
        if self._ros_runner is not None:
            try:
                self._ros_runner.stop()
            except Exception as e:
                print(f"[ROSBagRunner] Stop error: {e}")
            self._ros_runner = None
        self.process_button.config(text="Start Processing")
        self.model_combo.config(state="readonly")
        self.mode_combo.config(state="readonly")

    def _sleep_interruptible(self, seconds: float):
        if seconds <= 0:
            return
        self.stop_processing_flag.wait(seconds)

    def _processing_loop(self):
        mode = self._active_mode
        model_name = self._active_model_name
        set_active_model(model_name)
        if self.source_path is None:
            self.master.after(0, lambda: messagebox.showerror("Error", "Source path is not set."))
            self.master.after(0, self.stop_processing)
            return

        # --- Branch: Stereo Dataset Sim | ROS 2 Bag | standard video/image ---
        is_stereo_sim = (mode == "Stereo Dataset Sim")
        is_ros2_bag   = (mode == "ROS 2 Bag")
        stereo_loader = None
        source = None

        if is_stereo_sim:
            stereo_loader = StereoSimLoader(self.source_path)
            if not stereo_loader.is_opened():
                self.master.after(0, lambda: messagebox.showerror(
                    "Error",
                    "No matching left_images/ and disparity/ .png pairs found in the selected folder."
                ))
                self.master.after(0, self.stop_processing)
                return
        elif is_ros2_bag:
            from app.core.services.ros2_perception_node import ROSBagRunner, _ROS2_AVAILABLE
            if not _ROS2_AVAILABLE:
                self.master.after(0, lambda: messagebox.showerror(
                    "ROS 2 Tidak Tersedia",
                    "ROS 2 Python packages tidak ditemukan.\n\n"
                    "Jalankan terlebih dahulu:\n"
                    "    source /opt/ros/humble/setup.bash\n\n"
                    "Lalu restart aplikasi."
                ))
                self.master.after(0, self.stop_processing)
                return
            try:
                self._ros_runner = ROSBagRunner()
                self._ros_runner.start()
                self._start_bag_play(self._bag_path)
            except Exception as e:
                err = str(e)
                self.master.after(0, lambda: messagebox.showerror("ROS 2 Error", err))
                self.master.after(0, self.stop_processing)
                return
        else:
            source = cv2.VideoCapture(self.source_path)
            if mode == "Webcam":
                source.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not source.isOpened():
                self.master.after(0, lambda: messagebox.showerror("Error", "Could not open video/image source."))
                self.master.after(0, self.stop_processing)
                return

        source_fps = 0.0
        frame_interval = 0.0
        if mode == "Video File" and source is not None:
            source_fps = float(source.get(cv2.CAP_PROP_FPS) or 0.0)
            if source_fps != source_fps or source_fps <= 0.0:
                source_fps = 30.0
            frame_interval = 1.0 / source_fps

        start_time = time.perf_counter()
        frames_read = 0
        
        # --- BENCHMARKING VARS ---
        processed_frames_count = 0
        skipped_frames_count = 0
        latency_history = []
        fps_history = []
            
        while not self.stop_processing_flag.is_set():
            # --- Read frame from the appropriate source ---
            simulated_depth = None
            if is_stereo_sim:
                ret, frame, simulated_depth = stereo_loader.read_frame(frames_read)
                if not ret:
                    break
            elif is_ros2_bag:
                payload = self._ros_runner.get_frame(timeout=0.1)
                if payload is None:
                    continue  # Tunggu bag play, jangan break
                frame = payload['frame_bgr']
                simulated_depth = payload['depth_m']
                # Update calib dari CameraInfo (thread-safe karena overwrite atomic)
                if payload['calib'] is not None:
                    self._latest_calib = payload['calib']
                ret = True
            else:
                ret, frame = source.read()
                if not ret:
                    break
                
            frames_read += 1
            processed_frames_count += 1
            latency, frame_fps = self.process_and_display_frame(frame, external_depth=simulated_depth)
            latency_history.append(latency)
            if frame_fps > 0:
                fps_history.append(frame_fps)

            if mode == "Video File" and source is not None:
                target_time = start_time + (frames_read * frame_interval)
                delay = target_time - time.perf_counter()
                if delay > 0:
                    self._sleep_interruptible(delay)
                else:
                    # If inference is slower than the source video, drop unread frames
                    # so preview timing stays aligned to the original video clock.
                    frames_behind = int(abs(delay) / frame_interval) if frame_interval > 0 else 0
                    for _ in range(frames_behind):
                        if self.stop_processing_flag.is_set():
                            break
                        if not source.grab():
                            break
                        frames_read += 1
                        skipped_frames_count += 1
            
            if mode == "Image File":
                break
        
        total_time = time.perf_counter() - start_time
        if source is not None:
            source.release()
        
        if not self.stop_processing_flag.is_set():
            self.master.after(0, self.stop_processing)
            
        # --- SHOW BENCHMARK RESULTS ---
        if mode == "Video File" and processed_frames_count > 0:
            avg_latency = sum(latency_history) / len(latency_history) if latency_history else 0
            max_latency = max(latency_history) if latency_history else 0
            min_latency = min(latency_history) if latency_history else 0
            
            avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0
            max_fps = max(fps_history) if fps_history else 0
            min_fps = min(fps_history) if fps_history else 0
            
            benchmark_text = (
                f"===== BENCHMARK RESULTS =====\n"
                f"Model Used: {model_name}\n"
                f"Input Video FPS: {source_fps:.2f}\n"
                f"Total Time: {total_time:.2f} s\n"
                f"Total Frames Read: {frames_read}\n"
                f"Frames Processed (AI): {processed_frames_count}\n"
                f"Frames Skipped (Sync): {skipped_frames_count}\n\n"
                f"--- FPS STATS ---\n"
                f"Average FPS: {avg_fps:.1f}\n"
                f"Max FPS: {max_fps:.1f}\n"
                f"Min FPS: {min_fps:.1f}\n\n"
                f"--- LATENCY STATS ---\n"
                f"Average Latency: {avg_latency:.0f} ms\n"
                f"Max Latency: {max_latency:.0f} ms\n"
                f"Min Latency: {min_latency:.0f} ms\n"
            )
            self.master.after(0, lambda: messagebox.showinfo("Benchmark Results", benchmark_text))

    def process_and_display_frame(self, frame_bgr, external_depth=None):
        if self._frame_img_w is None:
            self._frame_img_w = frame_bgr.shape[1]
        if external_depth is not None:
            self._latest_depth_map = external_depth

        # ── Opsi A: Frame skip ───────────────────────────────────────────────
        # Path planning (A*) hanya dijalankan setiap _path_skip_interval frame.
        # Mengurangi beban CPU ~60% untuk path planning tanpa degradasi visual.
        self._frame_counter += 1
        run_path = (self._frame_counter % self._path_skip_interval == 0)

        use_mc = self.mc_dropout_var.get() and self.mc_pt_path is not None
        n_samples = int(self.mc_n_samples_var.get())
        results = get_combined_prediction_from_frame(
            frame_bgr,
            self.model_configs,
            use_mc_dropout=use_mc,
            pt_path=self.mc_pt_path,
            n_samples=n_samples,
            enable_path_planning=self.path_planning_var.get() and run_path,
            external_depth=external_depth,
            camera_intrinsics=self._latest_calib,
        )

        # Pertahankan path_data lama jika frame ini skip path planning
        if not run_path and not results.get("path_planning"):
            results["path_planning"] = getattr(self, "_last_path_data", None)
        else:
            self._last_path_data = results.get("path_planning")

        inference_error = results.get("error")
        if inference_error:
            if getattr(self, "_last_inference_error", None) != inference_error:
                self._last_inference_error = inference_error
                self.master.after(0, lambda: messagebox.showwarning("Inference Warning", inference_error))
        else:
            self._last_inference_error = None

        detections = results.get("detections", [])
        analyzed_dets, abs_poly = perform_road_analysis(frame_bgr.shape[:2], detections)
        results["detections"] = analyzed_dets

        show_unc = use_mc and self.show_uncertainty_var.get()
        path_data = results.get("path_planning")
        processed_frame = (
            draw_main_visualization(frame_bgr.copy(), results, abs_poly,
                                    show_uncertainty=show_unc,
                                    path_data=path_data,
                                    frame_shape=frame_bgr.shape[:2],
                                    camera_intrinsics=self._latest_calib)
            if self.show_polygon_enabled else frame_bgr
        )
        self._schedule_canvas_update(processed_frame)
        if external_depth is not None:
            self._schedule_depth_update(external_depth)

        latency = results.get("metrics", {}).get("processing_latency_ms", 0)

        end_time = time.perf_counter()
        if self._frame_start is None:
            self._frame_start = end_time
        elapsed = end_time - self._frame_start
        self._frame_start = end_time
        if elapsed > 0:
            self.fps_queue.append(1 / elapsed)
        avg_fps = sum(self.fps_queue) / len(self.fps_queue) if self.fps_queue else 0

        mc_tag = f" | MC✓ N={n_samples}" if use_mc else ""
        status_text = f"FPS: {avg_fps:.1f} | Latency: {latency:.0f} ms{mc_tag}"
        weather_pred = results.get("weather_prediction", "N/A")
        weather_text = weather_pred.capitalize() if weather_pred != "not_applicable" else "N/A"
        self.master.after(0, self._update_status_labels, status_text, weather_text)

        # BEV juga diupdate setiap _bev_skip_interval frame
        if self._frame_counter % self._bev_skip_interval == 0:
            self._schedule_bev_update(results.get("detections", []), path_data)

        return latency, avg_fps

    def _update_status_labels(self, status_text: str, weather_text: str):
        self.status_label.config(text=status_text)
        self.weather_label.config(text=weather_text)

    def _schedule_canvas_update(self, frame_bgr):
        """Schedule update view RGB."""
        self._latest_rgb_frame = frame_bgr
        if self._canvas_update_pending:
            return
        self._canvas_update_pending = True
        self.master.after(0, self._flush_canvas_update)

    def _flush_canvas_update(self):
        frame = self._latest_rgb_frame
        self._canvas_update_pending = False
        if frame is not None and self._active_view == "rgb":
            self._update_canvas(self.main_canvas, frame)

    def _schedule_depth_update(self, depth_m: np.ndarray):
        """Colorize depth map dan schedule update view Depth."""
        if depth_m is None:
            return
        # Normalize 0–50m → 0–255, colorize PLASMA
        clipped = np.clip(depth_m, 0.0, 50.0)
        normalized = (clipped / 50.0 * 255).astype(np.uint8)
        # Invert: objek dekat = terang
        normalized = 255 - normalized
        colorized = cv2.applyColorMap(normalized, cv2.COLORMAP_PLASMA)
        self._latest_depth_frame = colorized
        if self._depth_update_pending:
            return
        self._depth_update_pending = True
        self.master.after(0, self._flush_depth_update)

    def _flush_depth_update(self):
        self._depth_update_pending = False
        if self._latest_depth_frame is not None and self._active_view == "depth":
            self._update_canvas(self.main_canvas, self._latest_depth_frame)

    def _schedule_bev_update(self, detections, path_data=None):
        self._latest_bev_detections = detections
        self._latest_path_data = path_data
        if not self._bev_update_pending:
            self._bev_update_pending = True
            # Opsi B: submit render ke thread pool, bukan ke main thread langsung
            self._render_pool.submit(self._render_bev_worker)
        if path_data and "obstacle_grid" in path_data:
            self._latest_ogm_grid = path_data["obstacle_grid"]
            self._latest_ogm_path = path_data
            if not self._ogm_update_pending:
                self._ogm_update_pending = True
                self._render_pool.submit(self._render_ogm_worker)

    def _render_bev_worker(self):
        """Jalankan di thread pool — render BEV lalu dispatch hasilnya ke GUI thread."""
        try:
            self._flush_bev_update()
        except Exception as e:
            print(f"[BEV render error] {e}")

    def _render_ogm_worker(self):
        """Jalankan di thread pool — render OGM lalu dispatch hasilnya ke GUI thread."""
        try:
            self._flush_ogm_update()
        except Exception as e:
            print(f"[OGM render error] {e}")

    def _flush_bev_update(self):
        dets = self._latest_bev_detections
        path_data = self._latest_path_data
        self._latest_bev_detections = None
        self._latest_path_data = None
        self._bev_update_pending = False
        if dets is None:
            return

        canvas_w = self.main_canvas.winfo_width() or 600
        canvas_h = self.main_canvas.winfo_height() or 800
        depth_m = self._latest_depth_map

        if depth_m is not None:
            if self._latest_calib is not None:
                intrinsics = intrinsics_from_calib(self._latest_calib)
            else:
                orig_w = self._frame_img_w or 640
                intrinsics = intrinsics_from_frame_width(orig_w)

            dets_3d = backproject_detections(dets, depth_m, intrinsics)

            path_wpts = None
            if path_data and path_data.get("path_found"):
                from app.core.services.object_detection import BEV_WIDTH, BEV_HEIGHT
                sx = canvas_w / BEV_WIDTH
                sy = canvas_h / BEV_HEIGHT
                path_wpts = [(int(wx * sx), int(wy * sy))
                             for wx, wy in path_data["waypoints"]]

            # Dense point cloud BEV + detection overlay + path overlay
            bev_img = render_dense_bev(
                depth_map=depth_m,
                intrinsics=intrinsics,
                canvas_wh=(canvas_w, canvas_h),
                detections_3d=dets_3d,
                path_waypoints=path_wpts,
                downsample=3,        # GPU: abaikan (full-density); CPU: ~11% piksel
            )

            if path_data and not path_data.get("path_found"):
                cv2.putText(bev_img, "NO PATH", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            bev_img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            ego_px = canvas_w // 2
            cv2.rectangle(bev_img, (ego_px - 8, canvas_h - 20),
                          (ego_px + 8, canvas_h - 5), (0, 220, 220), -1)
            cv2.putText(bev_img, "Waiting for depth...", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        self._latest_bev_frame = bev_img
        # Dispatch ke GUI thread — _update_canvas harus dipanggil dari main thread
        if self._active_view == "bev":
            self.master.after(0, lambda img=bev_img: self._update_canvas(self.main_canvas, img))

    def _flush_ogm_update(self):
        grid = self._latest_ogm_grid
        path_data = self._latest_ogm_path
        self._latest_ogm_grid = None
        self._latest_ogm_path = None
        self._ogm_update_pending = False
        if grid is None:
            return

        ogm_rgb = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=np.uint8)
        ogm_rgb[grid > 127] = [0, 255, 255]
        ogm_rgb[grid > 200] = [0, 0, 255]

        if path_data and path_data.get("path_found"):
            waypoints = path_data.get("waypoints", [])
            for i in range(len(waypoints) - 1):
                cv2.line(ogm_rgb,
                         (int(waypoints[i][0]),   int(waypoints[i][1])),
                         (int(waypoints[i+1][0]), int(waypoints[i+1][1])),
                         (0, 200, 255), 2)
            if waypoints:
                cv2.circle(ogm_rgb, (int(waypoints[0][0]),  int(waypoints[0][1])),  4, (255, 255, 0), -1)
                cv2.circle(ogm_rgb, (int(waypoints[-1][0]), int(waypoints[-1][1])), 4, (0, 255, 0),   -1)

        h, w = ogm_rgb.shape[:2]
        # Draw ego vehicle as a vertical rectangle (narrow lateral, longer forward)
        ego_cx = w // 2
        ego_w, ego_h = 8, 18  # ~1m wide, ~2.2m long in BEV scale
        cv2.rectangle(ogm_rgb, (ego_cx - ego_w // 2, h - ego_h - 2), (ego_cx + ego_w // 2, h - 2), (0, 180, 180), -1)

        self._latest_ogm_frame = ogm_rgb
        if self._active_view == "ogm":
            self.master.after(0, lambda img=ogm_rgb: self._update_canvas(self.main_canvas, img))

    def _update_canvas(self, widget, frame_bgr):
        target_w, target_h = widget.winfo_width(), widget.winfo_height()
        if target_w < 2 or target_h < 2:
            return
        h, w = frame_bgr.shape[:2]
        ratio = min(target_w / w, target_h / h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        # Resize dulu (BGR, frame kecil), lalu color convert — ~4x lebih hemat RAM
        resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        frame_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_tk = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
        if self._canvas_image_id is None:
            self._canvas_image_id = widget.create_image(target_w / 2, target_h / 2, image=img_tk, anchor=tk.CENTER)
        else:
            widget.coords(self._canvas_image_id, target_w / 2, target_h / 2)
            widget.itemconfig(self._canvas_image_id, image=img_tk)
        widget.image = img_tk  # prevent garbage collection

    def _update_dummy_data(self):
        speed   = f"{random.randint(20, 40)} Km/h"
        battery = f"{random.randint(70, 95)} %"
        lat     = f"{random.uniform(-6.1, -6.3):.4f}"
        lon     = f"{random.uniform(106.7, 106.9):.4f}"
        self.speed_label.config(text=speed)
        self.battery_label.config(text=battery)
        self.location_label.config(text=f"{lat}, {lon}")
        self.master.after(2000, self._update_dummy_data)

    def on_closing(self):
        self.stop_processing_flag.set()
        self._render_pool.shutdown(wait=False, cancel_futures=True)
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1)
        self.master.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = DashboardGUI(root)
    root.mainloop()
