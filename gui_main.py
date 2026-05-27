# File: gui_main.py

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import cv2
import threading
import time
from collections import deque
import random
import numpy as np

from app.core.services.object_detection import (
    get_combined_prediction_from_frame,
    perform_road_analysis,
    draw_main_visualization,
    set_active_model,
    discover_models  # ← new import
)

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
        self.depth_model_path = None

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
        self.main_canvas.config(bg=colors["CANVAS"])

    # -------------------------------------------------------------------------

    def _create_widgets(self):
        self.master.columnconfigure(0, weight=7)
        self.master.columnconfigure(1, weight=3, minsize=280)
        self.master.rowconfigure(0, weight=1)

        main_display_frame = ttk.Frame(self.master)
        main_display_frame.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        self._create_main_display(main_display_frame)

        control_panel_frame = ttk.Frame(self.master)
        control_panel_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
        self._create_control_panel(control_panel_frame)

    def _create_main_display(self, parent):
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)

        self.main_canvas = tk.Canvas(parent, highlightthickness=0)
        self.main_canvas.grid(row=0, column=0, sticky="nsew")

        footer_frame = ttk.Frame(parent, style="Status.TLabel")
        footer_frame.grid(row=1, column=0, sticky="ew")
        footer_frame.columnconfigure(0, weight=1)

        self.show_polygon_check = ttk.Checkbutton(
            footer_frame, text="Show Polygon",
            variable=self.show_polygon, style="TCheckbutton",
            command=self._on_show_polygon_changed
        )
        self.show_polygon_check.pack(side=tk.LEFT, padx=10)

        self.status_label = ttk.Label(footer_frame, text="FPS: - | Latency: - ms", style="Status.TLabel")
        self.status_label.pack(side=tk.RIGHT, padx=10)

    def _create_control_panel(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=0)
        parent.rowconfigure(1, weight=0)
        parent.rowconfigure(2, weight=1)
        parent.rowconfigure(3, weight=1)

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
            values=["Select Source...", "Image File", "Video File", "Webcam"],
            state="readonly", width=20
        )
        self.mode_combo.set("Select Source...")
        self.mode_combo.pack(pady=4, fill=tk.X)
        self.mode_combo.bind("<<ComboboxSelected>>", self.on_mode_change)

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

        self.depth_model_label = ttk.Label(
            controls_frame, text="No depth model loaded",
            style="Header.TLabel", foreground="gray"
        )
        self.depth_model_label.pack(fill=tk.X)
        ttk.Button(controls_frame, text="Load Depth Model (.onnx)",
                   command=self._load_depth_model).pack(fill=tk.X, pady=2)

        ttk.Checkbutton(controls_frame, text="Enable Path Planning",
                        variable=self.path_planning_var,
                        style="TCheckbutton").pack(anchor="w")

        # --- Status cards ---
        status_cards_frame = ttk.Frame(parent)
        status_cards_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        status_cards_frame.columnconfigure(0, weight=1)
        self._create_status_cards_panel(status_cards_frame)

        # --- BEV & OGM placeholders ---
        self.bev_frame = ttk.Frame(parent, style="Card.TFrame")
        self.bev_frame.grid(row=2, column=0, sticky="nsew", pady=(0, 5))
        self.bev_canvas = tk.Canvas(self.bev_frame, bg="black", highlightthickness=0)
        self.bev_canvas.pack(expand=True, fill=tk.BOTH)
        self._bev_image_id = None
        self._latest_bev_detections = None
        self._latest_path_data = None
        self._bev_update_pending = False

        self.ogm_frame = ttk.Frame(parent, style="Card.TFrame")
        self.ogm_frame.grid(row=3, column=0, sticky="nsew", pady=(5, 0))
        ttk.Label(self.ogm_frame, text="Occupancy Grid Mapping", style="Placeholder.TLabel").pack(expand=True)

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

    def _load_depth_model(self):
        path = filedialog.askopenfilename(
            filetypes=[("ONNX Model", "*.onnx")]
        )
        if path:
            self.depth_model_path = path
            self.depth_model_label.config(
                text=os.path.basename(path), foreground="green"
            )

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

    def stop_processing(self):
        self.stop_processing_flag.set()
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
        source = cv2.VideoCapture(self.source_path)
        if mode == "Webcam":
            source.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not source.isOpened():
            self.master.after(0, lambda: messagebox.showerror("Error", "Could not open video/image source."))
            self.master.after(0, self.stop_processing)
            return

        source_fps = 0.0
        frame_interval = 0.0
        if mode == "Video File":
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
            ret, frame = source.read()
            if not ret:
                break
                
            frames_read += 1
            processed_frames_count += 1
            latency, frame_fps = self.process_and_display_frame(frame)
            latency_history.append(latency)
            if frame_fps > 0:
                fps_history.append(frame_fps)

            if mode == "Video File":
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

    def process_and_display_frame(self, frame_bgr):
        use_mc = self.mc_dropout_var.get() and self.mc_pt_path is not None
        n_samples = int(self.mc_n_samples_var.get())
        results = get_combined_prediction_from_frame(
            frame_bgr,
            self.model_configs,
            use_mc_dropout=use_mc,
            pt_path=self.mc_pt_path,
            n_samples=n_samples,
            depth_model_path=self.depth_model_path,
            enable_path_planning=self.path_planning_var.get()
        )
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
        processed_frame = (
            draw_main_visualization(frame_bgr.copy(), results, abs_poly,
                                    show_uncertainty=show_unc)
            if self.show_polygon_enabled else frame_bgr
        )
        self._schedule_canvas_update(processed_frame)

        latency = results.get("metrics", {}).get("processing_latency_ms", 0)

        # Recalculate FPS properly
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
        self._schedule_bev_update(
            results.get("detections", []),
            results.get("path_planning")
        )
        
        return latency, avg_fps

    def _update_status_labels(self, status_text: str, weather_text: str):
        self.status_label.config(text=status_text)
        self.weather_label.config(text=weather_text)

    def _schedule_canvas_update(self, frame_bgr):
        self._latest_canvas_frame = frame_bgr
        if self._canvas_update_pending:
            return
        self._canvas_update_pending = True
        self.master.after(0, self._flush_canvas_update)

    def _flush_canvas_update(self):
        frame_bgr = self._latest_canvas_frame
        self._latest_canvas_frame = None
        self._canvas_update_pending = False
        if frame_bgr is not None:
            self._update_canvas(self.main_canvas, frame_bgr)
        if self._latest_canvas_frame is not None and not self._canvas_update_pending:
            self._canvas_update_pending = True
            self.master.after(0, self._flush_canvas_update)

    def _schedule_bev_update(self, detections, path_data=None):
        self._latest_bev_detections = detections
        self._latest_path_data = path_data
        if not self._bev_update_pending:
            self._bev_update_pending = True
            self.master.after(0, self._flush_bev_update)

    def _flush_bev_update(self):
        dets = self._latest_bev_detections
        path_data = self._latest_path_data
        self._latest_bev_detections = None
        self._latest_path_data = None
        self._bev_update_pending = False
        if dets is None:
            return

        from app.core.services.object_detection import (
            BEV_WIDTH, BEV_HEIGHT, PIXELS_PER_METER, HAZARD_COLORS
        )

        canvas_w = self.bev_canvas.winfo_width() or BEV_WIDTH
        canvas_h = self.bev_canvas.winfo_height() or BEV_HEIGHT
        bev_img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        cx = canvas_w // 2
        cv2.line(bev_img, (cx, 0), (cx, canvas_h), (40, 40, 40), 1)
        cv2.rectangle(bev_img, (cx - 8, canvas_h - 20), (cx + 8, canvas_h - 5),
                      (0, 180, 180), -1)

        for det in dets:
            dist = det.get("distance_m", 0)
            if dist <= 0:
                continue

            hazard = det.get("hazard_level", "safe")
            color = HAZARD_COLORS.get(hazard, (128, 128, 128))
            bbox = det.get("bounding_box", [0, 0, 0, 0])
            img_cx = (bbox[0] + bbox[2]) / 2
            x_norm = (img_cx / 640.0) - 0.5
            x_bev = int(cx + x_norm * canvas_w * 0.8)
            y_bev = int(canvas_h - dist * PIXELS_PER_METER)
            y_bev = max(5, min(canvas_h - 5, y_bev))

            radius = 6
            cv2.circle(bev_img, (x_bev, y_bev), radius, color, -1)

            unc = det.get("uncertainty", 0)
            if unc >= 0.1:
                cv2.circle(bev_img, (x_bev, y_bev), radius + 4, (0, 0, 255), 1)

            cv2.putText(bev_img, det.get("label", "")[:4],
                        (x_bev + 8, y_bev + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        if path_data and path_data.get("path_found"):
            waypoints = path_data["waypoints"]
            for i in range(len(waypoints) - 1):
                pt1 = (int(waypoints[i][0]), int(waypoints[i][1]))
                pt2 = (int(waypoints[i + 1][0]), int(waypoints[i + 1][1]))
                cv2.line(bev_img, pt1, pt2, (0, 200, 255), 2)
            if waypoints:
                start_pt = (int(waypoints[0][0]), int(waypoints[0][1]))
                goal_pt = (int(waypoints[-1][0]), int(waypoints[-1][1]))
                cv2.circle(bev_img, start_pt, 5, (255, 255, 0), -1)
                cv2.circle(bev_img, goal_pt, 5, (0, 255, 0), -1)
        elif path_data and not path_data.get("path_found"):
            cv2.putText(bev_img, "NO PATH", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        img_rgb = cv2.cvtColor(bev_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)
        if self._bev_image_id is None:
            self._bev_image_id = self.bev_canvas.create_image(
                canvas_w // 2, canvas_h // 2, image=img_tk, anchor=tk.CENTER)
        else:
            self.bev_canvas.itemconfig(self._bev_image_id, image=img_tk)
        self.bev_canvas.image = img_tk

    def _update_canvas(self, widget, frame_bgr):
        target_w, target_h = widget.winfo_width(), widget.winfo_height()
        if target_w < 2 or target_h < 2:
            return
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = frame_rgb.shape
        ratio = min(target_w / w, target_h / h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        # Use INTER_LINEAR for better performance on CPU
        resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        img_tk = ImageTk.PhotoImage(image=Image.fromarray(resized))
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
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1)
        self.master.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = DashboardGUI(root)
    root.mainloop()
