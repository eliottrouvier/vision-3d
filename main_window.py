"""
MainWindow — Vision-3D Studio Application Window (PySide6)
-----------------------------------------------------------
Integrates:
1. Live Video Feed with 3D Volumetric Mesh & Skeleton overlay
2. Real-Time 3D Mannequin Avatar Viewport (OpenGL)
3. Modulable Split-View & Floating Picture-in-Picture (PiP) modes
4. Complete Transport, Recording, and Source Management
"""

import os
import sys
import time
import cv2
import numpy as np

from PySide6.QtCore import Qt, QTimer, QThread, Signal, Slot, QSize
from PySide6.QtGui import QImage, QPixmap, QIcon, QFont, QColor
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QSlider, QCheckBox, QFrame,
    QSplitter, QFileDialog, QMessageBox, QSizePolicy
)

from tracker_3d import Tracker3D
from gl_avatar_widget import GLAvatarWidget

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


class VideoWorker(QThread):
    frame_ready = Signal(np.ndarray, dict)

    def __init__(self, source_path=None, is_webcam=False):
        super().__init__()
        self.source_path = source_path
        self.is_webcam = is_webcam
        self.running = True
        self.playing = False
        self.cap = None
        self.fps = 30.0
        self.speed = 1.0
        self.total_frames = 0
        self.tracker = Tracker3D()

    def open_source(self, source_path=None, is_webcam=False):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.is_webcam = is_webcam
        self.source_path = source_path

        if self.is_webcam:
            backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
            self.cap = cv2.VideoCapture(0, backend)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.fps = 30.0
            self.total_frames = 0
            self.playing = True
        else:
            self.cap = cv2.VideoCapture(self.source_path)
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            self.playing = False

    def seek(self, frame_idx):
        if self.cap and not self.is_webcam:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    def run(self):
        while self.running:
            if not self.playing:
                time.sleep(0.03)
                continue

            t0 = time.perf_counter()
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.05)
                continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                if not self.is_webcam and self.total_frames > 0:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                if not ret or frame is None:
                    time.sleep(0.05)
                    continue

            # Process 3D tracking
            tracking_res = self.tracker.process(frame)
            self.frame_ready.emit(frame, tracking_res if tracking_res else {})

            elapsed = time.perf_counter() - t0
            target_delay = (1.0 / self.fps) / max(0.1, self.speed)
            rem = target_delay - elapsed
            if rem > 0:
                time.sleep(rem)

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()


class MainWindow(QMainWindow):
    def __init__(self, initial_source=None, is_webcam=False):
        super().__init__()

        self.setWindowTitle("Vision-3D — Human Mesh Recovery & 3D Avatar")
        self.resize(1380, 860)
        self.setMinimumSize(1080, 680)

        # Style & Palette (Apple Pro Zinc theme)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #09090b;
            }
            QWidget {
                color: #f4f4f5;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            }
            QFrame#card {
                background-color: #121215;
                border: 1px solid #27272a;
                border-radius: 8px;
            }
            QPushButton {
                background-color: #1c1d22;
                border: 1px solid #27272a;
                border-radius: 6px;
                padding: 6px 12px;
                color: #e4e4e7;
                font-weight: 500;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #27272a;
                color: #ffffff;
            }
            QPushButton:pressed {
                background-color: #3f3f46;
            }
            QPushButton#btn_accent {
                background-color: #27272a;
                color: #f4f4f5;
                font-weight: bold;
            }
            QPushButton#btn_rec {
                background-color: #3f1218;
                border: 1px solid #7f1d1d;
                color: #fca5a5;
                font-weight: bold;
            }
            QComboBox {
                background-color: #18181b;
                border: 1px solid #27272a;
                border-radius: 6px;
                padding: 5px 10px;
                color: #f4f4f5;
                font-size: 12px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #27272a;
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal {
                background: #e4e4e7;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #ffffff;
                width: 14px;
                margin-top: -5px;
                margin-bottom: -5px;
                border-radius: 7px;
            }
            QCheckBox {
                font-size: 12px;
                color: #a1a1aa;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #3f3f46;
                border-radius: 4px;
                background: #18181b;
            }
            QCheckBox::indicator:checked {
                background: #e4e4e7;
                border-color: #ffffff;
            }
        """)

        # Demo Sample Videos Dictionary
        self.samples = {
            "🏋️ Fitness & Squats (Kinématique)": os.path.join(PROJECT_DIR, "samples", "squats_workout.mp4"),
            "💃 Danse & Posture (Mouvements)": os.path.join(PROJECT_DIR, "samples", "dance_movement.mp4"),
            "🤸 Gainage & Pompes (Haut du corps)": os.path.join(PROJECT_DIR, "samples", "pushups_workout.mp4"),
        }

        # Display Toggles State
        self.show_mesh = True
        self.show_skeleton = True
        self.show_bbox = True
        self.show_hud = True
        self.is_pip_mode = False
        self.is_recording = False
        self.record_start_time = 0.0

        # Video worker & Tracker
        default_video = initial_source or self.samples["🏋️ Fitness & Squats (Kinématique)"]
        self.worker = VideoWorker(source_path=default_video, is_webcam=is_webcam)
        self.worker.frame_ready.connect(self.on_frame_ready)

        self._build_ui()
        self.worker.open_source(source_path=default_video, is_webcam=is_webcam)
        self.worker.start()

        # Trigger first frame render
        QTimer.singleShot(150, self._render_initial_frame)

    def _build_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(16, 12, 16, 16)
        main_layout.setSpacing(12)

        # --------------------------------------------------------------
        # 1. TOP HEADER BAR
        # --------------------------------------------------------------
        top_bar = QHBoxLayout()
        top_bar.setSpacing(12)

        title_lbl = QLabel("👁‍🗨 Vision-3D")
        title_lbl.setFont(QFont("Arial", 15, QFont.Bold))
        title_lbl.setStyleSheet("color: #f4f4f5;")
        top_bar.addWidget(title_lbl)

        sub_lbl = QLabel("3D Human Mesh Recovery & Real-Time Avatar")
        sub_lbl.setFont(QFont("Arial", 11))
        sub_lbl.setStyleSheet("color: #71717a;")
        top_bar.addWidget(sub_lbl)

        top_bar.addStretch()

        # Accel Badge
        accel_pill = QLabel("● APPLE SILICON ACCELERATED (M4/MPS)")
        accel_pill.setStyleSheet("""
            background-color: #121215;
            color: #22c55e;
            border: 1px solid #27272a;
            border-radius: 12px;
            padding: 4px 12px;
            font-size: 11px;
            font-weight: bold;
        """)
        top_bar.addWidget(accel_pill)

        # View Mode Toggle (Split-View vs PiP)
        self.btn_view_mode = QPushButton("◫ Vue Côte à Côte")
        self.btn_view_mode.clicked.connect(self.toggle_view_mode)
        top_bar.addWidget(self.btn_view_mode)

        # Recording Button
        self.btn_rec = QPushButton("⏺ REC 0:00")
        self.btn_rec.setObjectName("btn_rec")
        self.btn_rec.clicked.connect(self.toggle_recording)
        top_bar.addWidget(self.btn_rec)

        # Snapshot Button
        self.btn_snap = QPushButton("📸 Capture HD")
        self.btn_snap.clicked.connect(self.take_snapshot)
        top_bar.addWidget(self.btn_snap)

        main_layout.addLayout(top_bar)

        # --------------------------------------------------------------
        # 2. CENTRAL VIEWPORT AREA (Splitter: Video on Left, 3D on Right)
        # --------------------------------------------------------------
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #27272a;
                width: 2px;
            }
        """)

        # Left Container: Video Feed
        self.video_container = QFrame()
        self.video_container.setObjectName("card")
        video_vbox = QVBoxLayout(self.video_container)
        video_vbox.setContentsMargins(4, 4, 4, 4)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #000000; border-radius: 6px;")
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        video_vbox.addWidget(self.video_label)

        # Right Container: 3D Avatar OpenGL Viewport
        self.avatar_container = QFrame()
        self.avatar_container.setObjectName("card")
        avatar_vbox = QVBoxLayout(self.avatar_container)
        avatar_vbox.setContentsMargins(4, 4, 4, 4)

        # OpenGL Widget
        self.gl_widget = GLAvatarWidget()
        self.gl_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Title bar for 3D Viewport
        av_top = QHBoxLayout()
        av_top.setContentsMargins(8, 6, 8, 4)
        av_top.setSpacing(6)
        av_title = QLabel("🧍 Mannequin 3D Facetté (SMPL)")
        av_title.setFont(QFont("Arial", 11, QFont.Bold))
        av_title.setStyleSheet("color: #e4e4e7;")
        av_top.addWidget(av_title)
        av_top.addStretch()

        btn_zoom_in = QPushButton("＋ Zoom")
        btn_zoom_in.setToolTip("Zoomer (+ ou molette)")
        btn_zoom_in.setStyleSheet("padding: 3px 8px; font-size: 11px;")
        btn_zoom_in.clicked.connect(lambda: self.gl_widget.zoom_in())
        av_top.addWidget(btn_zoom_in)

        btn_zoom_out = QPushButton("－ Dézoom")
        btn_zoom_out.setToolTip("Dézoomer (- ou molette)")
        btn_zoom_out.setStyleSheet("padding: 3px 8px; font-size: 11px;")
        btn_zoom_out.clicked.connect(lambda: self.gl_widget.zoom_out())
        av_top.addWidget(btn_zoom_out)

        btn_reset_cam = QPushButton("↺ Recentrer")
        btn_reset_cam.setToolTip("Recentrer la caméra 3D (0 ou double-clic)")
        btn_reset_cam.setStyleSheet("padding: 3px 8px; font-size: 11px;")
        btn_reset_cam.clicked.connect(lambda: self.gl_widget.reset_camera())
        av_top.addWidget(btn_reset_cam)
        avatar_vbox.addLayout(av_top)
        avatar_vbox.addWidget(self.gl_widget)

        self.splitter.addWidget(self.video_container)
        self.splitter.addWidget(self.avatar_container)
        self.splitter.setSizes([690, 690])
        main_layout.addWidget(self.splitter, 1)

        # --------------------------------------------------------------
        # 3. BOTTOM MEDIA TRANSPORT & TIMELINE
        # --------------------------------------------------------------
        transport_card = QFrame()
        transport_card.setObjectName("card")
        transport_vbox = QVBoxLayout(transport_card)
        transport_vbox.setContentsMargins(14, 8, 14, 8)
        transport_vbox.setSpacing(6)

        # Timeline Slider Row
        timeline_row = QHBoxLayout()
        self.time_lbl_left = QLabel("00:00")
        self.time_lbl_left.setStyleSheet("color: #f4f4f5; font-weight: bold; font-size: 11px;")
        timeline_row.addWidget(self.time_lbl_left)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.sliderMoved.connect(self.on_seek)
        timeline_row.addWidget(self.slider, 1)

        self.time_lbl_right = QLabel("00:00")
        self.time_lbl_right.setStyleSheet("color: #71717a; font-size: 11px;")
        timeline_row.addWidget(self.time_lbl_right)
        transport_vbox.addLayout(timeline_row)

        # Controls Row (Source, Toggles, Play/Pause)
        controls_row = QHBoxLayout()
        controls_row.setSpacing(10)

        # Play / Pause
        self.btn_play = QPushButton("▶ Lecture")
        self.btn_play.setObjectName("btn_accent")
        self.btn_play.setFixedWidth(90)
        self.btn_play.clicked.connect(self.toggle_play)
        controls_row.addWidget(self.btn_play)

        # Rewind
        self.btn_rewind = QPushButton("↺ Recommencer")
        self.btn_rewind.clicked.connect(self.rewind_video)
        controls_row.addWidget(self.btn_rewind)

        # Source Dropdown
        controls_row.addWidget(QLabel("Source :"))
        self.combo_source = QComboBox()
        self.combo_source.addItem("📹 Webcam Direct (macOS)")
        for name in self.samples.keys():
            self.combo_source.addItem(name)
        self.combo_source.addItem("📁 Parcourir un fichier vidéo...")
        self.combo_source.setCurrentIndex(1) # Default to squats
        self.combo_source.currentIndexChanged.connect(self.on_source_selected)
        controls_row.addWidget(self.combo_source)

        controls_row.addSpacing(12)

        # 3D Toggles
        self.chk_mesh = QCheckBox("🕸️ Maillage Vidéo")
        self.chk_mesh.setChecked(True)
        self.chk_mesh.stateChanged.connect(lambda s: setattr(self, 'show_mesh', s == Qt.Checked.value))
        controls_row.addWidget(self.chk_mesh)

        self.chk_wireframe = QCheckBox("📐 Arêtes Facettées")
        self.chk_wireframe.setChecked(True)
        self.chk_wireframe.stateChanged.connect(lambda s: self.gl_widget.set_show_wireframe(s == Qt.Checked.value))
        controls_row.addWidget(self.chk_wireframe)

        self.chk_smooth = QCheckBox("✨ Surface Lisse")
        self.chk_smooth.setChecked(False)
        self.chk_smooth.stateChanged.connect(lambda s: self.gl_widget.set_shading_mode(not (s == Qt.Checked.value)))
        controls_row.addWidget(self.chk_smooth)

        self.chk_skel = QCheckBox("🦴 Squelette")
        self.chk_skel.setChecked(False)
        self.chk_skel.stateChanged.connect(lambda s: setattr(self.gl_widget, 'show_skeleton', s == Qt.Checked.value) or self.gl_widget.update())
        controls_row.addWidget(self.chk_skel)

        self.chk_grid = QCheckBox("🌐 Grille 3D")
        self.chk_grid.setChecked(True)
        self.chk_grid.stateChanged.connect(lambda s: setattr(self.gl_widget, 'show_grid', s == Qt.Checked.value) or self.gl_widget.update())
        controls_row.addWidget(self.chk_grid)

        self.chk_filter = QCheckBox("🛡️ Stabilisation 1€")
        self.chk_filter.setChecked(True)
        self.chk_filter.setToolTip("Filtre adaptatif 1€ éliminant les micro-tremblements")
        self.chk_filter.stateChanged.connect(lambda s: setattr(self.worker.tracker, 'filter_enabled', s == Qt.Checked.value))
        controls_row.addWidget(self.chk_filter)

        self.chk_ground = QCheckBox("⚓ Ancrage Sol")
        self.chk_ground.setChecked(True)
        self.chk_ground.setToolTip("Maintient les pieds au sol (en squat, le bassin descend réellement)")
        self.chk_ground.stateChanged.connect(lambda s: self.gl_widget.set_foot_grounding(s == Qt.Checked.value))
        controls_row.addWidget(self.chk_ground)

        controls_row.addStretch()

        # Speed selector
        controls_row.addWidget(QLabel("Vitesse :"))
        self.combo_speed = QComboBox()
        self.combo_speed.addItems(["0.5x", "1x", "1.5x", "2x"])
        self.combo_speed.setCurrentText("1x")
        self.combo_speed.currentTextChanged.connect(self.on_speed_changed)
        controls_row.addWidget(self.combo_speed)

        transport_vbox.addLayout(controls_row)
        main_layout.addWidget(transport_card)

        # Recording timer
        self.rec_timer = QTimer(self)
        self.rec_timer.timeout.connect(self._update_rec_timer)

    def _render_initial_frame(self):
        if self.worker.cap and self.worker.cap.isOpened():
            self.worker.seek(0)
            ret, frame = self.worker.cap.read()
            if ret and frame is not None:
                self.worker.seek(0)
                res = self.worker.tracker.process(frame)
                self.on_frame_ready(frame, res if res else {})

    @Slot(np.ndarray, dict)
    def on_frame_ready(self, frame, tracking_res):
        # 1. Update 3D Avatar Viewport
        if tracking_res and "pts_world" in tracking_res:
            self.gl_widget.set_landmarks_3d(tracking_res["pts_world"], tracking_res.get("vis"))

        # 2. Annotate Video Frame (Mesh + Skeleton + BBox + HUD)
        annotated = self.worker.tracker.annotate_frame(
            frame, tracking_res if tracking_res else None,
            show_mesh=self.show_mesh,
            show_skeleton=self.show_skeleton,
            show_bbox=self.show_bbox,
            show_hud=self.show_hud
        )

        # 3. Convert to QPixmap and display
        h, w, ch = annotated.shape
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        bytes_per_line = ch * w
        q_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        lbl_w = max(100, self.video_label.width())
        lbl_h = max(100, self.video_label.height())
        scaled = QPixmap.fromImage(q_img).scaled(lbl_w, lbl_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled)

        # 4. Update Scrubber
        if not self.worker.is_webcam and self.worker.total_frames > 0:
            cur_f = int(self.worker.cap.get(cv2.CAP_PROP_POS_FRAMES)) if self.worker.cap else 0
            self.slider.setRange(0, self.worker.total_frames - 1)
            self.slider.setValue(cur_f)

            sec = int(cur_f / max(1.0, self.worker.fps))
            tot_sec = int(self.worker.total_frames / max(1.0, self.worker.fps))
            self.time_lbl_left.setText(f"{sec // 60:02d}:{sec % 60:02d}")
            self.time_lbl_right.setText(f"{tot_sec // 60:02d}:{tot_sec % 60:02d}")

    # ==============================================================
    # CONTROLS & EVENT HANDLERS
    # ==============================================================
    def toggle_play(self):
        self.worker.playing = not self.worker.playing
        self.btn_play.setText("⏸ Pause" if self.worker.playing else "▶ Lecture")

    def rewind_video(self):
        self.worker.tracker.reset_filter()
        self.gl_widget.reset_filter()
        self.worker.seek(0)
        self.slider.setValue(0)
        self._render_initial_frame()

    def on_seek(self, val):
        self.worker.tracker.reset_filter()
        self.gl_widget.reset_filter()
        self.worker.seek(val)
        if not self.worker.playing:
            self._render_initial_frame()

    def on_speed_changed(self, text):
        val = float(text.replace("x", ""))
        self.worker.speed = val

    def on_toggle_mannequin(self, state):
        self.gl_widget.show_mannequin = (state == Qt.Checked.value)
        self.gl_widget.update()

    def reset_avatar_camera(self):
        self.gl_widget.reset_camera()

    def toggle_view_mode(self):
        self.is_pip_mode = not self.is_pip_mode
        if self.is_pip_mode:
            self.btn_view_mode.setText("🗖 Incrustation (PiP)")
            self.splitter.setSizes([1000, 380])
        else:
            self.btn_view_mode.setText("◫ Vue Côte à Côte")
            self.splitter.setSizes([690, 690])

    def toggle_recording(self):
        self.is_recording = not self.is_recording
        if self.is_recording:
            self.record_start_time = time.time()
            self.btn_rec.setText("⏹ ARRÊTER (00:00)")
            self.btn_rec.setStyleSheet("background-color: #991b1b; color: #ffffff; font-weight: bold;")
            self.rec_timer.start(1000)
        else:
            self.rec_timer.stop()
            self.btn_rec.setText("⏺ REC 0:00")
            self.btn_rec.setStyleSheet("background-color: #3f1218; border: 1px solid #7f1d1d; color: #fca5a5; font-weight: bold;")
            QMessageBox.information(self, "Enregistrement 3D", "Séquence de mouvement 3D enregistrée avec succès !")

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key_Plus, Qt.Key_Equal):
            self.gl_widget.zoom_in()
        elif key in (Qt.Key_Minus, Qt.Key_Underscore):
            self.gl_widget.zoom_out()
        elif key in (Qt.Key_0, Qt.Key_R):
            self.gl_widget.reset_camera()
        elif key == Qt.Key_Space:
            self.toggle_play()
        else:
            super().keyPressEvent(event)

    def _update_rec_timer(self):
        elapsed = int(time.time() - self.record_start_time)
        m = elapsed // 60
        s = elapsed % 60
        self.btn_rec.setText(f"⏹ ARRÊTER ({m:02d}:{s:02d})")

    def take_snapshot(self):
        if self.video_label.pixmap():
            filename = f"capture_3d_{int(time.time())}.jpg"
            self.video_label.pixmap().save(filename, "JPG")
            QMessageBox.information(self, "Capture Sauvegardée", f"Image HD enregistrée sous :\n{filename}")

    def on_source_selected(self, idx):
        self.worker.tracker.reset_filter()
        self.gl_widget.reset_filter()
        text = self.combo_source.currentText()
        if "Webcam" in text:
            self.worker.open_source(is_webcam=True)
            self.btn_play.setText("⏸ Pause")
            self.time_lbl_left.setText("LIVE")
            self.time_lbl_right.setText("LIVE")
        elif "Parcourir" in text:
            chosen, _ = QFileDialog.getOpenFileName(
                self, "Choisir une vidéo", PROJECT_DIR, "Vidéos (*.mp4 *.avi *.mov *.mkv)"
            )
            if chosen:
                self.worker.open_source(source_path=chosen, is_webcam=False)
                self.btn_play.setText("▶ Lecture")
                self._render_initial_frame()
            else:
                self.combo_source.setCurrentIndex(1)
        else:
            path = self.samples.get(text)
            if path and os.path.exists(path):
                self.worker.open_source(source_path=path, is_webcam=False)
                self.btn_play.setText("▶ Lecture")
                self._render_initial_frame()

    def closeEvent(self, event):
        self.worker.stop()
        self.worker.wait(500)
        event.accept()
