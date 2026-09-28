# 👁‍🗨 Vision-3D Studio
### 3D Human Mesh Recovery (HMR) & Real-Time Mannequin Avatar Studio

Vision-3D is an advanced, high-performance computer vision desktop application for **3D Human Mesh Recovery (HMR)** and **real-time kinematic avatar replication**. 

Built with **PySide6**, **PyOpenGL**, and **MediaPipe 3D Pose World Landmarker**, it reproduces the visual aesthetic and precision of professional motion capture (MoCap) and biomechanical analysis suites on Apple Silicon (M4 / MPS / AVFoundation).

---

## 🌟 Key Features

- **Volumetric 3D Wireframe Mesh**: Overlays polygonal cross-sectional rings (chest, abdomen, arms, forearms, thighs, shins, head) connected by longitudinal guides directly onto the 2D video feed.
- **3D Mannequin Avatar Mirror (OpenGL)**: A shaded, studio-lit anatomical humanoid avatar that reproduces the subject's exact 3D metric posture in real time.
- **360° Interactive Orbit Viewport**:
  - **Left Click + Drag**: Orbit / 360° rotate around the avatar.
  - **Right Click + Drag**: Pan camera in 3D space.
  - **Scroll Wheel**: Smooth zoom in / out.
  - **Double Click**: Instantly reset camera to standard front view.
- **Dual View Modes**:
  - **Split View**: Side-by-side video feed and 3D avatar viewport with adjustable ratio splitter.
  - **Picture-in-Picture (PiP)**: Inset 3D viewport superimposed directly onto the video feed, mirroring MoCap analysis HUDs.
- **Biomechanical HUD**: Real-time analysis FPS, tracking latency in milliseconds, bounding box, horizon pitch line, and active tracking indicators.
- **Pro Media Transport**:
  - Play / Pause / Step / Rewind.
  - Interactive timeline scrubber with timestamp display.
  - Multi-speed playback (0.5x, 0.75x, 1.0x, 1.25x, 1.5x, 2.0x).
- **Multi-Source Input**:
  - Built-in cinematic motion samples (Squats Kinematics, Dance Posture, Pushups Movement).
  - Native macOS FaceTime HD camera via AVFoundation.
  - Custom local video files via open file dialog.
- **Export & Capture**:
  - 1-click HD snapshot capture.
  - Video recording toggle (`● REC`) with live timer.

---

## 🚀 Quick Start

### Global Command
Once installed, launch the studio from anywhere in your terminal:

```bash
vision3d
```

### Options
```bash
# Start directly on FaceTime HD / USB webcam
vision3d --webcam

# Analyze a specific video file
vision3d /path/to/movement.mp4
```

---

## 🛠 Architecture & Tech Stack

| Component | Technology | Rationale |
|---|---|---|
| **GUI Framework** | PySide6 (Qt for Python) | Native macOS dark zinc design, high-DPI scaling, responsive threading |
| **3D Rendering** | PyOpenGL & OpenGL 2.1/Core | Hardware-accelerated 3D mannequin rendering with dual studio lighting |
| **Pose Estimation** | MediaPipe 3D Landmarker (`pose_landmarker_full.task`) | Sub-20ms 3D metric world coordinates $(X, Y, Z)$ on Apple M4 |
| **Video Engine** | OpenCV (`cv2.CAP_AVFOUNDATION`) | Low-latency hardware-accelerated video decoding and camera capture |

---

## 📦 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/eliottrouvier/vision-3d.git
   cd vision-3d
   ```

2. **Create virtual environment & install dependencies**:
   ```bash
   uv venv .venv
   source .venv/bin/activate
   uv pip install PySide6 PyOpenGL mediapipe opencv-python numpy pillow
   ```

3. **Install the CLI command**:
   ```bash
   chmod +x main.py vision3d_launcher.sh
   ln -sf "$(pwd)/vision3d_launcher.sh" ~/.local/bin/vision3d
   ```

---

## 🎮 Keyboard & Mouse Shortcuts

- `Space`: Play / Pause playback
- `Left Click + Drag (3D View)`: Rotate 360° around avatar
- `Right Click + Drag (3D View)`: Pan camera
- `Mouse Wheel (3D View)`: Zoom in / out
- `Double Click (3D View)`: Center and reset camera

---

## 📄 License
MIT License. Created by Eliott Rouvier.
