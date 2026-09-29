#!/usr/bin/env python3
"""
Vision-3D — 3D Human Mesh Recovery & Real-Time Avatar Studio
============================================================
Executable entry point for the Vision-3D desktop application.

Usage:
    vision3d                      # Start in demo mode (Squats Kinematics)
    vision3d --webcam             # Start directly with FaceTime HD / USB camera
    vision3d path/to/video.mp4    # Start with a specific video file
"""

import sys
import os
import signal
import argparse

# Fix for Qt plugin loader when workspace path contains special characters or colons
qt_share_plugins = os.path.expanduser("~/.local/share/vision3d/plugins")
qt_share_platforms = os.path.join(qt_share_plugins, "platforms")
if not os.path.exists(os.path.join(qt_share_platforms, "libqcocoa.dylib")):
    try:
        import PySide6
        src_plugins = os.path.join(os.path.dirname(PySide6.__file__), "Qt", "plugins")
        if os.path.exists(src_plugins):
            import shutil
            os.makedirs(qt_share_plugins, exist_ok=True)
            shutil.copytree(src_plugins, qt_share_plugins, dirs_exist_ok=True)
    except Exception:
        pass

if os.path.exists(qt_share_plugins):
    os.environ["QT_PLUGIN_PATH"] = qt_share_plugins
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = qt_share_platforms

from PySide6.QtCore import QCoreApplication
if os.path.exists(qt_share_plugins):
    QCoreApplication.addLibraryPath(qt_share_plugins)

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
from PySide6.QtGui import QIcon

from main_window import MainWindow


def parse_args():
    parser = argparse.ArgumentParser(
        description="Vision-3D — 3D Human Mesh Recovery & Real-Time Avatar Studio",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "video",
        nargs="?",
        default=None,
        help="Optional path to a video file to analyze on startup"
    )
    parser.add_argument(
        "-w", "--webcam",
        action="store_true",
        help="Start directly in live Webcam tracking mode"
    )
    parser.add_argument(
        "-s", "--source",
        type=str,
        default=None,
        help="Alternative flag to specify input video file"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine initial video source
    initial_source = args.source or args.video
    if initial_source:
        initial_source = os.path.abspath(initial_source)
        if not os.path.exists(initial_source):
            print(f"[!] Warning: Video file not found: {initial_source}")
            initial_source = None

    # Initialize Qt Application
    app = QApplication(sys.argv)
    app.setApplicationName("Vision-3D Studio")
    app.setOrganizationName("EliottRouvier")

    # Graceful handling of Ctrl+C in terminal
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    # A tiny timer keeps the Python interpreter active to process signals
    sig_timer = QTimer()
    sig_timer.start(500)
    sig_timer.timeout.connect(lambda: None)

    # Create and display MainWindow
    window = MainWindow(initial_source=initial_source, is_webcam=args.webcam)
    window.show()

    # Enter Qt Event Loop with clean worker thread shutdown
    try:
        sys.exit(app.exec())
    finally:
        if 'window' in locals() and hasattr(window, 'worker') and window.worker.isRunning():
            window.worker.stop()
            window.worker.wait(1000)


if __name__ == "__main__":
    main()
