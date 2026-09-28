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

# Ensure local modules are found
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

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

    # Enter Qt Event Loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
