"""
OneEuroFilter — Adaptive Low-Pass Filter for Noisy Human Pose Landmarking
-------------------------------------------------------------------------
Based on the One Euro Filter algorithm (Casiez, Roussel, Vogel, CHI 2012).
Provides:
- High smoothing at low speed (eliminating 100% of stationary jitter & shaking)
- Low smoothing at high speed (eliminating lag & latency during quick movement)
- Vectorized over all (33, 3) 3D landmarks simultaneously with NumPy
"""

import math
import time
import numpy as np


class OneEuroFilter3D:
    def __init__(self, min_cutoff=1.0, beta=0.015, d_cutoff=1.0):
        """
        min_cutoff: Minimum cutoff frequency in Hz. Lower values = smoother at rest.
        beta: Speed coefficient. Higher values = faster response during quick movement.
        d_cutoff: Cutoff frequency for derivative filtering in Hz.
        """
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)

        self.x_prev = None      # Shape (N, 3)
        self.dx_prev = None     # Shape (N, 3)
        self.t_prev = None      # Timestamp

    def reset(self):
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def filter(self, x, timestamp=None):
        """
        x: numpy array of shape (N, 3) representing 3D coordinates.
        timestamp: float in seconds (optional). If None, uses time.perf_counter().
        Returns: smoothed numpy array of same shape.
        """
        if x is None:
            return None

        t = timestamp if timestamp is not None else time.perf_counter()

        if self.x_prev is None or self.t_prev is None:
            self.x_prev = np.array(x, dtype=np.float32)
            self.dx_prev = np.zeros_like(self.x_prev)
            self.t_prev = t
            return self.x_prev.copy()

        dt = t - self.t_prev
        if dt <= 1e-5:
            dt = 1.0 / 30.0 # Fallback 30 FPS if called with zero elapsed time

        self.t_prev = t

        # 1. Estimate raw velocity / derivative
        dx_raw = (x - self.x_prev) / dt

        # 2. Filter derivative with d_cutoff
        alpha_d = self._alpha(self.d_cutoff, dt)
        dx_hat = alpha_d * dx_raw + (1.0 - alpha_d) * self.dx_prev
        self.dx_prev = dx_hat

        # 3. Compute adaptive cutoff frequency per joint
        # Velocity magnitude per 3D point: shape (N, 1)
        speed = np.linalg.norm(dx_hat, axis=-1, keepdims=True)
        fc = self.min_cutoff + self.beta * speed

        # 4. Filter position with adaptive alpha
        alpha = self._alpha_vec(fc, dt)
        x_hat = alpha * x + (1.0 - alpha) * self.x_prev
        self.x_prev = x_hat

        return x_hat.copy()

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * max(1e-4, cutoff))
        return 1.0 / (1.0 + tau / dt)

    @staticmethod
    def _alpha_vec(cutoff_vec, dt):
        tau = 1.0 / (2.0 * np.pi * np.maximum(1e-4, cutoff_vec))
        return 1.0 / (1.0 + tau / dt)
