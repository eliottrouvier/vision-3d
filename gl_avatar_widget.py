"""
GLAvatarWidget — High-Precision 3D SMPL-Style Continuous Faceted Mesh & Avatar Viewport
----------------------------------------------------------------------------------------
PySide6 QOpenGLWidget rendering:
1. Continuous 3D Faceted Human Mesh (SMPL-lite) with anatomical contours
   (sculpted torso/ribcage, muscular deltoids/biceps/quads/calves, oriented head & feet)
2. Shading Modes: Faceted Mesh (sharp CAD/clay polygons) vs Smooth Skin (Gouraud/Phong)
3. Full Zoom & Viewport Interaction:
   - On-screen floating HUD zoom buttons ([+], [-], [⟲ Reset])
   - MacBook trackpad pinch-to-zoom (NativeGesture)
   - Mouse wheel proportional zoom
   - 360° Mouse orbit (Left-drag rotate, Right-drag pan, Double-click reset)
   - Keyboard shortcuts (+, -, 0)
"""

import math
import numpy as np

from PySide6.QtCore import Qt, QPoint, QEvent
from PySide6.QtWidgets import QWidget, QHBoxLayout, QPushButton, QToolTip
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtGui import QNativeGestureEvent, QFont, QCursor
from OpenGL import GL


# Major joint indices
MAJOR_JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# 3D Bones skeleton connections
SKELETON_PAIRS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 31), (28, 32), (27, 29), (28, 30)
]


def _ortho_basis(v):
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32), np.array([0.0, 1.0, 0.0], dtype=np.float32)
    v_norm = v / norm
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float32) if abs(v_norm[2]) < 0.85 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
    u = np.cross(v_norm, ref)
    u = u / (np.linalg.norm(u) + 1e-8)
    w = np.cross(v_norm, u)
    return u.astype(np.float32), w.astype(np.float32)


class GLAvatarWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)

        # 3D Camera Orbit State
        self.camera_yaw = 20.0     # Horizontal angle in degrees
        self.camera_pitch = 12.0   # Vertical angle in degrees
        self.camera_dist = 2.4     # Distance from avatar in meters
        self.camera_target = np.array([0.0, 0.05, 0.0], dtype=np.float32)

        # Mouse interaction state
        self.last_mouse_pos = QPoint()

        # Display & Shading Modes
        self.show_mannequin = True
        self.show_skeleton = False
        self.show_joints = False
        self.show_grid = True
        self.faceted_mode = True    # True = Faceted polygon surface; False = Smooth skin
        self.show_wireframe = True  # Overlay polygonal mesh lines

        # Spatial Grounding & Floor Reference
        self.foot_grounding = True
        self.smooth_ground_y = None
        self.smooth_center_x = 0.0
        self.smooth_center_z = 0.0

        # Floating HUD Zoom Toolbar (tactile buttons top-right)
        self._build_floating_toolbar()

    def reset_filter(self):
        self.smooth_ground_y = None
        self.smooth_center_x = 0.0
        self.smooth_center_z = 0.0

    def set_foot_grounding(self, enabled: bool):
        self.foot_grounding = enabled
        self.reset_filter()
        self.update()

    def _build_floating_toolbar(self):
        """Creates semi-transparent floating zoom controls overlay in the top-right corner."""
        self.overlay_toolbar = QWidget(self)
        layout = QHBoxLayout(self.overlay_toolbar)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        btn_style = """
            QPushButton {
                background-color: rgba(24, 24, 27, 0.78);
                color: #e4e4e7;
                border: 1px solid rgba(63, 63, 70, 0.65);
                border-radius: 6px;
                font-size: 13px;
                font-weight: bold;
                min-width: 30px;
                max-width: 30px;
                min-height: 28px;
                max-height: 28px;
            }
            QPushButton:hover {
                background-color: rgba(39, 39, 42, 0.95);
                border-color: #a1a1aa;
                color: #ffffff;
            }
            QPushButton:pressed {
                background-color: #52525b;
            }
        """

        self.btn_zoom_in = QPushButton("+", self.overlay_toolbar)
        self.btn_zoom_in.setToolTip("Zoom avant (+ ou molette)")
        self.btn_zoom_in.setStyleSheet(btn_style)
        self.btn_zoom_in.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_zoom_in.clicked.connect(self.zoom_in)
        layout.addWidget(self.btn_zoom_in)

        self.btn_zoom_out = QPushButton("−", self.overlay_toolbar)
        self.btn_zoom_out.setToolTip("Zoom arrière (- ou molette)")
        self.btn_zoom_out.setStyleSheet(btn_style)
        self.btn_zoom_out.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_zoom_out.clicked.connect(self.zoom_out)
        layout.addWidget(self.btn_zoom_out)

        self.btn_reset = QPushButton("⟲", self.overlay_toolbar)
        self.btn_reset.setToolTip("Recentrer la caméra 3D (0 ou Double-clic)")
        self.btn_reset.setStyleSheet(btn_style)
        self.btn_reset.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_reset.clicked.connect(self.reset_camera)
        layout.addWidget(self.btn_reset)

        self.overlay_toolbar.adjustSize()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Position toolbar in top-right corner
        if hasattr(self, "overlay_toolbar"):
            tb_w = self.overlay_toolbar.width()
            tb_h = self.overlay_toolbar.height()
            self.overlay_toolbar.move(self.width() - tb_w - 12, 12)

    def set_landmarks_3d(self, pts_world, vis=None):
        if pts_world is not None:
            self.pts_world = pts_world.copy()
            self.vis = vis.copy() if vis is not None else np.ones(len(pts_world))
            self.update()

    def set_shading_mode(self, faceted: bool):
        self.faceted_mode = faceted
        self.update()

    def set_show_wireframe(self, show: bool):
        self.show_wireframe = show
        self.update()

    # ==============================================================
    # ZOOM & CAMERA CONTROLS
    # ==============================================================
    def zoom_in(self, factor=0.82):
        self.camera_dist = max(0.35, min(12.0, self.camera_dist * factor))
        self.update()

    def zoom_out(self, factor=1.22):
        self.camera_dist = max(0.35, min(12.0, self.camera_dist * factor))
        self.update()

    def reset_camera(self):
        self.camera_yaw = 20.0
        self.camera_pitch = 12.0
        self.camera_dist = 2.4
        self.camera_target = np.array([0.0, 0.05, 0.0], dtype=np.float32)
        self.update()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta == 0:
            delta = event.pixelDelta().y()
        factor = 0.90 if delta > 0 else 1.10
        self.camera_dist = max(0.35, min(12.0, self.camera_dist * factor))
        self.update()

    def event(self, e):
        # Native MacBook Trackpad Pinch-to-Zoom
        if e.type() == QEvent.Type.NativeGesture:
            if hasattr(e, "gestureType") and e.gestureType() == Qt.NativeGestureType.ZoomNativeGesture:
                scale_factor = 1.0 - e.value() * 0.75
                self.camera_dist = max(0.35, min(12.0, self.camera_dist * scale_factor))
                self.update()
                return True
        return super().event(e)

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key_Plus, Qt.Key_Equal):
            self.zoom_in()
        elif key in (Qt.Key_Minus, Qt.Key_Underscore):
            self.zoom_out()
        elif key in (Qt.Key_0, Qt.Key_Home, Qt.Key_R):
            self.reset_camera()
        else:
            super().keyPressEvent(event)

    def mousePressEvent(self, event):
        self.last_mouse_pos = event.position().toPoint()

    def mouseMoveEvent(self, event):
        cur_pos = event.position().toPoint()
        dx = cur_pos.x() - self.last_mouse_pos.x()
        dy = cur_pos.y() - self.last_mouse_pos.y()

        if event.buttons() & Qt.LeftButton:
            # Orbit rotation
            self.camera_yaw += dx * 0.55
            self.camera_pitch = max(-85.0, min(85.0, self.camera_pitch + dy * 0.55))
            self.update()
        elif event.buttons() & Qt.RightButton or (event.buttons() & Qt.LeftButton and (event.modifiers() & Qt.ShiftModifier)):
            # Pan camera target
            self.camera_target[0] -= dx * 0.003
            self.camera_target[1] += dy * 0.003
            self.update()

        self.last_mouse_pos = cur_pos

    def mouseDoubleClickEvent(self, event):
        self.reset_camera()

    # ==============================================================
    # OPENGL RENDERING PIPELINE
    # ==============================================================
    def initializeGL(self):
        GL.glClearColor(0.06, 0.06, 0.08, 1.0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LEQUAL)
        GL.glDisable(GL.GL_CULL_FACE) # Allow viewing from inside or back

        GL.glShadeModel(GL.GL_SMOOTH)
        GL.glEnable(GL.GL_LINE_SMOOTH)
        GL.glHint(GL.GL_LINE_SMOOTH_HINT, GL.GL_NICEST)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

        # Studio Lighting
        GL.glEnable(GL.GL_LIGHTING)
        GL.glEnable(GL.GL_LIGHT0)
        GL.glEnable(GL.GL_LIGHT1)
        GL.glEnable(GL.GL_NORMALIZE)
        GL.glEnable(GL.GL_COLOR_MATERIAL)
        GL.glColorMaterial(GL.GL_FRONT_AND_BACK, GL.GL_AMBIENT_AND_DIFFUSE)

        # Key Light
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, [2.5, 3.5, 3.0, 1.0])
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_DIFFUSE, [0.95, 0.95, 1.0, 1.0])
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_SPECULAR, [0.45, 0.45, 0.55, 1.0])

        # Fill Light
        GL.glLightfv(GL.GL_LIGHT1, GL.GL_POSITION, [-2.5, 1.5, -2.5, 1.0])
        GL.glLightfv(GL.GL_LIGHT1, GL.GL_DIFFUSE, [0.38, 0.40, 0.48, 1.0])

        GL.glLightModelfv(GL.GL_LIGHT_MODEL_AMBIENT, [0.22, 0.22, 0.27, 1.0])

    def resizeGL(self, w, h):
        GL.glViewport(0, 0, w, max(1, h))
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        aspect = float(w) / float(max(1, h))
        self._perspective(42.0, aspect, 0.1, 50.0)
        GL.glMatrixMode(GL.GL_MODELVIEW)

    def _perspective(self, fov_deg, aspect, z_near, z_far):
        f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
        m = [
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (z_far + z_near) / (z_near - z_far), -1,
            0, 0, (2 * z_far * z_near) / (z_near - z_far), 0
        ]
        GL.glMultMatrixf(m)

    def paintGL(self):
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()

        # Orbit camera matrix
        rad_yaw = math.radians(self.camera_yaw)
        rad_pitch = math.radians(self.camera_pitch)
        cam_x = self.camera_target[0] + self.camera_dist * math.sin(rad_yaw) * math.cos(rad_pitch)
        cam_y = self.camera_target[1] + self.camera_dist * math.sin(rad_pitch)
        cam_z = self.camera_target[2] + self.camera_dist * math.cos(rad_yaw) * math.cos(rad_pitch)

        self._look_at(cam_x, cam_y, cam_z,
                      self.camera_target[0], self.camera_target[1], self.camera_target[2],
                      0.0, 1.0, 0.0)

        # 1. Ground Grid & Pedestal
        if self.show_grid:
            self._draw_ground_grid()

        # 2. Render Continuous SMPL Human Mesh
        if self.pts_world is not None:
            self._render_smpl_avatar(self.pts_world)
        else:
            self._render_idle_smpl_avatar()

    def _look_at(self, eye_x, eye_y, eye_z, target_x, target_y, target_z, up_x, up_y, up_z):
        f = np.array([target_x - eye_x, target_y - eye_y, target_z - eye_z], dtype=np.float32)
        f = f / max(1e-6, np.linalg.norm(f))
        up = np.array([up_x, up_y, up_z], dtype=np.float32)
        up = up / max(1e-6, np.linalg.norm(up))
        s = np.cross(f, up)
        s = s / max(1e-6, np.linalg.norm(s))
        u = np.cross(s, f)

        m = [
             s[0],  u[0], -f[0], 0,
             s[1],  u[1], -f[1], 0,
             s[2],  u[2], -f[2], 0,
            -np.dot(s, [eye_x, eye_y, eye_z]),
            -np.dot(u, [eye_x, eye_y, eye_z]),
             np.dot(f, [eye_x, eye_y, eye_z]), 1.0
        ]
        GL.glMultMatrixf(m)

    def _draw_ground_grid(self):
        GL.glDisable(GL.GL_LIGHTING)
        GL.glLineWidth(1.0)
        grid_size = 2.2
        step = 0.22
        y_pos = -0.95

        GL.glBegin(GL.GL_LINES)
        n_lines = int(grid_size / step)
        for i in range(-n_lines, n_lines + 1):
            coord = i * step
            if i == 0:
                GL.glColor4f(0.25, 0.30, 0.40, 0.65)
            else:
                GL.glColor4f(0.12, 0.14, 0.18, 0.35)

            GL.glVertex3f(-grid_size, y_pos, coord)
            GL.glVertex3f(grid_size, y_pos, coord)
            GL.glVertex3f(coord, y_pos, -grid_size)
            GL.glVertex3f(coord, y_pos, grid_size)
        GL.glEnd()

        # Pedestal rings
        GL.glBegin(GL.GL_LINE_LOOP)
        GL.glColor4f(0.0, 0.85, 0.65, 0.40)
        for k in range(40):
            ang = 2 * math.pi * k / 40
            GL.glVertex3f(0.50 * math.cos(ang), y_pos, 0.50 * math.sin(ang))
        GL.glEnd()

        GL.glEnable(GL.GL_LIGHTING)

    # ==============================================================
    # SMPL-LITE CONTINUOUS FACETED 3D MESH GENERATION & RENDERING
    # ==============================================================
    def _render_smpl_avatar(self, raw_pts):
        """Builds and renders the continuous multi-ring anatomical SMPL mesh with spatial grounding."""
        pts = raw_pts.copy()
        pts[:, 1] = -pts[:, 1]  # Y up in OpenGL
        pts[:, 2] = -pts[:, 2]  # Z depth alignment

        # Smooth horizontal centering (X, Z) so the avatar stays in frame without lateral jerking
        raw_cx = float((pts[23, 0] + pts[24, 0]) * 0.5)
        raw_cz = float((pts[23, 2] + pts[24, 2]) * 0.5)
        if abs(self.smooth_center_x) < 1e-4 and abs(self.smooth_center_z) < 1e-4:
            self.smooth_center_x = raw_cx
            self.smooth_center_z = raw_cz
        else:
            self.smooth_center_x = 0.85 * self.smooth_center_x + 0.15 * raw_cx
            self.smooth_center_z = 0.85 * self.smooth_center_z + 0.15 * raw_cz

        pts[:, 0] -= self.smooth_center_x
        pts[:, 2] -= self.smooth_center_z

        if self.foot_grounding:
            # Intelligent Foot Grounding:
            # Anchor lowest supporting foot contact point on the ground grid (Y = -0.92)
            foot_indices = [27, 28, 29, 30, 31, 32]
            lowest_foot_y = float(np.min(pts[foot_indices, 1]))
            ground_plane_y = -0.92

            target_offset_y = ground_plane_y - lowest_foot_y
            if self.smooth_ground_y is None:
                self.smooth_ground_y = target_offset_y
            else:
                # Smooth low-pass filtering on ground contact so feet don't jitter
                self.smooth_ground_y = 0.85 * self.smooth_ground_y + 0.15 * target_offset_y

            pts[:, 1] += self.smooth_ground_y
        else:
            # Standard Hip Centering
            hip_y = (pts[23, 1] + pts[24, 1]) * 0.5
            pts[:, 1] -= hip_y

        triangles = []  # List of ((v0, v1, v2), normal)
        edges = []      # List of (p1, p2) for wireframe

        # ----------------------------------------------------------
        # 1. CONTINUOUS LOFTED TORSO & PELVIS (7 cross-sectional rings)
        # ----------------------------------------------------------
        s_mid = (pts[11] + pts[12]) * 0.5
        h_mid = (pts[23] + pts[24]) * 0.5
        v_spine = s_mid - h_mid
        h_torso = np.linalg.norm(v_spine)
        u_spine = v_spine / max(1e-4, h_torso)

        u_lat = pts[11] - pts[12]
        u_lat = u_lat / max(1e-4, np.linalg.norm(u_lat))
        u_ant = np.cross(u_lat, u_spine)
        u_ant = u_ant / max(1e-4, np.linalg.norm(u_ant))
        u_lat = np.cross(u_spine, u_ant)

        w_hips = max(0.18, float(np.linalg.norm(pts[23] - pts[24])))
        w_shoulders = max(0.24, float(np.linalg.norm(pts[11] - pts[12])))

        torso_levels = [
            # (t, rx, rz, forward_offset)
            (-0.06, w_hips * 0.44, 0.090, 0.00),  # Pelvis floor / Crotch
            (0.15,  w_hips * 0.52, 0.105, 0.00),  # Iliac crest / Lower pelvis
            (0.38,  w_hips * 0.44, 0.095, -0.01), # Waist / Navel
            (0.62,  w_shoulders * 0.45, 0.118, 0.01), # Lower ribcage & sternum
            (0.84,  w_shoulders * 0.52, 0.138, 0.02), # Mid-chest pectorals
            (0.98,  w_shoulders * 0.54, 0.112, 0.00), # Clavicles / Shoulders line
            (1.08,  w_shoulders * 0.28, 0.075, 0.00), # Neck base
        ]

        torso_rings = []
        n_torso = 12
        thetas_t = np.linspace(0, 2 * np.pi, n_torso, endpoint=False)
        for t, rx, rz, f_off in torso_levels:
            c = h_mid + t * v_spine + f_off * u_ant
            ring = [c + rx * math.cos(th) * u_lat + rz * math.sin(th) * u_ant for th in thetas_t]
            torso_rings.append(ring)

        self._loft_rings(torso_rings, triangles, edges, cap_bottom=True, cap_top=False)

        # ----------------------------------------------------------
        # 2. SCULPTED HEAD & NECK
        # ----------------------------------------------------------
        nose = pts[0]
        ears_mid = (pts[7] + pts[8]) * 0.5
        head_dir = nose - s_mid
        head_dir = head_dir / max(1e-4, np.linalg.norm(head_dir))

        u_head_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        u_head_lat = np.cross(u_head_up, head_dir)
        u_head_lat = u_head_lat / max(1e-4, np.linalg.norm(u_head_lat))
        u_head_up = np.cross(head_dir, u_head_lat)

        head_base = s_mid + u_spine * 0.08
        head_center = s_mid + u_spine * 0.26 + head_dir * 0.03

        # Neck cylinder rings
        neck_rings = []
        for t in [0.0, 0.5, 1.0]:
            nc = head_base + t * (head_center - head_base) * 0.7
            n_r = 0.055
            ring = [nc + n_r * math.cos(th) * u_head_lat + n_r * math.sin(th) * head_dir for th in thetas_t]
            neck_rings.append(ring)
        self._loft_rings(neck_rings, triangles, edges, cap_bottom=False, cap_top=False)

        # Sculpted faceted head (Cranium + Brow + Jaw/Chin)
        head_levels = [
            (-0.10, 0.065, 0.075, 0.04), # Chin & Jaw apex
            (0.00,  0.085, 0.095, 0.03), # Mouth & Cheeks
            (0.08,  0.095, 0.105, 0.01), # Eyes & Brow ridge
            (0.16,  0.090, 0.095, -0.01),# Forehead & Temples
            (0.22,  0.060, 0.065, -0.02),# Cranium top
        ]
        head_rings = []
        for y_off, rx, rz, z_off in head_levels:
            hc = head_center + y_off * u_head_up + z_off * head_dir
            ring = [hc + rx * math.cos(th) * u_head_lat + rz * math.sin(th) * head_dir for th in thetas_t]
            head_rings.append(ring)
        self._loft_rings(head_rings, triangles, edges, cap_bottom=True, cap_top=True)

        # ----------------------------------------------------------
        # 3. CONTINUOUS LIMBS WITH ANATOMICAL MUSCULAR CONTOURS
        # ----------------------------------------------------------
        # Deltoids / Shoulder caps
        self._add_joint_capsule(pts[11], 0.065, triangles, edges)
        self._add_joint_capsule(pts[12], 0.065, triangles, edges)

        # Left Arm: Shoulder (11) -> Elbow (13) -> Wrist (15) -> Hand (17)
        self._add_continuous_arm(pts[11], pts[13], pts[15], pts[17], triangles, edges)
        # Right Arm: Shoulder (12) -> Elbow (14) -> Wrist (16) -> Hand (18)
        self._add_continuous_arm(pts[12], pts[14], pts[16], pts[18], triangles, edges)

        # Left Leg: Hip (23) -> Knee (25) -> Ankle (27) -> Foot (31, 29)
        self._add_continuous_leg(pts[23], pts[25], pts[27], pts[31], pts[29], triangles, edges)
        # Right Leg: Hip (24) -> Knee (26) -> Ankle (28) -> Foot (32, 30)
        self._add_continuous_leg(pts[24], pts[26], pts[28], pts[32], pts[30], triangles, edges)

        # ----------------------------------------------------------
        # 4. RENDER SHADED FACETED / SMOOTH MESH
        # ----------------------------------------------------------
        # Clay Material Settings (Matte titanium gray with subtle specular)
        GL.glColor4f(0.80, 0.82, 0.86, 1.0)
        GL.glMaterialfv(GL.GL_FRONT_AND_BACK, GL.GL_SPECULAR, [0.35, 0.35, 0.40, 1.0])
        GL.glMaterialf(GL.GL_FRONT_AND_BACK, GL.GL_SHININESS, 20.0)

        # Draw Solid Mesh
        GL.glBegin(GL.GL_TRIANGLES)
        for (v0, v1, v2), n in triangles:
            GL.glNormal3f(n[0], n[1], n[2])
            GL.glVertex3f(v0[0], v0[1], v0[2])
            GL.glVertex3f(v1[0], v1[1], v1[2])
            GL.glVertex3f(v2[0], v2[1], v2[2])
        GL.glEnd()

        # ----------------------------------------------------------
        # 5. WIREFRAME EDGES OVERLAY (If enabled)
        # ----------------------------------------------------------
        if self.show_wireframe:
            GL.glDisable(GL.GL_LIGHTING)
            GL.glEnable(GL.GL_POLYGON_OFFSET_LINE)
            GL.glPolygonOffset(-1.0, -1.0)
            GL.glLineWidth(1.1)

            # High-contrast slate cyan/zinc wireframe lines
            GL.glColor4f(0.20, 0.25, 0.32, 0.55)
            GL.glBegin(GL.GL_LINES)
            for p1, p2 in edges:
                GL.glVertex3f(p1[0], p1[1], p1[2])
                GL.glVertex3f(p2[0], p2[1], p2[2])
            GL.glEnd()

            GL.glDisable(GL.GL_POLYGON_OFFSET_LINE)
            GL.glEnable(GL.GL_LIGHTING)

        # ----------------------------------------------------------
        # 6. INTERIOR SKELETON (Optional)
        # ----------------------------------------------------------
        if self.show_skeleton:
            GL.glDisable(GL.GL_LIGHTING)
            GL.glLineWidth(2.5)
            GL.glColor4f(0.0, 1.0, 0.55, 0.90) # Lime green
            GL.glBegin(GL.GL_LINES)
            for j1, j2 in SKELETON_PAIRS:
                GL.glVertex3f(pts[j1][0], pts[j1][1], pts[j1][2])
                GL.glVertex3f(pts[j2][0], pts[j2][1], pts[j2][2])
            GL.glEnd()
            GL.glEnable(GL.GL_LIGHTING)

    def _loft_rings(self, rings, triangles, edges, cap_bottom=False, cap_top=False):
        """Lofts a sequence of cross-sectional rings into connected quad/triangle facets."""
        n_rings = len(rings)
        if n_rings < 2:
            return
        n_pts = len(rings[0])

        for r in range(n_rings - 1):
            r0 = rings[r]
            r1 = rings[r + 1]
            for i in range(n_pts):
                next_i = (i + 1) % n_pts
                v0 = r0[i]
                v1 = r0[next_i]
                v2 = r1[next_i]
                v3 = r1[i]

                # Triangle 1
                n1 = np.cross(v1 - v0, v2 - v0)
                norm1 = np.linalg.norm(n1)
                n1 = n1 / (norm1 + 1e-8)
                triangles.append(((v0, v1, v2), n1))

                # Triangle 2
                n2 = np.cross(v2 - v0, v3 - v0)
                norm2 = np.linalg.norm(n2)
                n2 = n2 / (norm2 + 1e-8)
                triangles.append(((v0, v2, v3), n2))

                # Edge contours
                edges.append((v0, v1))
                edges.append((v0, v3))
                edges.append((v1, v2))

        # Bottom cap
        if cap_bottom:
            c_bot = np.mean(rings[0], axis=0)
            n_bot = np.array([0.0, -1.0, 0.0], dtype=np.float32)
            for i in range(n_pts):
                next_i = (i + 1) % n_pts
                triangles.append(((c_bot, rings[0][next_i], rings[0][i]), n_bot))
                edges.append((c_bot, rings[0][i]))

        # Top cap
        if cap_top:
            c_top = np.mean(rings[-1], axis=0)
            n_top = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            for i in range(n_pts):
                next_i = (i + 1) % n_pts
                triangles.append(((c_top, rings[-1][i], rings[-1][next_i]), n_top))
                edges.append((c_top, rings[-1][i]))

    def _add_continuous_arm(self, p_shoulder, p_elbow, p_wrist, p_hand, triangles, edges):
        """Constructs an arm with bicep bulge, articulated elbow, forearm and hand paddle."""
        v_upper = p_elbow - p_shoulder
        v_forearm = p_wrist - p_elbow
        u1, w1 = _ortho_basis(v_upper)
        u2, w2 = _ortho_basis(v_forearm)

        n_seg = 10
        thetas = np.linspace(0, 2 * np.pi, n_seg, endpoint=False)

        # Multi-ring arm progression: Shoulder -> Bicep -> Elbow -> Forearm -> Wrist
        arm_specs = [
            # (center, radius_u, radius_w, u_basis, w_basis)
            (p_shoulder, 0.055, 0.050, u1, w1),
            (p_shoulder + 0.40 * v_upper, 0.052, 0.048, u1, w1), # Biceps peak
            (p_shoulder + 0.80 * v_upper, 0.042, 0.038, u1, w1), # Lower bicep
            (p_elbow, 0.038, 0.035, (u1 + u2) * 0.5, (w1 + w2) * 0.5), # Elbow joint
            (p_elbow + 0.35 * v_forearm, 0.040, 0.034, u2, w2), # Upper forearm bulge
            (p_elbow + 0.75 * v_forearm, 0.032, 0.026, u2, w2), # Mid-forearm taper
            (p_wrist, 0.026, 0.018, u2, w2),                    # Wrist oval
        ]

        arm_rings = []
        for c, ru, rw, ub, wb in arm_specs:
            ring = [c + ru * math.cos(th) * ub + rw * math.sin(th) * wb for th in thetas]
            arm_rings.append(ring)
        self._loft_rings(arm_rings, triangles, edges, cap_bottom=False, cap_top=False)

        # Hand Paddle
        v_hand = p_hand - p_wrist
        h_len = np.linalg.norm(v_hand)
        if h_len > 0.03:
            h_dir = v_hand / h_len
            palm_center = p_wrist + h_dir * 0.05
            hand_rings = [
                arm_rings[-1],
                [palm_center + 0.028 * math.cos(th) * u2 + 0.014 * math.sin(th) * w2 for th in thetas],
                [palm_center + h_dir * 0.04 + 0.020 * math.cos(th) * u2 + 0.008 * math.sin(th) * w2 for th in thetas],
            ]
            self._loft_rings(hand_rings, triangles, edges, cap_bottom=False, cap_top=True)

    def _add_continuous_leg(self, p_hip, p_knee, p_ankle, p_toe, p_heel, triangles, edges):
        """Constructs a leg with quadriceps bulge, patella knee, gastrocnemius calf and foot."""
        v_thigh = p_knee - p_hip
        v_shin = p_ankle - p_knee
        u1, w1 = _ortho_basis(v_thigh)
        u2, w2 = _ortho_basis(v_shin)

        n_seg = 10
        thetas = np.linspace(0, 2 * np.pi, n_seg, endpoint=False)

        # Leg multi-ring progression: Hip -> Quad -> Knee -> Calf -> Ankle
        leg_specs = [
            (p_hip, 0.088, 0.080, u1, w1),
            (p_hip + 0.35 * v_thigh, 0.082, 0.076, u1, w1), # Upper thigh
            (p_hip + 0.70 * v_thigh, 0.068, 0.062, u1, w1), # Mid quad
            (p_knee, 0.052, 0.048, (u1 + u2) * 0.5, (w1 + w2) * 0.5), # Knee / Patella
            (p_knee + 0.30 * v_shin, 0.060, 0.054, u2, w2), # Calf bulge (Gastrocnemius)
            (p_knee + 0.70 * v_shin, 0.045, 0.040, u2, w2), # Lower calf taper
            (p_ankle, 0.038, 0.034, u2, w2),                # Ankle ring
        ]

        leg_rings = []
        for c, ru, rw, ub, wb in leg_specs:
            ring = [c + ru * math.cos(th) * ub + rw * math.sin(th) * wb for th in thetas]
            leg_rings.append(ring)
        self._loft_rings(leg_rings, triangles, edges, cap_bottom=False, cap_top=False)

        # Anatomical Foot Wedge
        v_toe = p_toe - p_ankle
        v_heel = p_heel - p_ankle
        foot_fwd = v_toe / max(1e-4, np.linalg.norm(v_toe))
        foot_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        foot_lat = np.cross(foot_up, foot_fwd)
        foot_lat = foot_lat / max(1e-4, np.linalg.norm(foot_lat))

        foot_rings = [
            leg_rings[-1],
            [p_ankle + foot_fwd * 0.06 - foot_up * 0.03 + 0.038 * math.cos(th) * foot_lat + 0.022 * math.sin(th) * foot_up for th in thetas],
            [p_ankle + foot_fwd * 0.13 - foot_up * 0.05 + 0.034 * math.cos(th) * foot_lat + 0.014 * math.sin(th) * foot_up for th in thetas],
        ]
        self._loft_rings(foot_rings, triangles, edges, cap_bottom=False, cap_top=True)

    def _add_joint_capsule(self, center, radius, triangles, edges):
        """Adds a smooth spherical capsule at a joint node."""
        lats, lons = 6, 8
        rings = []
        for i in range(lats + 1):
            lat = math.pi * (-0.5 + float(i) / lats)
            y = radius * math.sin(lat)
            r = radius * math.cos(lat)
            ring = []
            for j in range(lons):
                lng = 2 * math.pi * float(j) / lons
                x = r * math.cos(lng)
                z = r * math.sin(lng)
                ring.append(center + np.array([x, y, z], dtype=np.float32))
            rings.append(ring)
        self._loft_rings(rings, triangles, edges, cap_bottom=True, cap_top=True)

    def _render_idle_smpl_avatar(self):
        """Renders an idle athletic A-pose avatar when no live tracking input is detected."""
        pts = np.zeros((33, 3), dtype=np.float32)
        # Hips & Shoulders
        pts[23] = [-0.14, 0.0, 0.0]
        pts[24] = [0.14, 0.0, 0.0]
        pts[11] = [-0.21, 0.52, 0.0]
        pts[12] = [0.21, 0.52, 0.0]

        # Arms in athletic A-pose
        pts[13] = [-0.36, 0.28, 0.02]
        pts[14] = [0.36, 0.28, 0.02]
        pts[15] = [-0.46, 0.04, 0.04]
        pts[16] = [0.46, 0.04, 0.04]
        pts[17] = [-0.50, -0.06, 0.04]
        pts[18] = [0.50, -0.06, 0.04]

        # Legs standing
        pts[25] = [-0.15, -0.45, 0.01]
        pts[26] = [0.15, -0.45, 0.01]
        pts[27] = [-0.16, -0.88, 0.0]
        pts[28] = [0.16, -0.88, 0.0]
        pts[29] = [-0.16, -0.92, -0.06]
        pts[30] = [0.16, -0.92, -0.06]
        pts[31] = [-0.16, -0.92, 0.12]
        pts[32] = [0.16, -0.92, 0.12]

        # Head
        pts[0] = [0.0, 0.72, 0.04]
        pts[7] = [-0.08, 0.74, -0.04]
        pts[8] = [0.08, 0.74, -0.04]

        # Invert Y and Z as _render_smpl_avatar expects raw MediaPipe coordinates
        pts[:, 1] = -pts[:, 1]
        pts[:, 2] = -pts[:, 2]

        self._render_smpl_avatar(pts)
