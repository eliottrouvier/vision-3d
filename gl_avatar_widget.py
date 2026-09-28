"""
GLAvatarWidget — Interactive 3D Real-Time Humanoid Mannequin Avatar Viewport
----------------------------------------------------------------------------
Uses PySide6 QOpenGLWidget to render:
1. 3D Rigged / Skinned Humanoid Mannequin mirror driven by metric 3D landmarks
2. Ground grid & atmospheric directional lighting
3. 360° Mouse Orbit Camera (Left-drag to rotate, Right-drag to pan, Scroll to zoom)
"""

import math
import numpy as np
from PySide6.QtCore import Qt, QPoint
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL import GL

# Body segments: (joint1_idx, joint2_idx, radius_m)
MANNEQUIN_BONES = [
    # Spine & Torso
    ("spine_lower", 23, 24, 0.08), # Hips center
    ("spine_upper", 11, 12, 0.09), # Shoulders center
    # Arms
    ("upper_arm_l", 11, 13, 0.045),
    ("forearm_l", 13, 15, 0.035),
    ("upper_arm_r", 12, 14, 0.045),
    ("forearm_r", 14, 16, 0.035),
    # Legs
    ("thigh_l", 23, 25, 0.065),
    ("shin_l", 25, 27, 0.048),
    ("thigh_r", 24, 26, 0.065),
    ("shin_r", 26, 28, 0.048),
    # Feet
    ("foot_l", 27, 31, 0.040),
    ("foot_r", 28, 32, 0.040),
]

# Joint sphere indices
MAJOR_JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


class GLAvatarWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # 3D Camera Orbit State
        self.camera_yaw = 20.0     # Horizontal angle in degrees
        self.camera_pitch = 10.0   # Vertical angle in degrees
        self.camera_dist = 2.4     # Distance from avatar in meters
        self.camera_target = np.array([0.0, 0.1, 0.0], dtype=np.float32) # Look-at point

        # Mouse interaction state
        self.last_mouse_pos = QPoint()

        # Display Toggles
        self.show_mannequin = True
        self.show_skeleton = True
        self.show_joints = True
        self.show_grid = True
        self.wireframe_mode = False

        # Current 3D landmarks in meters: shape (33, 3)
        self.pts_world = None
        self.vis = None

        # Precomputed unit sphere & cylinder geometry for high FPS
        self._init_geometry_cache()

    def _init_geometry_cache(self):
        """Precomputes mesh vertices and normals for fast immediate-mode rendering."""
        # Unit Sphere (10 lat x 12 lon)
        lats, lons = 10, 12
        self.sphere_verts = []
        for i in range(lats):
            lat0 = math.pi * (-0.5 + float(i) / lats)
            z0 = math.sin(lat0)
            zr0 = math.cos(lat0)

            lat1 = math.pi * (-0.5 + float(i + 1) / lats)
            z1 = math.sin(lat1)
            zr1 = math.cos(lat1)

            quad_strip = []
            for j in range(lons + 1):
                lng = 2 * math.pi * float(j) / lons
                x = math.cos(lng)
                y = math.sin(lng)
                quad_strip.append((x * zr0, y * zr0, z0))
                quad_strip.append((x * zr1, y * zr1, z1))
            self.sphere_verts.append(quad_strip)

    def set_landmarks_3d(self, pts_world, vis=None):
        """
        Updates the 3D avatar pose.
        pts_world: numpy array of shape (33, 3) in real-world metric meters.
        """
        if pts_world is not None:
            self.pts_world = pts_world.copy()
            self.vis = vis.copy() if vis is not None else np.ones(len(pts_world))
            self.update()

    def initializeGL(self):
        GL.glClearColor(0.05, 0.05, 0.07, 1.0) # Sleek dark zinc (#0d0d12)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LEQUAL)
        GL.glEnable(GL.GL_CULL_FACE)
        GL.glCullFace(GL.GL_BACK)

        # Smooth shading & antialiasing
        GL.glShadeModel(GL.GL_SMOOTH)
        GL.glEnable(GL.GL_LINE_SMOOTH)
        GL.glHint(GL.GL_LINE_SMOOTH_HINT, GL.GL_NICEST)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

        # Professional 3D Studio Lighting
        GL.glEnable(GL.GL_LIGHTING)
        GL.glEnable(GL.GL_LIGHT0)
        GL.glEnable(GL.GL_LIGHT1)
        GL.glEnable(GL.GL_NORMALIZE)
        GL.glEnable(GL.GL_COLOR_MATERIAL)
        GL.glColorMaterial(GL.GL_FRONT_AND_BACK, GL.GL_AMBIENT_AND_DIFFUSE)

        # Key Light (Upper Front Right)
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, [2.0, 3.0, 3.0, 1.0])
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_DIFFUSE, [0.95, 0.95, 1.0, 1.0])
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_SPECULAR, [0.4, 0.4, 0.5, 1.0])

        # Fill Light (Lower Back Left)
        GL.glLightfv(GL.GL_LIGHT1, GL.GL_POSITION, [-2.5, 1.0, -2.5, 1.0])
        GL.glLightfv(GL.GL_LIGHT1, GL.GL_DIFFUSE, [0.35, 0.38, 0.45, 1.0])

        # Ambient baseline
        GL.glLightModelfv(GL.GL_LIGHT_MODEL_AMBIENT, [0.22, 0.22, 0.26, 1.0])

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

        # Camera transform from Orbit Angles
        rad_yaw = math.radians(self.camera_yaw)
        rad_pitch = math.radians(self.camera_pitch)

        cam_x = self.camera_target[0] + self.camera_dist * math.sin(rad_yaw) * math.cos(rad_pitch)
        cam_y = self.camera_target[1] + self.camera_dist * math.sin(rad_pitch)
        cam_z = self.camera_target[2] + self.camera_dist * math.cos(rad_yaw) * math.cos(rad_pitch)

        self._look_at(cam_x, cam_y, cam_z,
                      self.camera_target[0], self.camera_target[1], self.camera_target[2],
                      0.0, 1.0, 0.0)

        # 1. Ground Grid
        if self.show_grid:
            self._draw_ground_grid()

        # 2. Render Avatar if 3D pose is available
        if self.pts_world is not None:
            self._draw_humanoid_avatar()
        else:
            self._draw_idle_mannequin()

    def _look_at(self, eye_x, eye_y, eye_z, target_x, target_y, target_z, up_x, up_y, up_z):
        f = np.array([target_x - eye_x, target_y - eye_y, target_z - eye_z], dtype=np.float32)
        f = f / np.linalg.norm(f)
        up = np.array([up_x, up_y, up_z], dtype=np.float32)
        up = up / np.linalg.norm(up)
        s = np.cross(f, up)
        s = s / np.linalg.norm(s)
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

        # Fine dark zinc grid
        GL.glBegin(GL.GL_LINES)
        grid_size = 2.0
        step = 0.20
        y_pos = -0.95

        n_lines = int(grid_size / step)
        for i in range(-n_lines, n_lines + 1):
            coord = i * step
            # Accentuate center cross
            if i == 0:
                GL.glColor4f(0.25, 0.28, 0.35, 0.6)
            else:
                GL.glColor4f(0.12, 0.14, 0.18, 0.4)

            # Lines along X
            GL.glVertex3f(-grid_size, y_pos, coord)
            GL.glVertex3f(grid_size, y_pos, coord)

            # Lines along Z
            GL.glVertex3f(coord, y_pos, -grid_size)
            GL.glVertex3f(coord, y_pos, grid_size)
        GL.glEnd()

        # Circular pedestal disk under avatar
        GL.glBegin(GL.GL_LINE_LOOP)
        GL.glColor4f(0.0, 0.85, 0.6, 0.35)
        for k in range(36):
            ang = 2 * math.pi * k / 36
            GL.glVertex3f(0.45 * math.cos(ang), y_pos, 0.45 * math.sin(ang))
        GL.glEnd()

        GL.glEnable(GL.GL_LIGHTING)

    def _draw_humanoid_avatar(self):
        """Renders the shaded 3D mannequin avatar matching the screenshot."""
        # Convert MediaPipe 3D coordinates (Y down, Z forward) to OpenGL (Y up, Z out)
        pts = self.pts_world.copy()
        pts[:, 1] = -pts[:, 1] # Invert Y so up is positive
        pts[:, 2] = -pts[:, 2] # Align depth

        # Center hips at origin (0, 0, 0)
        hip_center = (pts[23] + pts[24]) * 0.5
        pts = pts - hip_center

        # Avatar Shading Material (Apple Pro Matte Gray Mannequin like screenshot)
        GL.glColor4f(0.82, 0.84, 0.88, 1.0)
        GL.glMaterialfv(GL.GL_FRONT_AND_BACK, GL.GL_SPECULAR, [0.35, 0.35, 0.4, 1.0])
        GL.glMaterialf(GL.GL_FRONT_AND_BACK, GL.GL_SHININESS, 25.0)

        # 1. Torso Volume (Pelvis to Chest & Shoulders)
        shoulders = (pts[11] + pts[12]) * 0.5
        hips = (pts[23] + pts[24]) * 0.5

        if self.show_mannequin:
            # Chest / Torso capsule
            self._draw_cylinder(hips, shoulders, 0.12, 0.15)
            # Pelvis belt
            self._draw_cylinder(pts[23], pts[24], 0.08, 0.08)
            # Shoulders bar
            self._draw_cylinder(pts[11], pts[12], 0.07, 0.07)

            # Head (3D Sphere)
            nose = pts[0]
            head_center = shoulders + (nose - shoulders) * 1.3
            self._draw_sphere(head_center, 0.11)

            # Neck
            neck_base = shoulders
            self._draw_cylinder(neck_base, head_center, 0.055, 0.055)

        # 2. Limbs Volumes (Upper Arms, Forearms, Thighs, Shins)
        limbs = [
            (11, 13, 0.045, 0.040), # Left Upper Arm
            (13, 15, 0.038, 0.030), # Left Forearm
            (12, 14, 0.045, 0.040), # Right Upper Arm
            (14, 16, 0.038, 0.030), # Right Forearm
            (23, 25, 0.075, 0.060), # Left Thigh
            (25, 27, 0.055, 0.042), # Left Shin
            (24, 26, 0.075, 0.060), # Right Thigh
            (26, 28, 0.055, 0.042), # Right Shin
            (27, 31, 0.040, 0.030), # Left Foot
            (28, 32, 0.040, 0.030), # Right Foot
        ]

        if self.show_mannequin:
            for p1_id, p2_id, r1, r2 in limbs:
                p1 = pts[p1_id]
                p2 = pts[p2_id]
                self._draw_cylinder(p1, p2, r1, r2)

        # 3. Joint Spheres (Nodes)
        if self.show_joints:
            GL.glColor4f(0.95, 0.95, 1.0, 1.0)
            for j_id in MAJOR_JOINTS:
                radius = 0.055 if j_id in [11, 12, 23, 24] else 0.045
                self._draw_sphere(pts[j_id], radius)

        # 4. Interior / Overlay 3D Skeleton Bones (Lime Green / Neon Cyan Lines)
        if self.show_skeleton:
            GL.glDisable(GL.GL_LIGHTING)
            GL.glLineWidth(2.5)
            GL.glBegin(GL.GL_LINES)
            GL.glColor4f(0.0, 1.0, 0.55, 0.85) # Lime neon skeleton inside avatar

            bone_pairs = [
                (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
                (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
                (24, 26), (26, 28), (27, 31), (28, 32)
            ]
            for p1_id, p2_id in bone_pairs:
                p1, p2 = pts[p1_id], pts[p2_id]
                GL.glVertex3f(p1[0], p1[1], p1[2])
                GL.glVertex3f(p2[0], p2[1], p2[2])
            GL.glEnd()
            GL.glEnable(GL.GL_LIGHTING)

    def _draw_sphere(self, center, radius):
        """Draws a smooth 3D sphere at center with given radius."""
        GL.glPushMatrix()
        GL.glTranslatef(float(center[0]), float(center[1]), float(center[2]))
        GL.glScalef(float(radius), float(radius), float(radius))

        for quad_strip in self.sphere_verts:
            GL.glBegin(GL.GL_QUAD_STRIP)
            for v in quad_strip:
                GL.glNormal3f(v[0], v[1], v[2])
                GL.glVertex3f(v[0], v[1], v[2])
            GL.glEnd()

        GL.glPopMatrix()

    def _draw_cylinder(self, p1, p2, r1, r2):
        """Draws a smooth truncated cone / cylinder between p1 and p2."""
        d = p2 - p1
        length = np.linalg.norm(d)
        if length < 1e-4:
            return

        d_norm = d / length
        u, v = self._ortho_basis(d_norm)

        n_sides = 12
        GL.glBegin(GL.GL_QUAD_STRIP)
        for i in range(n_sides + 1):
            theta = 2.0 * math.pi * float(i) / n_sides
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)

            normal = cos_t * u + sin_t * v
            GL.glNormal3f(normal[0], normal[1], normal[2])

            v1 = p1 + r1 * normal
            v2 = p2 + r2 * normal

            GL.glVertex3f(v1[0], v1[1], v1[2])
            GL.glVertex3f(v2[0], v2[1], v2[2])
        GL.glEnd()

    def _ortho_basis(self, v):
        ref = np.array([0.0, 0.0, 1.0], dtype=np.float32) if abs(v[2]) < 0.85 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        u = np.cross(v, ref)
        u = u / (np.linalg.norm(u) + 1e-8)
        w = np.cross(v, u)
        return u, w

    def _draw_idle_mannequin(self):
        """Draws an idle standing T-pose mannequin when no camera input is active."""
        GL.glColor4f(0.5, 0.52, 0.58, 0.6)
        self._draw_sphere(np.array([0.0, 0.55, 0.0]), 0.10)
        self._draw_cylinder(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.45, 0.0]), 0.12, 0.14)
        self._draw_cylinder(np.array([0.0, 0.42, 0.0]), np.array([-0.45, 0.42, 0.0]), 0.045, 0.035)
        self._draw_cylinder(np.array([0.0, 0.42, 0.0]), np.array([0.45, 0.42, 0.0]), 0.045, 0.035)
        self._draw_cylinder(np.array([-0.12, 0.0, 0.0]), np.array([-0.14, -0.65, 0.0]), 0.065, 0.045)
        self._draw_cylinder(np.array([0.12, 0.0, 0.0]), np.array([0.14, -0.65, 0.0]), 0.065, 0.045)

    # ==============================================================
    # 360° MOUSE ORBIT CONTROLS
    # ==============================================================
    def mousePressEvent(self, event):
        self.last_mouse_pos = event.position().toPoint()

    def mouseMoveEvent(self, event):
        cur_pos = event.position().toPoint()
        dx = cur_pos.x() - self.last_mouse_pos.x()
        dy = cur_pos.y() - self.last_mouse_pos.y()

        if event.buttons() & Qt.LeftButton:
            # Orbit rotation
            self.camera_yaw += dx * 0.6
            self.camera_pitch = max(-85.0, min(85.0, self.camera_pitch + dy * 0.6))
            self.update()
        elif event.buttons() & Qt.RightButton:
            # Camera pan
            self.camera_target[0] -= dx * 0.003
            self.camera_target[1] += dy * 0.003
            self.update()

        self.last_mouse_pos = cur_pos

    def wheelEvent(self, event):
        # Zoom in / out
        delta = event.angleDelta().y()
        self.camera_dist = max(0.8, min(8.0, self.camera_dist - delta * 0.0025))
        self.update()

    def mouseDoubleClickEvent(self, event):
        # Reset camera view to front center
        self.camera_yaw = 20.0
        self.camera_pitch = 10.0
        self.camera_dist = 2.4
        self.camera_target = np.array([0.0, 0.1, 0.0], dtype=np.float32)
        self.update()
