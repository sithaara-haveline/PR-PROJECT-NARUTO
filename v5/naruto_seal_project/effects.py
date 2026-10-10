"""
effects.py
----------
Animations that play on top of the camera view when a jutsu is cast.

Supports:
1. VIDEO effect - if effects/<jutsu_name_slug>.mp4 exists.
2. PARTICLE / CLONE / TRACKED effect - drawn procedurally with OpenCV:
   theme "fire" | "multi_clone" | "smoke_clone" | "spiral" | "reaper" | "chidori" | "dark"
"""

import math
import os
import random
import re

import cv2
import numpy as np

VIDEO_EXTS = (".mp4", ".webm", ".avi", ".mov", ".gif")
MAX_PARTICLES = 450


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _envelope(t, duration):
    """0 -> 1 fade-in (0.3 s), 1 -> 0 fade-out (last 0.8 s)."""
    return max(0.0, min(1.0, t / 0.3, (duration - t) / 0.8))


class _P:
    __slots__ = ("x", "y", "vx", "vy", "life", "max_life", "size", "ang", "rad")

    def __init__(self, x, y, vx, vy, life, size, ang=0.0, rad=0.0):
        self.x, self.y, self.vx, self.vy = x, y, vx, vy
        self.life = self.max_life = life
        self.size, self.ang, self.rad = size, ang, rad


class ParticleEffect:

    def __init__(
        self,
        theme,
        w,
        h,
        origin,
        duration,
        reaper_img_path="reaper_shinigami.png",
    ):
        valid_themes = (
            "fire",
            "smoke_clone",
            "multi_clone",
            "spiral",
            "reaper",
            "chidori",
            "dark",
        )
        self.theme = theme if theme in valid_themes else "fire"
        self.w, self.h, self.duration = w, h, duration
        self.ox, self.oy = origin
        self.p = []
        self.last_t = 0.0
        self.burst_done = False

        self.user_snapshot = None

        # Load Reaper Shinigami Overlay Image if available
        self.reaper_img = None
        if os.path.exists(reaper_img_path):
            img = cv2.imread(reaper_img_path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                self.reaper_img = cv2.resize(img, (w, h))

        # 12 Clone configurations (Back, Middle, Front depth rows)
        self.clone_layout = [
            {"x_r": -0.38, "y_r": -0.18, "scale": 0.50, "delay": 0.15},
            {"x_r": -0.14, "y_r": -0.20, "scale": 0.52, "delay": 0.15},
            {"x_r": 0.14, "y_r": -0.20, "scale": 0.52, "delay": 0.15},
            {"x_r": 0.38, "y_r": -0.18, "scale": 0.50, "delay": 0.15},
            {"x_r": -0.42, "y_r": -0.06, "scale": 0.72, "delay": 0.30},
            {"x_r": -0.22, "y_r": -0.08, "scale": 0.75, "delay": 0.30},
            {"x_r": 0.22, "y_r": -0.08, "scale": 0.75, "delay": 0.30},
            {"x_r": 0.42, "y_r": -0.06, "scale": 0.72, "delay": 0.30},
            {"x_r": -0.32, "y_r": 0.04, "scale": 0.95, "delay": 0.45},
            {"x_r": -0.12, "y_r": 0.08, "scale": 0.98, "delay": 0.45},
            {"x_r": 0.12, "y_r": 0.08, "scale": 0.98, "delay": 0.45},
            {"x_r": 0.32, "y_r": 0.04, "scale": 0.95, "delay": 0.45},
        ]

    def _emit(self, t, dt):
        if len(self.p) > MAX_PARTICLES:
            return
        live = t < self.duration - 0.5

        if self.theme == "fire" and live:
            for _ in range(int(150 * dt) + 1):
                self.p.append(
                    _P(
                        self.ox + random.gauss(0, 18),
                        self.oy + random.gauss(0, 10),
                        random.gauss(0, 70),
                        -random.uniform(120, 360),
                        random.uniform(0.7, 1.4),
                        random.uniform(8, 20),
                    )
                )

        elif self.theme in ("smoke_clone", "multi_clone"):
            if not self.burst_done:
                self.burst_done = True
                for _ in range(250):
                    a, s = random.uniform(0, 2 * math.pi), random.uniform(
                        100, 500
                    )
                    self.p.append(
                        _P(
                            self.ox,
                            self.oy,
                            math.cos(a) * s,
                            math.sin(a) * s,
                            random.uniform(1.0, 2.2),
                            random.uniform(22, 55),
                        )
                    )

        elif self.theme == "spiral" and live:
            # Rasengan particle emissions
            for _ in range(int(300 * dt) + 1):
                self.p.append(
                    _P(
                        0,
                        0,
                        0,
                        0,
                        random.uniform(0.6, 1.2),
                        random.uniform(4, 12),
                        ang=random.uniform(0, 2 * math.pi),
                        rad=random.uniform(4, 25),
                    )
                )

        elif self.theme == "chidori" and live:
            # Chidori lightning spark particles
            for _ in range(int(220 * dt) + 1):
                ang = random.uniform(0, 2 * math.pi)
                spd = random.uniform(150, 450)
                self.p.append(
                    _P(
                        self.ox,
                        self.oy,
                        math.cos(ang) * spd,
                        math.sin(ang) * spd,
                        random.uniform(0.2, 0.6),
                        random.uniform(3, 8),
                    )
                )

        elif self.theme == "reaper" and live:
            # Floating blue soul flames
            for _ in range(int(80 * dt) + 1):
                self.p.append(
                    _P(
                        random.uniform(50, self.w - 50),
                        random.uniform(50, self.h - 50),
                        random.gauss(0, 15),
                        -random.uniform(20, 60),
                        random.uniform(1.0, 2.5),
                        random.uniform(12, 28),
                    )
                )

        elif self.theme == "dark" and live:
            for _ in range(int(170 * dt) + 1):
                self.p.append(
                    _P(
                        random.uniform(0, self.w),
                        self.h + 10,
                        random.gauss(0, 25),
                        -random.uniform(70, 220),
                        random.uniform(1.2, 2.4),
                        random.uniform(8, 24),
                    )
                )

    def _step(self, dt):
        alive = []
        for q in self.p:
            q.life -= dt
            if q.life <= 0:
                continue
            if self.theme == "spiral":
                q.ang += (12.0 - q.rad * 0.01) * dt
                q.rad += 120 * dt
                q.x = self.ox + q.rad * math.cos(q.ang)
                q.y = self.oy + q.rad * math.sin(q.ang) * 0.85
            else:
                q.x += q.vx * dt
                q.y += q.vy * dt
            alive.append(q)
        self.p = alive

    def _color(self, q):
        a = 1.0 - q.life / q.max_life
        f = 1.0 - a
        if self.theme == "fire":
            return (
                int(30 * f**3),
                int(170 * f**2.0),
                int(235 * (1 - 0.45 * a)),
            )
        if self.theme in ("smoke_clone", "multi_clone"):
            v = int(240 * f**0.8)
            return (v, v, v)
        if self.theme in ("spiral", "reaper"):
            return (
                int(255 * (0.8 + 0.2 * f)),
                int(220 * f),
                int(80 * f),
            )
        if self.theme == "chidori":
            # Bright cyan/white electric sparks
            return (255, int(255 * f), int(200 * f))
        return (int(200 * f), int(40 * f), int(150 * f))

    def _render_scaled_clone(self, frame, snapshot, x_offset, y_offset, scale):
        h, w, _ = frame.shape
        nw, nh = int(w * scale), int(h * scale)
        if nw < 10 or nh < 10:
            return
        resized_snap = cv2.resize(
            snapshot, (nw, nh), interpolation=cv2.INTER_LINEAR
        )

        x1, y1 = int(w // 2 - nw // 2 + x_offset), int(
            h // 2 - nh // 2 + y_offset
        )
        x2, y2 = x1 + nw, y1 + nh
        cx1, cx2, cy1, cy2 = (
            max(0, x1),
            min(w, x2),
            max(0, y1),
            min(h, y2),
        )
        if cx1 >= cx2 or cy1 >= cy2:
            return

        snap_crop = resized_snap[
            (cy1 - y1) : (cy2 - y1), (cx1 - x1) : (cx2 - x1)
        ]
        gray_crop = cv2.cvtColor(snap_crop, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_crop, 5, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        frame_crop = frame[cy1:cy2, cx1:cx2]

        aura = np.zeros_like(frame_crop)
        cv2.drawContours(aura, contours, -1, (255, 210, 60), 4)

        np.copyto(frame_crop, snap_crop, where=(mask[:, :, None] == 255))
        cv2.addWeighted(frame_crop, 1.0, aura, 0.4, 0, frame_crop)

    def _draw_lightning_bolt(self, layer, start_pt, end_pt, color, thickness=2):
        """Generates jagged lightning arcs from start_pt to end_pt."""
        sx, sy = start_pt
        ex, ey = end_pt
        dist = math.hypot(ex - sx, ey - sy)
        num_segments = int(max(3, dist // 25))

        points = [(sx, sy)]
        for i in range(1, num_segments):
            frac = i / num_segments
            nx = sx + frac * (ex - sx) + random.randint(-20, 20)
            ny = sy + frac * (ey - sy) + random.randint(-20, 20)
            points.append((int(nx), int(ny)))
        points.append((ex, ey))

        for i in range(len(points) - 1):
            cv2.line(
                layer, points[i], points[i + 1], color, thickness, cv2.LINE_AA
            )

    def draw(self, frame, t, right_hand_coords=None):
        dt = min(0.1, max(0.0, t - self.last_t))
        self.last_t = t
        env = _envelope(t, self.duration)

        if right_hand_coords is not None:
            self.ox, self.oy = right_hand_coords

        if self.user_snapshot is None:
            self.user_snapshot = frame.copy()

        # 1. REAPER DEATH SEAL BACKGROUND
        if self.theme == "reaper":
            frame[:] = (frame * (1.0 - 0.35 * env)).astype(np.uint8)
            if self.reaper_img is not None:
                shinigami_layer = self.reaper_img.copy()
                cv2.addWeighted(
                    frame, 1.0, shinigami_layer, 0.45 * env, 0, frame
                )

        # 2. SHADOW CLONES
        if self.theme in ("smoke_clone", "multi_clone"):
            for cfg in self.clone_layout:
                if t >= cfg["delay"]:
                    x_off = int(self.w * cfg["x_r"])
                    y_off = int(self.h * cfg["y_r"])
                    self._render_scaled_clone(
                        frame, self.user_snapshot, x_off, y_off, cfg["scale"]
                    )

        self._emit(t, dt)
        self._step(dt)

        # 3. CHIDORI & RASENGAN LIGHTING & PARTICLES
        layer = np.zeros_like(frame)

        # Draw Chidori Electric Arcs radiating from right hand
        if self.theme == "chidori":
            # Darken surroundings slightly to make electric glow pop
            frame[:] = (frame * (1.0 - 0.25 * env)).astype(np.uint8)

            # Generate 8-12 jagged lightning branches
            for _ in range(random.randint(8, 12)):
                angle = random.uniform(0, 2 * math.pi)
                length = random.uniform(80, 260)
                end_x = int(self.ox + math.cos(angle) * length)
                end_y = int(self.oy + math.sin(angle) * length)

                # Outer cyan lightning
                self._draw_lightning_bolt(
                    layer,
                    (int(self.ox), int(self.oy)),
                    (end_x, end_y),
                    (255, 230, 100),
                    3,
                )
                # Inner white hot core
                self._draw_lightning_bolt(
                    layer,
                    (int(self.ox), int(self.oy)),
                    (end_x, end_y),
                    (255, 255, 255),
                    1,
                )

            # Chidori Hand Core Glow
            core_r = int(22 + 6 * math.sin(t * 35))
            cv2.circle(
                layer,
                (int(self.ox), int(self.oy)),
                core_r,
                (255, 255, 255),
                -1,
                cv2.LINE_AA,
            )
            cv2.circle(
                layer,
                (int(self.ox), int(self.oy)),
                int(core_r * 1.8),
                (255, 220, 80),
                3,
                cv2.LINE_AA,
            )

        # Draw Standard Particles
        for q in self.p:
            r = q.size * (0.6 + 0.6 * (q.life / q.max_life))
            cv2.circle(
                layer,
                (int(q.x), int(q.y)),
                max(1, int(r)),
                self._color(q),
                -1,
                cv2.LINE_AA,
            )

        # Draw Rasengan Orb
        if self.theme == "spiral":
            core_r = int(32 + 8 * math.sin(t * 25))
            cv2.circle(
                layer,
                (int(self.ox), int(self.oy)),
                core_r,
                (255, 245, 210),
                -1,
                cv2.LINE_AA,
            )
            cv2.circle(
                layer,
                (int(self.ox), int(self.oy)),
                int(core_r * 1.4),
                (255, 200, 50),
                4,
                cv2.LINE_AA,
            )

        glow = cv2.GaussianBlur(layer, (0, 0), 11)
        layer = cv2.addWeighted(layer, 0.85, glow, 0.75, 0)
        layer = (layer * env).astype(np.uint8)
        frame[:] = cv2.add(frame, layer)


class VideoEffect:

    def __init__(self, path, w, h):
        self.cap = cv2.VideoCapture(path)
        self.ok = self.cap.isOpened()
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.pos = 0
        self.w, self.h = w, h
        self.done = not self.ok

    def draw(self, frame, t):
        if self.done:
            return
        target = int(t * self.fps)
        img = None
        while self.pos <= target:
            ok, im = self.cap.read()
            if not ok:
                self.done = True
                return
            img, self.pos = im, self.pos + 1
        if img is not None:
            img = cv2.resize(img, (self.w, self.h))
            frame[:] = cv2.add(frame, img)


def make_effect(theme, jutsu_name, w, h, origin, duration, effects_dir="effects"):
    for ext in VIDEO_EXTS:
        path = os.path.join(effects_dir, slug(jutsu_name) + ext)
        if os.path.exists(path):
            eff = VideoEffect(path, w, h)
            if eff.ok:
                return eff
    return ParticleEffect(theme, w, h, origin, duration)