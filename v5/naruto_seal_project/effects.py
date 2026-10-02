"""
effects.py
----------
Animations that play on top of the camera view when a jutsu is cast.

Two kinds, chosen automatically per jutsu:

1. VIDEO effect  - if  effects/<jutsu_name_slug>.mp4  exists (e.g.
   effects/fireball_jutsu.mp4) it is played over the camera image with
   ADDITIVE blending ("screen"). Use clips with a BLACK background (search for
   "fire effect black background" / "green-screen-free overlay" style clips;
   check the licence before using them in a report/demo).
2. PARTICLE effect - otherwise a procedural animation is drawn with OpenCV:
   theme "fire" | "smoke_clone" | "spiral" | "dark"   (set per jutsu in
   jutsu_config.json).

Everything is drawn onto the frame IN PLACE. No extra dependencies.
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
    def __init__(self, theme, w, h, origin, duration):
        self.theme = theme if theme in ("fire", "smoke_clone", "spiral", "dark") else "fire"
        self.w, self.h, self.duration = w, h, duration
        self.ox, self.oy = origin
        self.p = []
        self.last_t = 0.0
        self.burst_done = False

    # ---- emission ---------------------------------------------------------
    def _emit(self, t, dt):
        if len(self.p) > MAX_PARTICLES:
            return
        live = t < self.duration - 1.0
        n = 0
        if self.theme == "fire" and live:
            for _ in range(int(150 * dt) + 1):
                self.p.append(_P(self.ox + random.gauss(0, 18), self.oy + random.gauss(0, 10),
                                 random.gauss(0, 70), -random.uniform(120, 360),
                                 random.uniform(0.7, 1.4), random.uniform(8, 20)))
        elif self.theme == "smoke_clone":
            if not self.burst_done:
                self.burst_done = True
                for _ in range(150):
                    a, s = random.uniform(0, 2 * math.pi), random.uniform(80, 400)
                    self.p.append(_P(self.ox, self.oy, math.cos(a) * s, math.sin(a) * s,
                                     random.uniform(1.0, 1.9), random.uniform(20, 48)))
            elif t < 1.2:
                for _ in range(int(60 * dt) + 1):
                    self.p.append(_P(self.ox + random.gauss(0, 60), self.oy + random.gauss(0, 60),
                                     random.gauss(0, 30), -random.uniform(10, 60),
                                     random.uniform(0.8, 1.4), random.uniform(18, 40)))
        elif self.theme == "spiral" and live:
            for _ in range(int(280 * dt) + 1):
                self.p.append(_P(0, 0, 0, 0, random.uniform(0.7, 1.4), random.uniform(4, 10),
                                 ang=random.uniform(0, 2 * math.pi), rad=random.uniform(4, 20)))
        elif self.theme == "dark" and live:
            for _ in range(int(170 * dt) + 1):
                self.p.append(_P(random.uniform(0, self.w), self.h + 10, random.gauss(0, 25),
                                 -random.uniform(70, 220), random.uniform(1.2, 2.4),
                                 random.uniform(8, 24)))

    # ---- physics ------------------------------------------------------------
    def _step(self, dt):
        alive = []
        for q in self.p:
            q.life -= dt
            if q.life <= 0:
                continue
            if self.theme == "spiral":
                q.ang += (10.0 - q.rad * 0.012) * dt
                q.rad += 170 * dt
                q.x = self.ox + q.rad * math.cos(q.ang)
                q.y = self.oy + q.rad * math.sin(q.ang) * 0.85
            else:
                q.x += q.vx * dt
                q.y += q.vy * dt
                if self.theme == "smoke_clone":
                    q.vx *= 0.96
                    q.vy *= 0.96
            alive.append(q)
        self.p = alive

    # ---- colour -------------------------------------------------------------
    def _color(self, q):
        a = 1.0 - q.life / q.max_life  # 0 = newborn, 1 = dying
        f = 1.0 - a
        if self.theme == "fire":
            return (int(30 * f ** 3), int(170 * f ** 2.0), int(235 * (1 - 0.45 * a)))
        if self.theme == "smoke_clone":
            v = int(235 * f ** 0.8)
            return (v, v, v)
        if self.theme == "spiral":
            return (int(255 * f), int(210 * f), int(70 * f))
        return (int(200 * f), int(40 * f), int(150 * f))  # dark / purple

    # ---- draw ---------------------------------------------------------------
    def draw(self, frame, t):
        dt = min(0.1, max(0.0, t - self.last_t))
        self.last_t = t
        env = _envelope(t, self.duration)

        if self.theme == "dark":  # darken the whole scene
            frame[:] = (frame * (1.0 - 0.5 * env)).astype(np.uint8)

        self._emit(t, dt)
        self._step(dt)

        layer = np.zeros_like(frame)
        for q in self.p:
            r = q.size * (0.6 + 0.6 * (q.life / q.max_life)) if self.theme != "smoke_clone" \
                else q.size * (1.4 - 0.6 * (q.life / q.max_life))
            cv2.circle(layer, (int(q.x), int(q.y)), max(1, int(r)), self._color(q), -1, cv2.LINE_AA)

        if self.theme == "spiral":  # glowing core
            core = 26 + 6 * math.sin(t * 22)
            cv2.circle(layer, (int(self.ox), int(self.oy)), int(core), (255, 235, 190), -1, cv2.LINE_AA)

        glow = cv2.GaussianBlur(layer, (0, 0), 9)
        layer = cv2.addWeighted(layer, 0.8, glow, 0.7, 0)
        layer = (layer * env).astype(np.uint8)
        frame[:] = cv2.add(frame, layer)

        if self.theme == "smoke_clone" and t < 0.25:  # white flash
            flash = int(200 * (1 - t / 0.25))
            frame[:] = cv2.add(frame, np.full_like(frame, flash))


class VideoEffect:
    """Plays a black-background clip with additive blending. Plays once."""

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
