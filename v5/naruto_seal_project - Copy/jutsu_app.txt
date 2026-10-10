"""
jutsu_app.py  -  live hand-seal recognition + jutsu sequences
=============================================================
    python jutsu_app.py
    python jutsu_app.py --camera 1          # another webcam
    python jutsu_app.py --cam-w 1120 --cam-h 630   # bigger picture (window is resizable too)

HOW TO USE
  * Seal reference pictures live in  seal_images/<SealName>.png  (or .jpg). They
    appear in the right-hand panel. Missing pictures show a name tile instead.
  * Show BOTH hands to make a seal. Hold it ~0.7 s and it registers once.
    (Same seal twice in a row? Relax your hands in between.)
  * CONTROL SEALS (set in jutsu_config.json):
        START  -> begins recording a sequence
        UNDO   -> erases the last seal in the current sequence
        CAST   -> matches the sequence to a jutsu and plays its animation
  * SELECT JUTSU button (top-right of the camera view): raise ONE hand and hold
    your index fingertip inside the button for 7 s to open/close the reference
    list of jutsu -> seal sequences. (One hand = pointer, two hands = seals.)
  * Keyboard backups:  J = toggle list | C = clear sequence | H = toggle hints | Q/Esc = quit
"""

import argparse
import math
import os
import sys
import time

import cv2
import joblib
import numpy as np

from common import extract_features, make_hands
from effects import make_effect
from sequence_engine import (IDLE, RECORDING, RESULT, DwellButton, JutsuBook,
                             SealStabilizer, SequenceMachine, load_config)

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = cv2.FONT_HERSHEY_SIMPLEX
DUPLEX = cv2.FONT_HERSHEY_DUPLEX

WHITE, BLACK, GREEN, RED = (255, 255, 255), (0, 0, 0), (80, 220, 80), (70, 70, 255)
YELLOW, ORANGE, GRAY, DARK = (0, 220, 255), (0, 150, 255), (150, 150, 150), (35, 35, 35)
ROLE_COLOR = {"start": (80, 200, 80), "undo": (60, 160, 255), "cast": (60, 60, 230)}

HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
                    (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
                    (18, 19), (19, 20), (0, 17)]


# ------------------------------------------------------------------ drawing helpers
def put(img, text, org, scale=0.6, color=WHITE, thick=1, font=FONT):
    cv2.putText(img, text, org, font, scale, BLACK, thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, font, scale, color, thick, cv2.LINE_AA)


def rect_alpha(img, p1, p2, color, alpha):
    x1, y1 = max(0, p1[0]), max(0, p1[1])
    x2, y2 = min(img.shape[1], p2[0]), min(img.shape[0], p2[1])
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    img[y1:y2, x1:x2] = cv2.addWeighted(roi, 1 - alpha, np.full_like(roi, color), alpha, 0)


def fit(img, w, h, bg=DARK):
    """Resize keeping aspect ratio, centred on a w x h tile."""
    tile = np.full((h, w, 3), bg, np.uint8)
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * s)), max(1, int(ih * s))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    tile[y0:y0 + nh, x0:x0 + nw] = r
    return tile


def draw_hand(img, hand, w, h, color=(0, 255, 120)):
    pts = [(int(l.x * w), int(l.y * h)) for l in hand.landmark]
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, p, 3, (0, 0, 255), -1, cv2.LINE_AA)


def find_image(folder, name):
    variants = {name, name.replace("/", "_"), name.replace(" ", "_"), name.lower(),
                name.replace("/", "_").replace(" ", "_").lower()}
    for v in variants:
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            p = os.path.join(folder, v + ext)
            if os.path.exists(p):
                img = cv2.imread(p)
                if img is not None:
                    return img
    return None


# ------------------------------------------------------------------ the app
class App:
    def __init__(self, cfg, bundle, cam_w=960, cam_h=540, panel_w=320, bar_h=140,
                 images_dir=None, effects_dir=None):
        self.cfg, self.pipe, self.classes = cfg, bundle["pipeline"], list(bundle["classes"])
        self.CAM_W, self.CAM_H, self.PANEL_W, self.BAR_H = cam_w, cam_h, panel_w, bar_h
        self.W, self.H = cam_w + panel_w, cam_h + bar_h
        self.effects_dir = effects_dir or os.path.join(HERE, "effects")

        tm = cfg["timing"]
        self.idle_labels = set(cfg["idle_labels"])
        self.controls = cfg["controls"]
        self.book = JutsuBook.from_config(cfg)
        self.machine = SequenceMachine(self.book, self.controls, tm["max_len"], tm["result_s"], tm["timeout_s"])
        self.stab = SealStabilizer(hold_s=tm["hold_s"], min_conf=tm["min_conf"])
        self.button = DwellButton(dwell_s=tm["dwell_s"])
        self.show_list, self.show_hints = False, True
        self.effect, self.effect_t0 = None, 0.0
        self.last_origin = (cam_w // 2, cam_h // 2)

        # reference images (seal_images/<Seal>.png)
        self.seal_names = [c for c in self.classes if c not in self.idle_labels]
        folder = images_dir or os.path.join(HERE, "seal_images")
        self.images = {s: find_image(folder, s) for s in self.seal_names}
        missing = [s for s, im in self.images.items() if im is None]
        if missing:
            print(f"[info] no reference picture yet for: {missing}  (put PNG/JPG files in {folder})")
        self.btn = (cam_w - 250, 10, 240, 56)  # x, y, w, h  (inside the camera view)

    # -- model ---------------------------------------------------------------
    def classify(self, vec):
        proba = self.pipe.predict_proba(vec.reshape(1, -1))[0]
        i = int(np.argmax(proba))
        return self.classes[i], float(proba[i])

    # -- one frame -------------------------------------------------------------
    def step(self, frame_bgr, results, t):
        vec, hands_two = extract_features(results)
        all_hands = results.multi_hand_landmarks or []

        label, conf = (None, 0.0)
        if vec is not None:
            label, conf = self.classify(vec)

        disp = cv2.resize(frame_bgr, (self.CAM_W, self.CAM_H))
        for hnd in all_hands:
            draw_hand(disp, hnd, self.CAM_W, self.CAM_H)
        if len(hands_two) == 2:
            wx = np.mean([h.landmark[0].x for h in hands_two]) * self.CAM_W
            wy = np.mean([h.landmark[0].y for h in hands_two]) * self.CAM_H
            self.last_origin = (wx, wy)

        # ---- seal events -> state machine
        fired = self.stab.update(label, conf, t)
        if fired is not None and fired not in self.idle_labels:
            ev = self.machine.handle_seal(fired, t)
            if ev == "cast":
                self._start_effect(self.machine.result, t)
        self.machine.tick(t)
        if self.machine.state != RESULT:
            self.effect = None

        # ---- pointer + select-jutsu button (only with exactly ONE hand visible)
        pointer = None
        if len(all_hands) == 1:
            tip = all_hands[0].landmark[8]
            pointer = (int(tip.x * self.CAM_W), int(tip.y * self.CAM_H))
        bx, by, bw, bh = self.btn
        inside = pointer is not None and bx <= pointer[0] <= bx + bw and by <= pointer[1] <= by + bh
        if self.button.update(inside, t):
            self.show_list = not self.show_list

        # ---- animation on top of the camera picture
        if self.effect is not None:
            self.effect.draw(disp, t - self.effect_t0)

        self._draw_camera_overlays(disp, label, conf, len(all_hands), pointer, t)

        canvas = np.zeros((self.H, self.W, 3), np.uint8)
        canvas[:self.CAM_H, :self.CAM_W] = disp
        self._draw_panel(canvas, t)
        self._draw_bar(canvas, t)
        return canvas

    def _start_effect(self, entry, t):
        self.effect_t0 = t
        self.effect = make_effect(entry.get("theme", "fire"), entry["name"], self.CAM_W, self.CAM_H,
                                  self.last_origin, self.cfg["timing"]["result_s"], self.effects_dir)

    # -- overlays on the camera view ---------------------------------------------
    def _draw_camera_overlays(self, disp, label, conf, n_hands, pointer, t):
        # status line
        if n_hands == 0:
            status, col = "Show both hands", GRAY
        elif n_hands == 1:
            status, col = "1 hand = pointer mode", YELLOW
        elif label is None:
            status, col = "Seal: ?", GRAY
        else:
            status, col = f"Seal: {label} ({conf * 100:.0f}%)", (GREEN if conf >= self.cfg["timing"]["min_conf"] else ORANGE)
        put(disp, status, (14, 30), 0.8, col, 2, DUPLEX)
        prog = self.stab.hold_progress(t)
        if prog > 0 and self.stab.current not in self.idle_labels:
            cv2.rectangle(disp, (14, 40), (14 + int(220 * prog), 48), GREEN, -1)
            cv2.rectangle(disp, (14, 40), (234, 48), WHITE, 1)

        msg = self.machine.message(t)
        if msg:
            put(disp, msg, (14, 78), 0.7, YELLOW, 2)

        # button
        bx, by, bw, bh = self.btn
        rect_alpha(disp, (bx, by), (bx + bw, by + bh), (60, 40, 20), 0.75)
        cv2.rectangle(disp, (bx, by), (bx + bw, by + bh), WHITE, 2)
        label_txt = "CLOSE LIST" if self.show_list else "SELECT JUTSU"
        (tw, _), _ = cv2.getTextSize(label_txt, DUPLEX, 0.7, 2)
        put(disp, label_txt, (bx + (bw - tw) // 2, by + 32), 0.7, WHITE, 2, DUPLEX)
        p = self.button.progress(t)
        cv2.rectangle(disp, (bx + 4, by + bh - 14), (bx + 4 + int((bw - 8) * p), by + bh - 6), GREEN, -1)
        put(disp, f"hover {self.cfg['timing']['dwell_s']:.0f}s", (bx + 6, by + bh + 18), 0.45, GRAY)
        if pointer is not None:
            cv2.circle(disp, pointer, 12, YELLOW, 2, cv2.LINE_AA)
            cv2.circle(disp, pointer, 3, YELLOW, -1, cv2.LINE_AA)

        if self.show_list:
            self._draw_list(disp)

        # result banner
        m = self.machine
        if m.state == RESULT:
            if m.result is not None:
                txt = m.result["name"].upper()
                (tw, _), _ = cv2.getTextSize(txt, DUPLEX, 1.6, 4)
                x = max(10, (self.CAM_W - tw) // 2)
                rect_alpha(disp, (0, self.CAM_H - 150), (self.CAM_W, self.CAM_H - 60), (0, 0, 0), 0.55)
                put(disp, txt, (x, self.CAM_H - 88), 1.6, ORANGE, 4, DUPLEX)
                put(disp, " > ".join(m.result["seals"]), (x, self.CAM_H - 68), 0.5, WHITE, 1)
            else:
                rect_alpha(disp, (0, self.CAM_H - 120), (self.CAM_W, self.CAM_H - 60), (0, 0, 0), 0.55)
                put(disp, "NO JUTSU MATCHES THAT SEQUENCE", (30, self.CAM_H - 80), 1.0, RED, 3, DUPLEX)

    def _draw_list(self, disp):
        x1, y1, x2, y2 = 20, 90, self.CAM_W - 20, self.CAM_H - 20
        rect_alpha(disp, (x1, y1), (x2, y2), (15, 15, 15), 0.85)
        cv2.rectangle(disp, (x1, y1), (x2, y2), WHITE, 1)
        c = self.controls
        put(disp, "JUTSU  ->  SEAL SEQUENCE      (reference)", (x1 + 14, y1 + 28), 0.7, YELLOW, 2)
        put(disp, f"START = {c['start']}   |   UNDO = {c['undo']}   |   CAST = {c['cast']}",
            (x1 + 14, y1 + 52), 0.5, GRAY)
        n = max(1, len(self.book.entries))
        step = min(62, (y2 - y1 - 80) // n)
        y = y1 + 90
        for e in self.book.entries:
            put(disp, e["name"], (x1 + 14, y), 0.7, ORANGE, 2)
            put(disp, "  >  ".join(e["seals"]), (x1 + 34, y + 24), 0.55, WHITE, 1)
            y += step

    # -- right-hand reference panel -------------------------------------------
    def _draw_panel(self, canvas, t):
        x0 = self.CAM_W
        canvas[:self.CAM_H, x0:] = (28, 28, 28)
        put(canvas, "HAND SEALS", (x0 + 12, 26), 0.7, YELLOW, 2)
        n = max(1, len(self.seal_names))
        cols = 2
        rows = math.ceil(n / cols)
        tile_w = (self.PANEL_W - 12 * 3) // cols
        tile_h = min(130, (self.CAM_H - 44 - 8 * rows) // rows)
        role_of = {v: k for k, v in self.controls.items()}
        for i, s in enumerate(self.seal_names):
            r, c = divmod(i, cols)
            x = x0 + 12 + c * (tile_w + 12)
            y = 40 + r * (tile_h + 8)
            img_h = tile_h - 20
            if self.images[s] is not None:
                tile = fit(self.images[s], tile_w, img_h)
            else:
                tile = np.full((img_h, tile_w, 3), (55, 55, 55), np.uint8)
                put(tile, "no image", (8, img_h // 2), 0.45, GRAY)
            canvas[y:y + img_h, x:x + tile_w] = tile
            active = self.stab.current == s
            cv2.rectangle(canvas, (x, y), (x + tile_w, y + img_h), GREEN if active else (90, 90, 90),
                          3 if active else 1)
            if active:
                bw = int(tile_w * self.stab.hold_progress(t))
                cv2.rectangle(canvas, (x, y + img_h + 2), (x + bw, y + img_h + 5), GREEN, -1)
            role = role_of.get(s)
            put(canvas, s, (x + 2, y + tile_h - 3), 0.45, ROLE_COLOR.get(role, WHITE))
            if role:
                cv2.rectangle(canvas, (x, y), (x + 52, y + 16), ROLE_COLOR[role], -1)
                put(canvas, role.upper(), (x + 3, y + 12), 0.4, WHITE)

    # -- bottom bar: state + sequence chips -------------------------------------
    def _draw_bar(self, canvas, t):
        y0 = self.CAM_H
        canvas[y0:, :] = (20, 20, 20)
        cv2.line(canvas, (0, y0), (self.W, y0), (90, 90, 90), 1)
        m, c = self.machine, self.controls
        if m.state == IDLE:
            put(canvas, f"IDLE - show {c['start']} (START) to begin a sequence", (14, y0 + 26), 0.7, WHITE, 2)
        elif m.state == RECORDING:
            put(canvas, f"RECORDING  |  UNDO: {c['undo']}   CAST: {c['cast']}", (14, y0 + 26), 0.7, GREEN, 2)
        else:
            put(canvas, "CASTING...", (14, y0 + 26), 0.7, ORANGE, 2)

        n_slots = self.cfg["timing"]["max_len"]
        chip_w = min(150, (self.W - 28) // n_slots - 6)
        for i in range(n_slots):
            x = 14 + i * (chip_w + 6)
            y = y0 + 42
            filled = i < len(m.seq)
            cv2.rectangle(canvas, (x, y), (x + chip_w, y + 48), (70, 70, 70) if filled else (45, 45, 45),
                          -1 if filled else 1)
            if filled:
                s = m.seq[i]
                img = self.images.get(s)
                if img is not None:
                    canvas[y + 6:y + 42, x + 4:x + 4 + 48] = fit(img, 48, 36)
                put(canvas, f"{i + 1}. {s}", (x + (54 if img is not None else 6), y + 30), 0.42, WHITE)
            else:
                put(canvas, str(i + 1), (x + 6, y + 30), 0.5, (90, 90, 90))

        if m.state == RECORDING and self.show_hints:
            cands = self.book.candidates(m.seq) if m.seq else []
            if m.seq and not cands:
                put(canvas, "No jutsu starts with this sequence (UNDO to fix)", (14, y0 + 116), 0.55, ORANGE)
            elif cands:
                put(canvas, "Possible: " + ", ".join(e["name"] for e in cands), (14, y0 + 116), 0.55, GRAY)
        put(canvas, "J list | C clear | H hints | Q quit", (self.W - 290, y0 + self.BAR_H - 10), 0.45, GRAY)

    # -- keyboard --------------------------------------------------------------
    def key(self, k, t):
        if k == ord("j"):
            self.show_list = not self.show_list
        elif k == ord("c"):
            self.machine.clear(t)
        elif k == ord("h"):
            self.show_hints = not self.show_hints


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--model", default=os.path.join(HERE, "model.pkl"))
    ap.add_argument("--config", default=os.path.join(HERE, "jutsu_config.json"))
    ap.add_argument("--cam-w", type=int, default=960, help="camera view width on screen")
    ap.add_argument("--cam-h", type=int, default=540, help="camera view height on screen")
    args = ap.parse_args()

    cfg = load_config(args.config)
    bundle = joblib.load(args.model)
    app = App(cfg, bundle, cam_w=args.cam_w, cam_h=args.cam_h)

    problems = app.book.validate(set(app.classes), app.controls, app.idle_labels)
    if problems:
        print("\nFix jutsu_config.json first:")
        for p in problems:
            print("  -", p)
        sys.exit(1)

    hands = make_hands()
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    win = "Naruto Hand Seals"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, app.W, app.H)

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        t = time.time()
        cv2.imshow(win, app.step(frame, results, t))
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        app.key(k, t)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
