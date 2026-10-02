"""
test_app_headless.py  -  run:  python test_app_headless.py
Simulates a whole session WITHOUT a webcam or MediaPipe: fake hand landmarks,
scripted seal predictions. Checks the state machine end to end and writes a few
screenshots to ./screens/ so you can eyeball the layout.
"""

import os
import random

import cv2
import joblib
import numpy as np

import jutsu_app
from sequence_engine import IDLE, RECORDING, RESULT, load_config

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, "screens"), exist_ok=True)
FPS = 30


class LM:
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


class Hand:
    def __init__(self, wx, wy, tip=None):
        rnd = random.Random(int(wx * 1000))
        self.landmark = [LM(wx, wy)] + [LM(wx + rnd.uniform(-.08, .08), wy - rnd.uniform(.02, .2)) for _ in range(20)]
        self.landmark[9] = LM(wx, wy - 0.1)
        if tip:
            self.landmark[8] = LM(*tip)


class Res:
    def __init__(self, hands):
        self.multi_hand_landmarks = hands or None


def camera_frame():
    g = np.linspace(40, 120, 1280, dtype=np.uint8)
    f = np.dstack([np.tile(g, (720, 1))] * 3)
    cv2.putText(f, "(fake camera)", (500, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, (200, 200, 200), 3)
    return f


def fake_images(folder):
    os.makedirs(folder, exist_ok=True)
    for name, col in [("Ram", (60, 60, 200)), ("Boar", (60, 160, 60)), ("Horse", (200, 120, 60))]:
        img = np.full((300, 400, 3), col, np.uint8)
        cv2.putText(img, name, (60, 170), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
        cv2.imwrite(os.path.join(folder, name + ".png"), img)


def main():
    cfg = load_config(os.path.join(HERE, "jutsu_config.json"))
    bundle = joblib.load(os.path.join(HERE, "model.pkl"))
    img_dir = os.path.join(HERE, "screens", "_fake_seal_images")
    fake_images(img_dir)
    app = jutsu_app.App(cfg, bundle, images_dir=img_dir)
    assert app.book.validate(set(app.classes), app.controls, app.idle_labels) == []

    state = {"label": "Normal/Neutral"}
    app.classify = lambda vec: (state["label"], 0.95)
    two = [Hand(0.35, 0.7), Hand(0.6, 0.7)]
    frame = camera_frame()
    c = cfg["controls"]
    t = [1000.0]
    shots = {}

    def run(seconds, label, hands, shot=None):
        state["label"] = label
        for i in range(int(seconds * FPS)):
            canvas = app.step(frame, Res(hands), t[0])
            t[0] += 1 / FPS
            if shot and i == int(seconds * FPS) - 2:
                cv2.imwrite(os.path.join(HERE, "screens", shot + ".png"), canvas)
                shots[shot] = canvas.shape
        return canvas

    run(1.0, None, [])                                             # nobody in view
    run(1.0, "Normal/Neutral", two)
    run(1.0, c["start"], two)
    assert app.machine.state == RECORDING, app.machine.state
    for seal in ["Ram", "Horse", "Boar"]:
        run(0.8, "Normal/Neutral", two)
        run(1.0, seal, two)
    run(0.8, "Normal/Neutral", two, "1_recording")
    assert app.machine.seq == ["Ram", "Horse", "Boar"], app.machine.seq

    run(1.0, c["undo"], two)
    assert app.machine.seq == ["Ram", "Horse"], app.machine.seq
    run(0.8, "Normal/Neutral", two)
    run(1.0, "Boar", two)
    run(0.8, "Normal/Neutral", two)
    run(1.0, c["cast"], two)
    assert app.machine.state == RESULT and app.machine.result["name"] == "Fireball Jutsu", app.machine.result
    run(0.6, c["cast"], two, "2_cast_fire_early")
    run(1.5, c["cast"], two, "3_cast_fire_mid")
    run(5.0, "Normal/Neutral", two)
    assert app.machine.state == IDLE

    # wrong sequence -> no match message; then the hover button opens the list
    run(1.0, c["start"], two)
    run(0.8, "Normal/Neutral", two)
    run(1.0, "Hare", two)
    run(0.8, "Normal/Neutral", two)
    run(1.0, c["cast"], two, "4_no_match")
    assert app.machine.result is None and app.machine.state == RESULT
    run(7.0, "Normal/Neutral", two)

    bx, by, bw, bh = app.btn
    tip = ((bx + bw / 2) / app.CAM_W, (by + bh / 2) / app.CAM_H)
    one = [Hand(0.5, 0.6, tip=tip)]
    assert not app.show_list
    run(6.0, None, one, "5_hover_progress")
    assert not app.show_list, "6 s hover must NOT press a 7 s button"
    run(1.5, None, one)
    assert app.show_list, "7+ s hover should open the list"
    run(1.0, None, [], "6_list_open")
    run(1.0, None, one)                                            # still latched
    assert app.show_list

    # Sample effects for other themes
    for theme in ["smoke_clone", "spiral", "dark"]:
        name = {"smoke_clone": "Shadow Clone Jutsu", "spiral": "Rasengan", "dark": "Reaper Death Seal"}[theme]
        entry = [e for e in cfg["jutsu"] if e["name"] == name][0]
        app.show_list = False
        app.machine.state, app.machine.result = RESULT, entry
        app.machine.result_until = t[0] + 99
        app._start_effect(entry, t[0])
        run(1.0, "Normal/Neutral", two, "7_" + theme)
    print("ALL HEADLESS CHECKS PASSED; screenshots:", sorted(shots))
    print("canvas size (h, w, c):", shots["1_recording"])


if __name__ == "__main__":
    main()
