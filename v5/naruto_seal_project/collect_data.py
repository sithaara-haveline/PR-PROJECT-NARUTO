"""
collect_data.py
---------------
Collect training samples for ONE seal at a time, in 9 varied conditions
(distance / tilt / position in frame).

    python collect_data.py --seal Tiger
    python collect_data.py --seal "Special Cross Seal"
    python collect_data.py --seal Normal/Neutral     # the "hands relaxed / no seal" class

Controls:
    SPACE : start the 5-second "get ready" countdown for the current condition
    R     : throw away the current condition's samples and redo it
    Q/ESC : quit (finished conditions are already saved)

Rules enforced here:
  * A frame is saved ONLY if both hands are detected (no zero-padding).
  * All 25 frames of a condition are written together, so a half-finished
    condition never pollutes the dataset.
  * Every condition gets its own session_id -> train_compare.py splits by
    session so near-identical consecutive frames can't leak into the test set.
"""

import argparse
import csv
import os
import time

import cv2
import mediapipe as mp
import pandas as pd

from common import extract_features, feature_columns, make_hands

FRAMES_PER_CONDITION = 25
PREP_SECONDS = 5
CAPTURE_INTERVAL_SEC = 0.1

CONDITIONS = [
    "Straight-on, medium distance, centered",
    "Straight-on, close to camera, centered",
    "Straight-on, far from camera, centered",
    "Tilted slightly LEFT, medium distance, centered",
    "Tilted slightly RIGHT, medium distance, centered",
    "Straight-on, medium distance, shifted LEFT in frame",
    "Straight-on, medium distance, shifted RIGHT in frame",
    "Tilted LEFT, close to camera, centered",
    "Tilted RIGHT, far from camera, centered",
]

mp_draw = mp.solutions.drawing_utils
mp_hands_mod = mp.solutions.hands


def ensure_header(path):
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(["seal", "session_id", "condition"] + feature_columns())


def existing_summary(path, seal):
    if not os.path.exists(path):
        return 0, 0
    try:
        df = pd.read_csv(path, usecols=["seal", "session_id"])
    except Exception:
        return 0, 0
    d = df[df["seal"] == seal]
    return len(d), d["session_id"].nunique()


def write_rows(path, seal, session_id, condition, rows):
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        for vec in rows:
            w.writerow([seal, session_id, condition] + [float(v) for v in vec])


def put(frame, text, y, color=(255, 255, 255), scale=0.6):
    cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seal", required=True, help="exact class name, e.g. Tiger")
    ap.add_argument("--out", default="hand_seal_dataset_clean.csv")
    args = ap.parse_args()

    ensure_header(args.out)
    n_rows, n_sess = existing_summary(args.out, args.seal)
    print(f"'{args.seal}' already has {n_rows} rows in {n_sessions_text(n_sess)} in {args.out}")
    if n_sess:
        print("Note: running again adds MORE sessions for this seal (good if you want more data).")

    hands = make_hands()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cond_idx = 0
    state = "idle"  # idle -> prep -> collecting
    prep_start = 0.0
    last_cap = 0.0
    buffer = []
    session_id = None

    while cap.isOpened() and cond_idx < len(CONDITIONS):
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        vec, hands_sorted = extract_features(res)
        for h in hands_sorted:
            mp_draw.draw_landmarks(frame, h, mp_hands_mod.HAND_CONNECTIONS)

        now = time.time()
        if state == "prep" and now - prep_start >= PREP_SECONDS:
            state = "collecting"
            buffer = []
            session_id = f"{args.seal}_{cond_idx}_{int(now)}"

        if state == "collecting" and vec is not None and now - last_cap >= CAPTURE_INTERVAL_SEC:
            buffer.append(vec)
            last_cap = now
            if len(buffer) >= FRAMES_PER_CONDITION:
                write_rows(args.out, args.seal, session_id, CONDITIONS[cond_idx], buffer)
                print(f"saved condition {cond_idx + 1}/9 ({len(buffer)} frames)")
                buffer, state = [], "idle"
                cond_idx += 1
                if cond_idx >= len(CONDITIONS):
                    break

        # ---- on-screen text ----
        put(frame, f"Seal: {args.seal}", 30, (0, 255, 0), 0.8)
        if cond_idx < len(CONDITIONS):
            put(frame, f"Condition {cond_idx + 1}/9: {CONDITIONS[cond_idx]}", 65, (255, 255, 0), 0.55)
        both = vec is not None
        put(frame, "BOTH HANDS OK" if both else "Need BOTH hands visible",
            100, (0, 255, 0) if both else (0, 0, 255))
        if state == "idle":
            put(frame, "Get into position, press SPACE", 135)
        elif state == "prep":
            put(frame, f"Get ready... {max(0.0, PREP_SECONDS - (now - prep_start)):.1f}s", 135, (0, 165, 255))
        else:
            put(frame, f"Collecting {len(buffer)}/{FRAMES_PER_CONDITION}  (only counts when both hands seen)",
                135, (0, 0, 255))

        cv2.imshow("Collect seal data", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" ") and state == "idle":
            state, prep_start = "prep", time.time()
        if key == ord("r"):
            state, buffer = "idle", []
            print("condition reset")

    cap.release()
    cv2.destroyAllWindows()
    n_rows, n_sess = existing_summary(args.out, args.seal)
    print(f"Done. '{args.seal}' now has {n_rows} rows in {n_sess} sessions.")


def n_sessions_text(n):
    return f"{n} sessions"


if __name__ == "__main__":
    main()
