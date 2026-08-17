"""
Live Seal Recognition
-----------------------
Purpose: load all saved seal templates (from capture_seal_template.py) and,
in real time, compare your current hand pose against each template. If the
closest match is under the distance threshold, show that seal's name.

Run capture_seal_template.py first for each seal you want recognized -
this script does nothing useful until seal_templates.json has entries.
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import os

TEMPLATE_FILE = "seal_templates.json"

# Max average per-point distance to count as a match. Lower = stricter.
# Start around 0.4-0.6 and tune based on false positives/negatives you see.
MATCH_THRESHOLD = 0.5

# How many consecutive matching frames needed before we lock in a display,
# so a single lucky/unlucky frame doesn't flicker the result.
STABLE_FRAMES_REQUIRED = 5

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def normalize_hand(landmarks):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
    wrist = pts[0].copy()
    pts -= wrist
    scale = np.linalg.norm(pts[9])
    if scale < 1e-6:
        scale = 1e-6
    pts /= scale
    return pts.flatten()


def classify_left_right(handedness_label, is_mirrored=True):
    if not is_mirrored:
        return handedness_label
    return "Left" if handedness_label == "Right" else "Right"


def match_seal(live_left, live_right, templates):
    """Compare live normalized hands against every stored template.
    Returns (best_seal_name, distance) or (None, None) if no templates."""
    best_name = None
    best_dist = float("inf")

    for name, template in templates.items():
        t_left = np.array(template["Left"])
        t_right = np.array(template["Right"])

        dist_left = np.linalg.norm(live_left - t_left) / len(t_left)
        dist_right = np.linalg.norm(live_right - t_right) / len(t_right)
        combined = (dist_left + dist_right) / 2

        if combined < best_dist:
            best_dist = combined
            best_name = name

    return best_name, best_dist


if not os.path.exists(TEMPLATE_FILE):
    print(f"No {TEMPLATE_FILE} found. Run capture_seal_template.py first "
          f"to build at least one seal template.")
    exit()

with open(TEMPLATE_FILE, "r") as f:
    templates = json.load(f)

if not templates:
    print(f"{TEMPLATE_FILE} exists but has no seals saved yet.")
    exit()

print(f"Loaded {len(templates)} seal templates: {list(templates.keys())}")
print("Controls: ESC = quit")

cap = cv2.VideoCapture(0)

stable_count = 0
last_candidate = None
locked_seal = None
locked_dist = None

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detection = hands.process(frame_rgb)

    both_hands = {}
    if detection.multi_hand_landmarks and detection.multi_handedness:
        for lm, handedness in zip(detection.multi_hand_landmarks, detection.multi_handedness):
            label = classify_left_right(handedness.classification[0].label)
            both_hands[label] = lm
            mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

    if "Left" in both_hands and "Right" in both_hands:
        live_left = normalize_hand(both_hands["Left"])
        live_right = normalize_hand(both_hands["Right"])
        candidate, dist = match_seal(live_left, live_right, templates)

        if dist <= MATCH_THRESHOLD:
            if candidate == last_candidate:
                stable_count += 1
            else:
                stable_count = 1
                last_candidate = candidate

            if stable_count >= STABLE_FRAMES_REQUIRED:
                locked_seal = candidate
                locked_dist = dist
        else:
            stable_count = 0
            last_candidate = None
            locked_seal = None
    else:
        stable_count = 0
        last_candidate = None
        locked_seal = None

    # ---- UI ----
    if locked_seal:
        cv2.putText(frame, f"Seal: {locked_seal}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(frame, f"dist: {locked_dist:.2f}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    else:
        cv2.putText(frame, "No match", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    cv2.imshow("Live Seal Recognition", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()