"""
collect_data.py
----------------
Collects individually-labeled hand seal samples across 9 varied conditions
(distance / tilt / frame position) to build a proper training dataset for
KNN, Logistic Regression, and Naive Bayes comparison.

Unlike captureseal.py (which averages frames into one template), this
script saves EVERY valid frame as its own row, tagged with a session_id.
This is required because:
  - KNN/LogReg/NaiveBayes need multiple real samples per class, not one
    averaged template.
  - session_id lets us later split train/test by SESSION, not by frame,
    avoiding data leakage from near-identical consecutive frames.

Run this once per seal. Data appends to hand_seal_dataset.csv, so you can
run it across multiple sessions/days without losing earlier progress.
"""

import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time

# ---- EDIT THIS before each run ----
SEAL_NAME = "Rat"   # change this for each seal you're collecting

OUTPUT_CSV = "hand_seal_dataset.csv"
FRAMES_PER_CONDITION = 25
PREP_SECONDS = 5
MIN_DETECTION_CONF = 0.5
MIN_TRACKING_CONF = 0.3

# 9 conditions mixing distance, tilt, and frame position.
# Read each one aloud to yourself before recording so you actually vary it.
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

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=MIN_DETECTION_CONF,
    min_tracking_confidence=MIN_TRACKING_CONF
)


def normalize_hand(landmarks):
    """
    Converts 21 raw MediaPipe landmarks into a position- and
    scale-invariant 63-value feature vector.
    - Wrist-relative: removes dependence on hand position in frame.
    - Scaled by wrist-to-middle-MCP distance: removes dependence on
      distance from camera.
    NOTE: does NOT remove rotation dependence -- that's why we
    deliberately vary tilt during collection instead.
    """
    wrist = landmarks[0]
    pts = np.array([[lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z]
                     for lm in landmarks])
    scale = np.linalg.norm(pts[9])  # wrist->middle MCP distance
    if scale < 1e-6:
        scale = 1e-6
    pts = pts / scale
    return pts.flatten()  # 63 values


def get_handedness_label(handedness_classification, mirrored=True):
    """Corrects MediaPipe's Left/Right label for our mirrored webcam view."""
    label = handedness_classification.label
    if mirrored:
        return "Right" if label == "Left" else "Left"
    return label


def ensure_csv_header():
    """Creates the CSV with a header row if it doesn't exist yet."""
    if not os.path.exists(OUTPUT_CSV):
        header = ["seal", "session_id", "condition"]
        header += [f"L_{i}" for i in range(63)]
        header += [f"R_{i}" for i in range(63)]
        with open(OUTPUT_CSV, "w", newline="") as f:
            csv.writer(f).writerow(header)


def save_sample(seal, session_id, condition, left_vec, right_vec):
    row = [seal, session_id, condition] + list(left_vec) + list(right_vec)
    with open(OUTPUT_CSV, "a", newline="") as f:
        csv.writer(f).writerow(row)


ensure_csv_header()
cap = cv2.VideoCapture(0)

condition_index = 0
collecting = False
preparing = False
prepare_start_time = None
frames_collected = 0
session_id = None

print(f"Collecting data for seal: {SEAL_NAME}")
print("Controls: SPACE = start collecting current condition | ESC = quit\n")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detection = hands.process(frame_rgb)

    left_vec, right_vec = None, None
    if detection.multi_hand_landmarks and detection.multi_handedness:
        for hand_landmarks, handedness in zip(
                detection.multi_hand_landmarks, detection.multi_handedness):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            side = get_handedness_label(handedness.classification[0])
            vec = normalize_hand(hand_landmarks.landmark)
            if side == "Left":
                left_vec = vec
            else:
                right_vec = vec

    both_hands_ok = left_vec is not None and right_vec is not None

    if preparing and time.time() - prepare_start_time >= PREP_SECONDS:
        preparing = False
        collecting = True
        frames_collected = 0
        session_id = f"{SEAL_NAME}_{condition_index}_{int(time.time())}"

    # ---- UI ----
    if condition_index < len(CONDITIONS):
        condition_text = CONDITIONS[condition_index]
        cv2.putText(frame, f"Seal: {SEAL_NAME}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Condition {condition_index+1}/9: {condition_text}",
                    (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        if preparing:
            remaining = max(0, PREP_SECONDS - (time.time() - prepare_start_time))
            cv2.putText(frame, f"Get ready... {remaining:.1f}s", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        elif not collecting:
            cv2.putText(frame, "Get into position, press SPACE", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(frame, f"Collecting: {frames_collected}/{FRAMES_PER_CONDITION}",
                        (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            status = "OK" if both_hands_ok else "Both hands not detected"
            cv2.putText(frame, status, (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if both_hands_ok else (0, 0, 255), 2)

            if both_hands_ok:
                save_sample(SEAL_NAME, session_id, condition_text, left_vec, right_vec)
                frames_collected += 1

            if frames_collected >= FRAMES_PER_CONDITION:
                collecting = False
                frames_collected = 0
                condition_index += 1
    else:
        cv2.putText(frame, f"Done! All 9 conditions collected for {SEAL_NAME}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, "Change SEAL_NAME in the script and rerun for the next seal",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("Data Collection", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break
    if key == 32 and not collecting and not preparing and condition_index < len(CONDITIONS):  # SPACE
        preparing = True
        prepare_start_time = time.time()

cap.release()
cv2.destroyAllWindows()
print(f"\nSaved to {OUTPUT_CSV}")