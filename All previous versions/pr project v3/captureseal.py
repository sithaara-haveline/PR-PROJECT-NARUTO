"""
Seal Template Capture
----------------------
Purpose: capture a reference "signature" (normalized hand landmarks) for
ONE seal at a time and save it into a shared templates.json file. Run this
once per seal you want to recognize later.

How it works:
1. Set SEAL_NAME below to the seal you're capturing.
2. Run the script. A live webcam feed opens.
3. Get into the seal pose, press SPACE to grab a burst of frames.
4. The script keeps only frames where both hands were detected, normalizes
   each hand's 21 landmarks (wrist-relative + scale-normalized), and
   averages across the good frames to build one clean template.
5. The template is saved/updated in seal_templates.json under SEAL_NAME.

Run this once per seal (Bird, Horse, Ox, Rat, Dog, Tiger, etc). Re-running
with the same SEAL_NAME overwrites that seal's old template.
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import os
import time

# ---- EDIT THIS before each run ----
SEAL_NAME = "Serpent"

TEMPLATE_FILE = "seal_templates.json"
BURST_SECONDS = 3
MIN_GOOD_FRAMES = 10  # need at least this many 2-hand frames to save a template

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
    """
    Convert 21 (x, y, z) landmarks into a wrist-relative, scale-normalized
    vector so it doesn't matter how far from the camera or where on screen
    the hand is. Returns a flat list of 63 numbers (21 points x 3 coords).
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
    wrist = pts[0].copy()
    pts -= wrist  # wrist-relative

    # scale by the distance from wrist to middle-finger MCP (landmark 9)
    # this is a stable "hand size" reference regardless of pose
    scale = np.linalg.norm(pts[9])
    if scale < 1e-6:
        scale = 1e-6
    pts /= scale

    return pts.flatten().tolist()


def classify_left_right(handedness_label, is_mirrored=True):
    """MediaPipe labels are from the camera's perspective; since we flip
    the frame for a mirror view, the label needs flipping back too."""
    if not is_mirrored:
        return handedness_label
    return "Left" if handedness_label == "Right" else "Right"


cap = cv2.VideoCapture(0)
captured_frames = []  # list of dicts: {"Left": [...], "Right": [...]}

recording = False
record_start_time = None

preparing = False
prepare_start_time = None
PREP_SECONDS = 3

print(f"Capturing template for seal: {SEAL_NAME}")
print("Controls: SPACE = start 3s capture burst | ESC = quit without saving")

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

    cv2.putText(frame, f"Seal: {SEAL_NAME}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if preparing:
        prep_elapsed = time.time() - prepare_start_time
        prep_remaining = PREP_SECONDS - prep_elapsed
        if prep_remaining > 0:
            cv2.putText(frame, f"Get ready... {prep_remaining:.1f}s", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
        else:
            preparing = False
            recording = True
            record_start_time = time.time()
            captured_frames = []
    elif not recording:
        cv2.putText(frame, "Press SPACE to capture (3s prep + 3s record)", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    else:
        elapsed = time.time() - record_start_time
        remaining = max(0, BURST_SECONDS - elapsed)
        cv2.putText(frame, f"Capturing... {remaining:.1f}s", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Hands: {list(both_hands.keys())}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if "Left" in both_hands and "Right" in both_hands:
            captured_frames.append({
                "Left": normalize_hand(both_hands["Left"]),
                "Right": normalize_hand(both_hands["Right"]),
            })

        if elapsed >= BURST_SECONDS:
            recording = False
            print(f"Captured {len(captured_frames)} good (2-hand) frames out of the burst.")
            break

    cv2.imshow("Capture Seal Template", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        print("Cancelled, nothing saved.")
        cap.release()
        cv2.destroyAllWindows()
        exit()

    if key == 32 and not recording and not preparing:
        preparing = True
        prepare_start_time = time.time()

cap.release()
cv2.destroyAllWindows()

if len(captured_frames) < MIN_GOOD_FRAMES:
    print(f"\nOnly got {len(captured_frames)} good frames (need >= {MIN_GOOD_FRAMES}).")
    print("This seal may be hard to detect reliably. Try again, hold steadier,")
    print("or accept that this seal might not be a good candidate for this method.")
    exit()

# Average across captured frames to get one clean template per hand
left_avg = np.mean([f["Left"] for f in captured_frames], axis=0).tolist()
right_avg = np.mean([f["Right"] for f in captured_frames], axis=0).tolist()

# Load existing templates file (if any) and update/add this seal
templates = {}
if os.path.exists(TEMPLATE_FILE):
    with open(TEMPLATE_FILE, "r") as f:
        templates = json.load(f)

templates[SEAL_NAME] = {
    "Left": left_avg,
    "Right": right_avg,
    "num_frames_averaged": len(captured_frames),
}

with open(TEMPLATE_FILE, "w") as f:
    json.dump(templates, f, indent=2)

print(f"\nSaved template for '{SEAL_NAME}' to {TEMPLATE_FILE}")
print(f"Averaged over {len(captured_frames)} frames.")
print(f"Seals currently in file: {list(templates.keys())}")