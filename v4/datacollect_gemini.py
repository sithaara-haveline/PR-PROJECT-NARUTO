import cv2
import csv
import os
import time
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

SEAL_NAME = "Hare"  # Update for each seal
OUTPUT_CSV = "hand_seal_dataset12.csv"
FRAMES_PER_CONDITION = 25
PREP_SECONDS = 5
CAPTURE_INTERVAL_SEC = 0.1
MIN_DETECTION_CONF = 0.3  # Lowered to maintain tracking during hand overlap
MIN_TRACKING_CONF = 0.3

# Custom drawing styles for distinct visual tracking
left_style_landmarks = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=3)     # Red dots
left_style_connections = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)                    # Green lines

right_style_landmarks = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=3, circle_radius=3)  # Yellow dots
right_style_connections = mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2)               # Cyan lines

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

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=MIN_DETECTION_CONF,
    min_tracking_confidence=MIN_TRACKING_CONF,
)

ZERO_HAND = np.zeros(63)


def normalize_hand(landmarks):
    wrist = landmarks[0]
    pts = np.array(
        [[lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z] for lm in landmarks]
    )
    scale = np.linalg.norm(pts[9])  # Distance from wrist to middle MCP
    if scale < 1e-6:
        scale = 1e-6
    pts = pts / scale
    return pts.flatten()


def sort_hands_spatially(landmarks_list):
    if not landmarks_list:
        return None, None
    if len(landmarks_list) == 1:
        return landmarks_list[0], None

    x0 = np.mean([lm.x for lm in landmarks_list[0].landmark])
    x1 = np.mean([lm.x for lm in landmarks_list[1].landmark])

    if x0 < x1:
        return landmarks_list[0], landmarks_list[1]
    return landmarks_list[1], landmarks_list[0]


def ensure_csv_header():
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
last_capture_time = time.time()
frames_collected = 0
session_id = None

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detection = hands.process(frame_rgb)

    left_vec, right_vec = None, None

    if detection.multi_hand_landmarks:
        # Sort hands by horizontal position (Screen Left vs Screen Right)
        sorted_hands = sorted(
            detection.multi_hand_landmarks,
            key=lambda h: np.mean([lm.x for lm in h.landmark])
        )

        # Hand 1 (Screen Left - Green skeleton)
        h1 = sorted_hands[0]
        left_vec = normalize_hand(h1.landmark)
        mp_drawing.draw_landmarks(
            frame, h1, mp_hands.HAND_CONNECTIONS,
            left_style_landmarks, left_style_connections
        )

        # Hand 2 (Screen Right - Cyan skeleton)
        if len(sorted_hands) > 1:
            h2 = sorted_hands[1]
            right_vec = normalize_hand(h2.landmark)
            mp_drawing.draw_landmarks(
                frame, h2, mp_hands.HAND_CONNECTIONS,
                right_style_landmarks, right_style_connections
            )
        else:
            # Zero-pad second hand if occluded so data saving doesn't crash
            right_vec = np.zeros(63)
    # At least one hand visible is sufficient to record intersecting seals
    has_hand_data = (left_vec is not None) or (right_vec is not None)

    if preparing and (time.time() - prepare_start_time >= PREP_SECONDS):
        preparing = False
        collecting = True
        frames_collected = 0
        session_id = f"{SEAL_NAME}_{condition_index}_{int(time.time())}"

    if condition_index < len(CONDITIONS):
        condition_text = CONDITIONS[condition_index]
        cv2.putText(
            frame,
            f"Seal: {SEAL_NAME}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Condition {condition_index+1}/9: {condition_text}",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2,
        )

        if preparing:
            remaining = max(
                0.0, PREP_SECONDS - (time.time() - prepare_start_time)
            )
            cv2.putText(
                frame,
                f"Get ready... {remaining:.1f}s",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 165, 255),
                2,
            )
        elif not collecting:
            cv2.putText(
                frame,
                "Get into position, press SPACE",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
        else:
            cv2.putText(
                frame,
                f"Collecting: {frames_collected}/{FRAMES_PER_CONDITION}",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

            status = "OK" if has_hand_data else "No hands detected"
            status_color = (0, 255, 0) if has_hand_data else (0, 0, 255)
            cv2.putText(
                frame,
                status,
                (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                status_color,
                2,
            )

            if has_hand_data and (
                time.time() - last_capture_time >= CAPTURE_INTERVAL_SEC
            ):
                l_data = left_vec if left_vec is not None else ZERO_HAND
                r_data = right_vec if right_vec is not None else ZERO_HAND
                save_sample(SEAL_NAME, session_id, condition_text, l_data, r_data)
                frames_collected += 1
                last_capture_time = time.time()

            if frames_collected >= FRAMES_PER_CONDITION:
                collecting = False
                frames_collected = 0
                condition_index += 1
    else:
        cv2.putText(frame, f"DONE! All 9 conditions collected for {SEAL_NAME}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, "Press ESC to exit, or change SEAL_NAME for next seal", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.imshow("Data Collection", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break
    if (
        key == 32
        and not collecting
        and not preparing
        and condition_index < len(CONDITIONS)
    ):
        preparing = True
        prepare_start_time = time.time()

cap.release()
cv2.destroyAllWindows()