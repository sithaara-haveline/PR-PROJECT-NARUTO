"""
Seal Detection Screening Test
------------------------------
Purpose: test each candidate hand seal to see if MediaPipe reliably
detects BOTH hands while you hold the pose. Use this to decide your
final 7-8 seal list based on real data, not guesswork.

How it works:
1. A seal name appears on screen.
2. Press SPACE to start a 3-second recording window for that seal.
3. Hold the seal steady during those 3 seconds.
4. The script logs, per frame, how many hands were detected.
5. At the end, it prints a PASS/FAIL summary per seal + saves a CSV.

PASS threshold: hands detected in >=80% of frames during the window.
(You can tighten/loosen this later once you understand your data.)
"""

import cv2
import mediapipe as mp
import time
import csv

# ---- EDIT THIS LIST with your candidate seals ----
CANDIDATE_SEALS = [
    "Boar", "Bird", "Tiger", "Ram", "Serpent",
    "Dragon", "Dog", "Horse", "Monkey", "Ox", "Hare", "Rat"
]

RECORD_SECONDS = 5
PASS_THRESHOLD = 0.65  # 80% of frames must show 2 hands

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
results_log = []  # (seal_name, frames_total, frames_with_2_hands, pass_fail)

seal_index = 0
recording = False
record_start_time = None
frame_count = 0
two_hand_count = 0

print("Controls: SPACE = start recording current seal | ESC = quit")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)  # mirror view, easier to pose against
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detection = hands.process(frame_rgb)

    num_hands_detected = 0
    if detection.multi_hand_landmarks:
        num_hands_detected = len(detection.multi_hand_landmarks)
        for hand_landmarks in detection.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # ---- UI overlay ----
    if seal_index < len(CANDIDATE_SEALS):
        current_seal = CANDIDATE_SEALS[seal_index]
        cv2.putText(frame, f"Seal: {current_seal}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if not recording:
            cv2.putText(frame, "Press SPACE to record (5s)", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            elapsed = time.time() - record_start_time
            remaining = max(0, RECORD_SECONDS - elapsed)
            cv2.putText(frame, f"Recording... {remaining:.1f}s", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Hands seen: {num_hands_detected}", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            frame_count += 1
            if num_hands_detected == 2:
                two_hand_count += 1

            if elapsed >= RECORD_SECONDS:
                pass_rate = two_hand_count / frame_count if frame_count else 0
                verdict = "PASS" if pass_rate >= PASS_THRESHOLD else "FAIL"
                results_log.append((current_seal, frame_count, two_hand_count, f"{pass_rate:.0%}", verdict))
                print(f"{current_seal}: {two_hand_count}/{frame_count} frames "
                      f"({pass_rate:.0%}) -> {verdict}")

                recording = False
                frame_count = 0
                two_hand_count = 0
                seal_index += 1
    else:
        cv2.putText(frame, "All seals tested! Press ESC to quit.", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Seal Screening Test", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break
    if key == 32 and not recording and seal_index < len(CANDIDATE_SEALS):  # SPACE
        recording = True
        record_start_time = time.time()
        frame_count = 0
        two_hand_count = 0

cap.release()
cv2.destroyAllWindows()

# ---- Save results ----
if results_log:
    with open("seal_screening_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seal", "total_frames", "two_hand_frames", "pass_rate", "verdict"])
        writer.writerows(results_log)
    print("\nSaved results to seal_screening_results.csv")
    print("\n--- SUMMARY ---")
    for row in results_log:
        print(row)