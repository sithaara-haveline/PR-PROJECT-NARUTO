import json
import os
import cv2
import joblib
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn


class HandSealNN(nn.Module):

    def __init__(self, input_dim, num_classes):
        super(HandSealNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(SCRIPT_DIR, "nn_metadata.json"), "r") as f:
    metadata = json.load(f)

scaler = joblib.load(os.path.join(SCRIPT_DIR, "nn_scaler.pkl"))
label_encoder = joblib.load(os.path.join(SCRIPT_DIR, "label_encoder.pkl"))
expected_cols = metadata["feature_columns"]

model = HandSealNN(metadata["input_dim"], len(metadata["classes"]))
model.load_state_dict(
    torch.load(
        os.path.join(SCRIPT_DIR, "seal_nn_model.pth"), weights_only=True
    )
)
model.eval()


def extract_normalized_features(hand_landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    wrist = coords[0]
    relative = coords - wrist

    dist = np.linalg.norm(relative[9])
    if dist > 0:
        normalized = relative / dist
    else:
        normalized = relative

    return normalized.flatten()


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)
print("\nStarting Live Recognition...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    current_seal = "No Hands Detected"
    conf_str = ""

    if results.multi_hand_landmarks:
        sorted_hands = sorted(
            results.multi_hand_landmarks, key=lambda lm: lm.landmark[0].x
        )

        feature_dict = {}
        has_l, has_r = 0.0, 0.0

        if len(sorted_hands) == 1:
            wrist_x = sorted_hands[0].landmark[0].x
            prefix = "L_" if wrist_x < 0.5 else "R_"
            if prefix == "L_":
                has_l = 1.0
            else:
                has_r = 1.0

            flat = extract_normalized_features(sorted_hands[0])
            for idx, val in enumerate(flat):
                feature_dict[f"{prefix}{idx}"] = val

            mp_drawing.draw_landmarks(
                frame,
                sorted_hands[0],
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

        elif len(sorted_hands) >= 2:
            has_l, has_r = 1.0, 1.0
            for prefix, hand_lm in zip(["L_", "R_"], sorted_hands[:2]):
                flat = extract_normalized_features(hand_lm)
                for idx, val in enumerate(flat):
                    feature_dict[f"{prefix}{idx}"] = val

                mp_drawing.draw_landmarks(
                    frame,
                    hand_lm,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

        vec = [feature_dict.get(col, 0.0) for col in expected_cols]
        vec.extend([has_l, has_r])

        scaled_vec = scaler.transform([vec])

        with torch.no_grad():
            tensor_in = torch.FloatTensor(scaled_vec)
            logits = model(tensor_in)
            probs = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)

            pct = conf.item() * 100
            pred_name = label_encoder.inverse_transform([pred.item()])[0]

            if pct > 40:
                current_seal = pred_name
            else:
                current_seal = f"Unsure ({pred_name})"
            conf_str = f"{pct:.1f}%"

    cv2.rectangle(frame, (10, 10), (420, 85), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"Seal: {current_seal}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    if conf_str:
        cv2.putText(
            frame,
            f"Confidence: {conf_str}",
            (20, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )

    cv2.imshow("Hand Seal Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()