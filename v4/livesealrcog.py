import json
import os
import cv2
import joblib
import mediapipe as mp
import numpy as np

# ---- 1. Load Saved Model & Metadata ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(SCRIPT_DIR, "best_seal_model.pkl")
SCALER_PATH = os.path.join(SCRIPT_DIR, "scaler.pkl")
PCA_PATH = os.path.join(SCRIPT_DIR, "pca.pkl")
METADATA_PATH = os.path.join(SCRIPT_DIR, "model_metadata.json")

print("Loading saved model and preprocessing tools...")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

use_pca = metadata.get("use_pca", False)
pca = joblib.load(PCA_PATH) if use_pca and os.path.exists(PCA_PATH) else None
expected_cols = metadata.get("feature_columns", [])

print(f"Model loaded successfully! Expecting {len(expected_cols)} features.")


# ---- 2. Normalization Function ----
def extract_normalized_features(hand_landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    wrist = coords[0]
    relative = coords - wrist
    dist = np.linalg.norm(relative[9])  # Distance to landmark 9 (Middle MCP)

    if dist > 0:
        normalized = relative / dist
    else:
        normalized = relative

    return normalized.flatten()  # 63 features


# ---- 3. Initialize MediaPipe Hands ----
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

# ---- 4. Start Video Stream ----
cap = cv2.VideoCapture(0)

print("\nStarting camera... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    current_seal = "No Hands Detected"
    confidence_text = ""

    if results.multi_hand_landmarks and results.multi_handedness:
        feature_dict = {}

        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            # Draw skeleton on screen
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

            # Determine Hand Prefix: L_ or R_
            hand_label = handedness.classification[0].label  # 'Left' or 'Right'
            prefix = "L_" if hand_label == "Left" else "R_"

            # Normalize 21 3D points -> 63 float values
            flat_63 = extract_normalized_features(hand_landmarks)

            # Assign to dictionary keys L_0..L_62 / R_0..R_62
            for idx, val in enumerate(flat_63):
                feature_dict[f"{prefix}{idx}"] = val

        # Fill feature vector matching CSV header
        feature_vector = [feature_dict.get(col, 0.0) for col in expected_cols]
        feature_array = np.array(feature_vector).reshape(1, -1)

        # Scale & PCA Transform
        scaled_features = scaler.transform(feature_array)
        final_features = (
            pca.transform(scaled_features)
            if use_pca and pca is not None
            else scaled_features
        )

        # Predict Seal
        prediction = model.predict(final_features)[0]

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(final_features)
            confidence = np.max(probs) * 100
            current_seal = f"{prediction}"
            confidence_text = f"{confidence:.1f}%"
        else:
            current_seal = f"{prediction}"

    # Visual UI Overlay
    cv2.rectangle(frame, (10, 10), (450, 90), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"Seal: {current_seal}",
        (25, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    if confidence_text:
        cv2.putText(
            frame,
            f"Confidence: {confidence_text}",
            (25, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.imshow("Real-Time Hand Seal Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()