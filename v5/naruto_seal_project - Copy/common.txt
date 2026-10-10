"""
common.py
---------
Single source of truth for hand detection + feature extraction.

collect_data.py and jutsu_app.py BOTH import from here, so the features the
model is trained on are computed exactly the way they are computed live.
(Your old collector and live script each had their own copy, and they placed a
single hand in different slots - that mismatch is now impossible.)

Feature vector = 126 numbers:
    [ screen-left hand : 21 landmarks x (x, y, z) ]  -> L_0 ... L_62
    [ screen-right hand: 21 landmarks x (x, y, z) ]  -> R_0 ... R_62
Each hand is made wrist-relative and scaled by the wrist -> middle-finger-base
distance, so position in the frame and distance from the camera don't matter.

STRICT RULE: a sample exists only if BOTH hands are detected. No zero-padding.

Needs:  pip install mediapipe==0.10.14 opencv-python numpy
"""

import numpy as np

# Same detection thresholds everywhere (collector + live).
DET_CONF = 0.5
TRACK_CONF = 0.5

N_LANDMARKS = 21
HAND_DIM = N_LANDMARKS * 3  # 63


def feature_columns():
    return [f"L_{i}" for i in range(HAND_DIM)] + [f"R_{i}" for i in range(HAND_DIM)]


def make_hands():
    """Create the MediaPipe Hands detector (import here so other modules
    that only need the maths don't require mediapipe)."""
    import mediapipe as mp

    return mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=DET_CONF,
        min_tracking_confidence=TRACK_CONF,
    )


def normalize_hand(landmarks):
    """21 MediaPipe landmarks -> 63 wrist-relative, scale-normalised numbers."""
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float64)
    pts = pts - pts[0]  # wrist becomes the origin
    scale = np.linalg.norm(pts[9])  # wrist -> middle finger base
    if scale < 1e-6:
        return None
    return (pts / scale).flatten()


def sort_hands(multi_hand_landmarks):
    """Order hands left-to-right ON SCREEN (by wrist x). The camera image is
    mirrored before detection, so this is what the user sees."""
    return sorted(multi_hand_landmarks, key=lambda h: h.landmark[0].x)


def extract_features(results):
    """
    results: output of hands.process(...)

    Returns (feature_vector_or_None, sorted_hands_list)
        feature_vector is a length-126 numpy array only when exactly two hands
        are visible; otherwise None (never zero-padded).
    """
    if not results.multi_hand_landmarks:
        return None, []

    hands_sorted = sort_hands(results.multi_hand_landmarks)
    if len(hands_sorted) < 2:
        return None, hands_sorted

    left = normalize_hand(hands_sorted[0].landmark)
    right = normalize_hand(hands_sorted[1].landmark)
    if left is None or right is None:
        return None, hands_sorted
    return np.concatenate([left, right]), hands_sorted[:2]
