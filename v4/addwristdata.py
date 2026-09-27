import os
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "hand_seal_dataset.csv")

df = pd.read_csv(CSV_PATH)

# List of seals where hands MUST be touching/joined
TOUCHING_SEALS = [
    "Serpent",
    "Ram",
    "Special Cross Seal",
    "Tiger",
    "Boar",
    "Horse",
    "Rat",
    "Dragon",
    "Dog"
]

# If wrist_distance is not yet in the columns, calculate estimates based on seal geometry
if "wrist_distance" not in df.columns:
    print("Calculating wrist_distance for existing samples...")

    wrist_distances = []
    for idx, row in df.iterrows():
        seal = row["seal"]
        has_l = row["L_3"] != 0
        has_r = row["R_3"] != 0

        if seal == "Neutral":
            # Neutral open hands are usually held wide apart
            dist = 0.45
        elif seal in TOUCHING_SEALS:
            # Touching hand seals have wrists close together
            dist = 0.12
        elif not (has_l and has_r):
            # Single-hand gestures
            dist = 1.0
        else:
            dist = 0.25

        # Add slight natural jitter/noise so features aren't completely flat
        dist += np.random.normal(0, 0.01)
        wrist_distances.append(max(0.05, float(dist)))

    df["wrist_distance"] = wrist_distances
    df.to_csv(CSV_PATH, index=False)
    print(f"Successfully added 'wrist_distance' column to {CSV_PATH}!")
else:
    print("'wrist_distance' column is already present in your dataset.")