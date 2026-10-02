"""
train_compare.py
-----------------
Loads hand_seal_dataset.csv (built by collect_data.py) and trains/compares
three classical ML classifiers: KNN, Logistic Regression, Naive Bayes.
"""

import json
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(SCRIPT_DIR, "hand_seal_dataset.csv")

# Disabled PCA so we use all 126 raw landmark features
USE_PCA = False
PCA_COMPONENTS = 0.99 

# ---- 1. Load data ----
df = pd.read_csv(DATA_CSV)
print(f"Loaded {len(df)} samples across {df['seal'].nunique()} seals:")
print(df["seal"].value_counts(), "\n")

feature_cols = [
    c for c in df.columns if c.startswith("L_") or c.startswith("R_")
]
X = df[feature_cols].values
y = df["seal"].values
groups = df["session_id"].values 

# ---- 2. Group-aware train/test split ----
splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=groups))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")
train_sessions = set(groups[train_idx])
test_sessions = set(groups[test_idx])
overlap = train_sessions & test_sessions
print(f"Session overlap between train/test: {len(overlap)} (should be 0)\n")

# ---- 3. Scale features ----
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---- 4. Optional PCA ----
pca = None
if USE_PCA:
    pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
    X_train_final = pca.fit_transform(X_train_scaled)
    X_test_final = pca.transform(X_test_scaled)
    print(
        f"PCA reduced {X_train_scaled.shape[1]} features -> "
        f"{X_train_final.shape[1]} components\n"
    )
else:
    X_train_final = X_train_scaled
    X_test_final = X_test_scaled
    print(
        f"PCA Disabled. Using all {X_train_scaled.shape[1]} raw landmark features.\n"
    )

# ---- 5. Define models ----
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": GaussianNB(),
}

# ---- 6. Train, evaluate, compare ----
results = []
trained_models = {}

for name, model in models.items():
    model.fit(X_train_final, y_train)
    y_pred = model.predict(X_test_final)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    results.append(
        {
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
        }
    )
    trained_models[name] = model

    print(f"=== {name} ===")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred, labels=sorted(set(y)))
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=sorted(set(y)),
        yticklabels=sorted(set(y)),
    )
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel("True Seal")
    plt.xlabel("Predicted Seal")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            SCRIPT_DIR, f"confusion_matrix_{name.replace(' ', '_')}.png"
        )
    )
    plt.close()

# ---- 7. Summary comparison table & Model Saving ----
results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
print("\n=== COMPARISON SUMMARY ===")
print(results_df.to_string(index=False))
results_df.to_csv(
    os.path.join(SCRIPT_DIR, "model_comparison_results.csv"), index=False
)

best_model_name = results_df.iloc[0]["Model"]
best_model_obj = trained_models[best_model_name]

print(f"\nBest performing model: {best_model_name}")

# Save the trained artifacts for real-time recognition
joblib.dump(best_model_obj, os.path.join(SCRIPT_DIR, "best_seal_model.pkl"))
joblib.dump(scaler, os.path.join(SCRIPT_DIR, "scaler.pkl"))

if USE_PCA and pca is not None:
    joblib.dump(pca, os.path.join(SCRIPT_DIR, "pca.pkl"))

# Save feature metadata for live prediction scripts
metadata = {
    "feature_columns": feature_cols,
    "num_features": len(feature_cols),
    "use_pca": USE_PCA,
    "best_model_name": best_model_name,
}

with open(os.path.join(SCRIPT_DIR, "model_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=4)

print("\nSaved artifacts:")
print(" - Model: best_seal_model.pkl")
print(" - Scaler: scaler.pkl")
if USE_PCA:
    print(" - PCA: pca.pkl")
print(" - Metadata: model_metadata.json")