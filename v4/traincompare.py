"""
train_compare.py
-----------------
Loads hand_seal_dataset.csv (built by collect_data.py) and trains/compares
three classical ML classifiers: KNN, Logistic Regression, Naive Bayes.

Key methodology point: we split train/test by SESSION, not by individual
row, using GroupShuffleSplit. This prevents near-duplicate frames from the
same recording session leaking across train and test, which would give
artificially inflated (fake) accuracy numbers.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(SCRIPT_DIR, "hand_seal_dataset.csv")
USE_PCA = True
PCA_COMPONENTS = 0.95  # keep enough components to explain 95% of variance

# ---- 1. Load data ----
df = pd.read_csv(DATA_CSV)
print(f"Loaded {len(df)} samples across {df['seal'].nunique()} seals:")
print(df['seal'].value_counts(), "\n")

feature_cols = [c for c in df.columns if c.startswith("L_") or c.startswith("R_")]
X = df[feature_cols].values
y = df["seal"].values
groups = df["session_id"].values  # this is what makes the split honest

# ---- 2. Group-aware train/test split ----
splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=groups))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")
# Sanity check: confirm no session appears in both sets
train_sessions = set(groups[train_idx])
test_sessions = set(groups[test_idx])
overlap = train_sessions & test_sessions
print(f"Session overlap between train/test: {len(overlap)} (should be 0)\n")

# ---- 3. Scale features (important for KNN and Logistic Regression) ----
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---- 4. Optional PCA ----
if USE_PCA:
    pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
    X_train_final = pca.fit_transform(X_train_scaled)
    X_test_final = pca.transform(X_test_scaled)
    print(f"PCA reduced {X_train_scaled.shape[1]} features -> "
          f"{X_train_final.shape[1]} components "
          f"(explaining {PCA_COMPONENTS*100:.0f}% variance)\n")
else:
    X_train_final = X_train_scaled
    X_test_final = X_test_scaled

# ---- 5. Define models ----
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": GaussianNB(),
}

# ---- 6. Train, evaluate, compare ----
results = []
for name, model in models.items():
    model.fit(X_train_final, y_train)
    y_pred = model.predict(X_test_final)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    results.append({"Model": name, "Accuracy": acc, "Precision": prec,
                     "Recall": rec, "F1-Score": f1})

    print(f"=== {name} ===")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred, labels=sorted(set(y)))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=sorted(set(y)), yticklabels=sorted(set(y)))
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel("True Seal")
    plt.xlabel("Predicted Seal")
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, f"confusion_matrix_{name.replace(' ', '_')}.png"))
    plt.close()

# ---- 7. Summary comparison table ----
results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
print("\n=== COMPARISON SUMMARY ===")
print(results_df.to_string(index=False))
results_df.to_csv(os.path.join(SCRIPT_DIR, "model_comparison_results.csv"), index=False)

best_model = results_df.iloc[0]["Model"]
print(f"\nBest performing model: {best_model}")
print("Confusion matrix images saved as confusion_matrix_<model>.png")
print("Comparison table saved as model_comparison_results.csv")