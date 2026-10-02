"""
train_compare.py
----------------
Compares KNN, Logistic Regression, Naive Bayes and a small neural network (MLP)
on the SAME features with the SAME evaluation, then saves one model for the
live app.

Evaluation = repeated 5-fold cross-validation grouped by session_id:
  * no session ever appears in both train and test (no leakage from
    near-identical consecutive frames)
  * mean +/- std over 15 folds, so you can see whether differences between
    models are real or just noise
  * confusion matrices / per-seal recall come from pooled out-of-fold
    predictions of the first repeat (every sample is predicted exactly once,
    by a model that never saw its session)

    python train_compare.py
    python train_compare.py --model KNN          # force the saved model
    python train_compare.py --data other.csv

Outputs (in this folder):
    model_comparison_results.csv, per_seal_recall.csv,
    confusion_matrix_<model>.png, model.pkl, model_metadata.json
"""

import argparse
import json
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import feature_columns

warnings.filterwarnings("ignore")

N_SPLITS = 5
N_REPEATS = 3
SEED = 42


def build_models():
    # Every model sits behind the same StandardScaler, inside a Pipeline, so the
    # scaler is fit on training folds only.
    return {
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "Naive Bayes": make_pipeline(StandardScaler(), GaussianNB()),
        # 126 -> 128 -> 64 -> classes, ReLU, Adam, L2 penalty (alpha)
        "Neural Network (MLP)": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=500, random_state=SEED),
        ),
    }


def grouped_folds(groups, repeat):
    """GroupKFold with the session->fold assignment shuffled differently per repeat."""
    rng = np.random.RandomState(SEED + repeat)
    uniq = np.unique(groups)
    perm = dict(zip(uniq, rng.permutation(len(uniq))))
    shuffled = np.array([perm[g] for g in groups])
    return GroupKFold(n_splits=N_SPLITS).split(groups, groups=shuffled)


def plot_confusion(cm, labels, title, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    ax.set_xlabel("Predicted seal")
    ax.set_ylabel("True seal")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="hand_seal_dataset_clean.csv")
    ap.add_argument("--model", default=None, help="force which model gets saved")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    X = df[feature_columns()].values
    y = df["seal"].values
    groups = df["session_id"].values
    labels = sorted(set(y))

    print(f"{len(df)} samples | {len(labels)} seals | {len(set(groups))} sessions")
    print(df.groupby("seal")["session_id"].nunique().rename("sessions per seal").to_string(), "\n")

    fold_scores = {name: [] for name in build_models()}
    oof_pred = {name: np.empty(len(y), dtype=object) for name in build_models()}

    for rep in range(N_REPEATS):
        for train_idx, test_idx in grouped_folds(groups, rep):
            for name, model in build_models().items():
                model.fit(X[train_idx], y[train_idx])
                pred = model.predict(X[test_idx])
                fold_scores[name].append(
                    (
                        accuracy_score(y[test_idx], pred),
                        precision_score(y[test_idx], pred, average="macro", zero_division=0),
                        recall_score(y[test_idx], pred, average="macro", zero_division=0),
                        f1_score(y[test_idx], pred, average="macro", zero_division=0),
                    )
                )
                if rep == 0:
                    oof_pred[name][test_idx] = pred

    # ---- summary table ----
    rows = []
    for name, sc in fold_scores.items():
        a = np.array(sc)
        rows.append(
            {
                "Model": name,
                "Accuracy": a[:, 0].mean(), "Acc_std": a[:, 0].std(),
                "Precision": a[:, 1].mean(),
                "Recall": a[:, 2].mean(),
                "F1 (macro)": a[:, 3].mean(), "F1_std": a[:, 3].std(),
            }
        )
    res = pd.DataFrame(rows).sort_values("F1 (macro)", ascending=False)
    res.to_csv("model_comparison_results.csv", index=False)
    print("=== COMPARISON (mean over %d grouped folds) ===" % (N_SPLITS * N_REPEATS))
    print(res.round(3).to_string(index=False))

    best, second = res.iloc[0], res.iloc[1]
    if best["F1 (macro)"] - second["F1 (macro)"] < max(best["F1_std"], second["F1_std"]):
        print(f"\nNOTE: '{best['Model']}' and '{second['Model']}' differ by less than one fold-to-fold "
              "standard deviation -> treat them as tied and say so in your report.")

    # ---- confusion matrices + per-seal recall (pooled out-of-fold) ----
    recall_table = {}
    for name in fold_scores:
        cm = confusion_matrix(y, oof_pred[name], labels=labels)
        plot_confusion(cm, labels, f"{name} (out-of-fold, grouped by session)",
                       f"confusion_matrix_{name.split(' (')[0].replace(' ', '_')}.png")
        recall_table[name] = (cm.diagonal() / cm.sum(axis=1)).round(3)
    per_seal = pd.DataFrame(recall_table, index=labels)
    per_seal.to_csv("per_seal_recall.csv")
    print("\n=== PER-SEAL RECALL (which seals are actually reliable?) ===")
    print(per_seal.to_string())

    # ---- fit the chosen model on ALL data and save ----
    chosen = args.model or best["Model"]
    if chosen not in build_models():
        raise SystemExit(f"--model must be one of {list(build_models())}")
    final = build_models()[chosen].fit(X, y)
    joblib.dump({"pipeline": final, "classes": list(final.classes_), "name": chosen}, "model.pkl")
    with open("model_metadata.json", "w") as f:
        json.dump({"model": chosen, "classes": list(final.classes_), "n_features": int(X.shape[1])}, f, indent=2)
    print(f"\nSaved model.pkl  ({chosen}, trained on all {len(df)} samples)")


if __name__ == "__main__":
    main()
