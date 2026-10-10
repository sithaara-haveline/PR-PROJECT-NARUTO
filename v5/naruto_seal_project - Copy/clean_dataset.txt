"""
clean_dataset.py
----------------
One-time cleanup of your OLD hand_seal_dataset.csv.

  1. Drops the 'wrist_distance' column (it was invented from the seal label by
     addwristdata.py -> label leakage).
  2. Drops every row where a hand was zero-padded (only one hand detected).
  3. Reports how much data each seal has left, and drops seals that end up too
     small to train on (default: < 100 rows) - those need re-collecting with
     collect_data.py.

    python clean_dataset.py
    python clean_dataset.py --in hand_seal_dataset.csv --out hand_seal_dataset_clean.csv --min-rows 100

Your original file is never modified.
"""

import argparse

import pandas as pd

from common import HAND_DIM, feature_columns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="hand_seal_dataset_clean.csv")
    ap.add_argument("--out", default="hand_seal_dataset_clean2.csv")
    ap.add_argument("--min-rows", type=int, default=100)
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    if "wrist_distance" in df.columns:
        df = df.drop(columns=["wrist_distance"])
        print("Dropped 'wrist_distance' column (was derived from labels).")

    l_cols = [f"L_{i}" for i in range(HAND_DIM)]
    r_cols = [f"R_{i}" for i in range(HAND_DIM)]
    has_l = df[l_cols].abs().sum(axis=1) > 0
    has_r = df[r_cols].abs().sum(axis=1) > 0
    both = has_l & has_r

    summary = pd.DataFrame(
        {
            "rows_before": df.groupby("seal").size(),
            "rows_both_hands": df[both].groupby("seal").size(),
            "sessions_left": df[both].groupby("seal")["session_id"].nunique(),
        }
    ).fillna(0).astype(int)
    summary["kept_%"] = (100 * summary["rows_both_hands"] / summary["rows_before"]).round(1)
    print("\n" + summary.sort_values("rows_both_hands").to_string())

    clean = df[both].copy()
    too_small = summary[summary["rows_both_hands"] < args.min_rows].index.tolist()
    if too_small:
        clean = clean[~clean["seal"].isin(too_small)]
        print(f"\nDropped seals with < {args.min_rows} clean rows: {too_small}")
        print("-> re-collect these with:  python collect_data.py --seal <name>")

    clean = clean[["seal", "session_id", "condition"] + feature_columns()]
    clean.to_csv(args.out, index=False)
    print(f"\nWrote {len(clean)} rows, {clean['seal'].nunique()} seals -> {args.out}")


if __name__ == "__main__":
    main()
