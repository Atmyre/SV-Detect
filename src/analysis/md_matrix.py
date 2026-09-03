"""Reproduce paper Table 8 top block (DetectRL Multi-Domain cross-source F1
matrix, one classifier per row) from our extraction.

Given the dot-product files produced by compute_dots_bulk.py, for each
train_subset (SV source):
  1. Train StandardScaler + LogReg on (train_real, train_fake) dots projected onto that SV
  2. For each eval_subset, evaluate on (test_real, test_fake) dots projected onto the same SV
  3. Report AUROC + F1 (mimicking metrics from the notebook)
Outputs a wide CSV plus a per-cell JSON so we can compare against paper table.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from src.extract.nb_pipeline import build_detector, metrics


def load_train(dots_dir, train_sub, method):
    real = np.load(os.path.join(
        dots_dir,
        f"{train_sub}_train_real_dot_products_{train_sub}_steering_vectors_{method}.npy",
    ))
    fake = np.load(os.path.join(
        dots_dir,
        f"{train_sub}_train_fake_dot_products_{train_sub}_steering_vectors_{method}.npy",
    ))
    X = np.concatenate([real, fake], axis=0)
    y = np.concatenate([np.zeros(len(real), np.int8), np.ones(len(fake), np.int8)])
    return X, y


def load_test(dots_dir, eval_sub, sv_sub, method):
    real = np.load(os.path.join(
        dots_dir,
        f"{eval_sub}_test_real_dot_products_{sv_sub}_steering_vectors_{method}.npy",
    ))
    fake = np.load(os.path.join(
        dots_dir,
        f"{eval_sub}_test_fake_dot_products_{sv_sub}_steering_vectors_{method}.npy",
    ))
    X = np.concatenate([real, fake], axis=0)
    y = np.concatenate([np.zeros(len(real), np.int8), np.ones(len(fake), np.int8)])
    return X, y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dots-dir", required=True)
    p.add_argument("--subsets", nargs="+", required=True)
    p.add_argument("--method", default="logreg")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    results = {}
    auroc_mat = pd.DataFrame(index=args.subsets, columns=args.subsets, dtype=float)
    f1_mat    = pd.DataFrame(index=args.subsets, columns=args.subsets, dtype=float)

    for train_sub in args.subsets:
        Xtr, ytr = load_train(args.dots_dir, train_sub, args.method)
        print(f"[train {train_sub}] X={Xtr.shape} y_pos={ytr.sum()}")
        clf = build_detector().fit(Xtr, ytr)

        for eval_sub in args.subsets:
            Xte, yte = load_test(args.dots_dir, eval_sub, train_sub, args.method)
            probs = clf.predict_proba(Xte)[:, 1]
            m = metrics(yte, probs)
            key = f"{train_sub}->{eval_sub}"
            results[key] = m
            auroc_mat.loc[train_sub, eval_sub] = m["AUROC"]
            f1_mat.loc[train_sub, eval_sub]    = m["F1"]
            print(f"  {key:35s} AUROC={m['AUROC']:.4f}  F1={m['F1']:.4f}")

    auroc_mat.to_csv(os.path.join(args.out_dir, f"md_auroc_{args.method}.csv"))
    f1_mat.to_csv(   os.path.join(args.out_dir, f"md_f1_{args.method}.csv"))
    with open(os.path.join(args.out_dir, f"md_full_{args.method}.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nAUROC matrix:")
    print(auroc_mat.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nF1 matrix:")
    print(f1_mat.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
