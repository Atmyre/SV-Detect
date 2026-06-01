"""Train the final logreg detector at the best-val C, then expose its per-layer
weights so we can see which layers carry the prediction.

Reports two views:
  1. raw |coef| per layer  -- which features the model put weight on (l1 zeros out the rest)
  2. |coef| * std(feature) -- actual contribution to the logit (scale-aware)

Optionally writes a TSV per file_tag.
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--method", default="logreg", choices=["mean", "pca", "logreg"])
    p.add_argument("--suffix", default="")
    p.add_argument("--C", type=float, required=True,
                   help="Use the best-val C from the earlier sweep.")
    p.add_argument("--penalty", default="l1", choices=["l1", "l2"])
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--out-tsv", default=None)
    return p.parse_args()


def load_xy(data_dir: str, split: str, file_tag: str):
    real = np.load(os.path.join(data_dir, f"real_{split}_dot_products_{file_tag}.npy"))
    fake = np.load(os.path.join(data_dir, f"fake_{split}_dot_products_{file_tag}.npy"))
    X = np.concatenate([real, fake], axis=0)
    y = np.concatenate([np.zeros(len(real)), np.ones(len(fake))], axis=0)
    return X, y


def main():
    args = parse_args()
    file_tag = f"{args.method}{args.suffix}"
    print(f"file_tag={file_tag}  C={args.C}  penalty={args.penalty}")

    X_train, y_train = load_xy(args.data_dir, "train", file_tag)
    print(f"train {X_train.shape}")

    solver = "saga" if args.penalty == "l1" else "lbfgs"
    clf = LogisticRegression(
        solver=solver, penalty=args.penalty, C=args.C,
        max_iter=1000, random_state=42,
    )
    clf.fit(X_train, y_train)
    coef = clf.coef_[0]                        # (L,)
    feat_std = X_train.std(axis=0)
    contribution = np.abs(coef) * feat_std

    L = len(coef)
    rows = []
    for layer in range(L):
        rows.append({
            "layer": layer,
            "coef": coef[layer],
            "abs_coef": abs(coef[layer]),
            "feat_std": feat_std[layer],
            "contribution": contribution[layer],
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    nonzero = (np.abs(coef) > 1e-8).sum()
    print(f"\nnon-zero coefs: {nonzero} / {L}")

    print(f"\nTop {args.top_k} layers by |coef| (model selection):")
    for i in np.argsort(-np.abs(coef))[:args.top_k]:
        if abs(coef[i]) < 1e-8:
            break
        sign = "fake-leaning" if coef[i] > 0 else "human-leaning"
        print(f"  layer {i:>2}: coef={coef[i]:+.4f}  ({sign})")

    print(f"\nTop {args.top_k} layers by contribution (|coef|*std):")
    for i in np.argsort(-contribution)[:args.top_k]:
        if contribution[i] < 1e-8:
            break
        sign = "fake-leaning" if coef[i] > 0 else "human-leaning"
        print(f"  layer {i:>2}: contribution={contribution[i]:.4f}  coef={coef[i]:+.4f}  std={feat_std[i]:.4f}  ({sign})")

    if args.out_tsv:
        df.to_csv(args.out_tsv, sep="\t", index=False)
        print(f"\nwrote {args.out_tsv}")


if __name__ == "__main__":
    main()
