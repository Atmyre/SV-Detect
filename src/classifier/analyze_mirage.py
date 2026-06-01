"""Mirage cross-evaluation + per-layer/per-direction weight inspection.

Mirage feature layout for each method ∈ {mean, pca, logreg_None, logreg_l1, logreg_l2}:
  - 3 per-task unit-norm steering vectors (generate / polish / rewrite),
    Gram-Schmidt'd per layer into an orthonormal triplet → shape (L, H, 3).
  - Each sample is projected onto these 3 directions per layer →
    96-dim feature (32 layers × 3 directions), saved as `*_dot_products_{method}.npy`.

Train per method:
  StandardScaler + LogisticRegression(C ∈ {0.001..100}, l1 or l2)
fit on the 800+800 train set, evaluated on the 6 test sets:
  {DIG, SIG} × {generate, polish, rewrite}

Reports both raw |coef| and "contribution" = |coef| × std(feature). Dumps a
per-(method, layer, dir) weight TSV for each method.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    f1_score, matthews_corrcoef, roc_auc_score, roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


METHODS = ["mean", "pca", "logreg_None", "logreg_l1", "logreg_l2"]
TEST_SOURCES = ["DIG", "SIG"]
TASKS = ["generate", "polish", "rewrite"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True,
                   help="Mirage folder with train_*, *_dot_products_*.npy, orthonormal_steering_vectors_*.npy")
    p.add_argument("--out-tsv-dir", default="interpret_out/mirage")
    p.add_argument("--cs", nargs="+", type=float, default=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
    p.add_argument("--penalty", default="l2", choices=["l1", "l2"])
    return p.parse_args()


def load_train(data_dir, method):
    real = np.load(os.path.join(data_dir, f"train_real_dot_products_{method}.npy"))
    fake = np.load(os.path.join(data_dir, f"train_fake_dot_products_{method}.npy"))
    X = np.concatenate([real, fake])
    y = np.concatenate([np.zeros(len(real)), np.ones(len(fake))])
    return X, y


def load_test(data_dir, source, task, method):
    real = np.load(os.path.join(data_dir, f"{source}_{task}_test_real_dot_products_{method}.npy"))
    fake = np.load(os.path.join(data_dir, f"{source}_{task}_test_fake_dot_products_{method}.npy"))
    X = np.concatenate([real, fake])
    y = np.concatenate([np.zeros(len(real)), np.ones(len(fake))])
    return X, y


def evaluate(model, scaler, X, y, threshold=0.5):
    Xs = scaler.transform(X)
    proba = model.predict_proba(Xs)[:, 1]
    pred = proba > threshold
    fpr, tpr, _ = roc_curve(y, proba)
    return {
        "accuracy":          accuracy_score(y, pred),
        "f1_macro":          f1_score(y, pred, average="macro"),
        "auroc":             roc_auc_score(y, proba),
        "aupr":              average_precision_score(y, proba),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "mcc":               matthews_corrcoef(y, pred),
        "tpr_at_fpr_5pct":   float(np.interp(0.05, fpr, tpr)),
    }


def main():
    args = parse_args()
    os.makedirs(args.out_tsv_dir, exist_ok=True)

    metrics_rows = []
    weight_rows = []

    solver = "saga" if args.penalty == "l1" else "lbfgs"

    for method in METHODS:
        print(f"\n================ method={method} ================")
        X_train, y_train = load_train(args.data_dir, method)
        print(f"  train: {X_train.shape}")

        # Pick C by best F1 on a held-out 20% slice of train (no separate val split)
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(y_train))
        cut = int(0.8 * len(y_train))
        tr_idx, val_idx = idx[:cut], idx[cut:]

        best_f1 = -1
        best_pipe = None
        best_C = None
        for C in args.cs:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("logreg", LogisticRegression(C=C, penalty=args.penalty, solver=solver,
                                              max_iter=2000, random_state=42)),
            ])
            pipe.fit(X_train[tr_idx], y_train[tr_idx])
            f1 = f1_score(
                y_train[val_idx],
                pipe.predict(X_train[val_idx]),
                average="macro",
            )
            if f1 > best_f1:
                best_f1 = f1; best_pipe = pipe; best_C = C
        # Refit on full train at best C
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(C=best_C, penalty=args.penalty, solver=solver,
                                          max_iter=2000, random_state=42)),
        ])
        pipe.fit(X_train, y_train)
        scaler = pipe["scaler"]
        coef = pipe["logreg"].coef_[0]                     # (96,)
        feat_std_pre = X_train.std(axis=0)
        contribution = np.abs(coef) * feat_std_pre
        print(f"  best C={best_C}  intercept={pipe['logreg'].intercept_[0]:.3f}")
        print(f"  |coef|.sum={abs(coef).sum():.2f}, max contrib={contribution.max():.3f}")

        # Eval on every test set
        for source in TEST_SOURCES:
            for task in TASKS:
                X_test, y_test = load_test(args.data_dir, source, task, method)
                m = evaluate(pipe["logreg"], scaler, X_test, y_test)
                metrics_rows.append({
                    "method": method, "source": source, "task": task,
                    "best_C": best_C, "n_test": len(y_test), **m,
                })
                print(f"    → {source}_{task}: acc={m['accuracy']:.4f} f1={m['f1_macro']:.4f} auroc={m['auroc']:.4f}")

        # Weights TSV: 32 layers × 3 directions, 96 entries
        for i, c in enumerate(coef):
            layer = i // 3
            direction = i % 3
            task_for_dir = TASKS[direction]      # informal; orthonormalization may have rotated
            weight_rows.append({
                "method": method, "layer": layer, "direction": direction,
                "task_in_orig_basis": task_for_dir,
                "coef": float(c),
                "feat_std_pre": float(feat_std_pre[i]),
                "contribution": float(contribution[i]),
            })

    # Save TSVs
    md = pd.DataFrame(metrics_rows)
    md_path = os.path.join(args.out_tsv_dir, "mirage_cross_eval_metrics.tsv")
    md.to_csv(md_path, sep="\t", index=False)
    print(f"\nwrote {md_path}")

    wd = pd.DataFrame(weight_rows)
    wd_path = os.path.join(args.out_tsv_dir, "mirage_weights.tsv")
    wd.to_csv(wd_path, sep="\t", index=False)
    print(f"wrote {wd_path}")

    print("\n=== Method × {DIG, SIG} × {generate, polish, rewrite}  AUROC ===")
    pivot_auroc = md.pivot_table(
        index=["source", "task"], columns="method", values="auroc",
    )
    print(pivot_auroc.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n=== Method × {DIG, SIG} × {generate, polish, rewrite}  F1 macro ===")
    pivot_f1 = md.pivot_table(
        index=["source", "task"], columns="method", values="f1_macro",
    )
    print(pivot_f1.to_string(float_format=lambda x: f"{x:.4f}"))

    # Compare to published JSON (mean / pca / logreg_l2 only)
    pub_path = os.path.join(args.data_dir, "logistic_regression_metrics.json")
    if os.path.exists(pub_path):
        print("\n=== vs published metrics ===")
        with open(pub_path) as fp:
            pub = json.load(fp)
        for source in TEST_SOURCES:
            for task in TASKS:
                for m_short in ["mean", "logreg_l2", "pca"]:
                    key = f"{source} {task} {m_short} AUROC"
                    pub_val = pub["AUROC"].get(key, [None])
                    pub_val = pub_val[0] if isinstance(pub_val, list) else pub_val
                    ours = md[(md["source"] == source) & (md["task"] == task) & (md["method"] == m_short)]
                    ours_v = ours["auroc"].iloc[0] if len(ours) else None
                    print(f"  {source} {task} {m_short:>10}: pub={pub_val:.4f}  ours={ours_v:.4f}"
                          if pub_val and ours_v else f"  {source} {task} {m_short}: missing")


if __name__ == "__main__":
    main()
