"""Cross-evaluation + per-layer weight inspection for the GPT-Neo DetectRL
steering vectors and dot products shipped from the Drive folder.

For each of the 4 train datasets (`direct_prompt`, `paraphrase_attacks_llm`,
`perturbation_attacks_llm`, `prompt_attacks_llm`) we:
  1. Load the train dot products (using that dataset's own steering vector)
  2. Fit StandardScaler + LogisticRegression(C=1, lbfgs) — matches the shipped
     pickled pipeline
  3. Evaluate on every test dot-product file projected onto the same SV — gives
     the 16-cell cross-evaluation matrix
  4. Dump per-layer weights so we can see which layers carry each detector

Naming convention from the Drive folder:
   {train_data}_{train|test}_{real|fake}_dot_products_{sv_data}_steering_vectors_logreg.npy
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef,
    roc_auc_score, average_precision_score, roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATASETS = [
    "direct_prompt",
    "paraphrase_attacks_llm",
    "perturbation_attacks_llm",
    "prompt_attacks_llm",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-tsv-dir", default="interpret_out")
    return p.parse_args()


def load_train(data_dir, train_data):
    real = np.load(os.path.join(
        data_dir,
        f"{train_data}_train_real_dot_products_{train_data}_steering_vectors_logreg.npy",
    ))
    fake = np.load(os.path.join(
        data_dir,
        f"{train_data}_train_fake_dot_products_{train_data}_steering_vectors_logreg.npy",
    ))
    X = np.concatenate([real, fake])
    y = np.concatenate([np.zeros(len(real)), np.ones(len(fake))])
    return X, y


def load_test(data_dir, test_data, sv_data):
    """Return X, y for `test_data`'s test set, projected on `sv_data`'s steering vector."""
    real_path = os.path.join(
        data_dir,
        f"{test_data}_test_real_dot_products_{sv_data}_steering_vectors_logreg.npy",
    )
    fake_path = os.path.join(
        data_dir,
        f"{test_data}_test_fake_dot_products_{sv_data}_steering_vectors_logreg.npy",
    )
    if not os.path.exists(real_path):
        return None, None
    real = np.load(real_path)
    fake = np.load(fake_path)
    X = np.concatenate([real, fake])
    y = np.concatenate([np.zeros(len(real)), np.ones(len(fake))])
    return X, y


def tpr_at_fpr(y_true, y_score, fpr_target=0.05):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.interp(fpr_target, fpr, tpr))


def main():
    args = parse_args()
    os.makedirs(args.out_tsv_dir, exist_ok=True)

    metrics_rows = []
    weights_summary_rows = []

    for sv_data in DATASETS:
        sv_path = os.path.join(args.data_dir, f"{sv_data}_steering_vectors_logreg.npy")
        sv = np.load(sv_path)
        print(f"\n================ SV: {sv_data} (shape={sv.shape}) ================")

        # 1. Load train + fit detector. If train data missing (e.g. shipped data
        #    didn't include prompt_attacks_llm train), fall back to the shipped
        #    pickled pipeline — same StandardScaler+LogReg(C=1) config.
        train_real_path = os.path.join(
            args.data_dir,
            f"{sv_data}_train_real_dot_products_{sv_data}_steering_vectors_logreg.npy",
        )
        if os.path.exists(train_real_path):
            X_train, y_train = load_train(args.data_dir, sv_data)
            print(f"  train {X_train.shape}")
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("logreg", LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000,
                                              random_state=42)),
            ])
            pipe.fit(X_train, y_train)
            feat_std_pre = X_train.std(axis=0)
        else:
            # Find shipped pkl
            pkl_candidates = [
                f"{sv_data}_classifier_pipe_logreg.pkl",
                f"{sv_data}_logistic_regression_pipe_logreg.pkl",
            ]
            pkl_path = None
            for c in pkl_candidates:
                if os.path.exists(os.path.join(args.data_dir, c)):
                    pkl_path = os.path.join(args.data_dir, c); break
            if pkl_path is None:
                print(f"  [SKIP] no train data and no pkl for {sv_data}")
                continue
            print(f"  [no train data] using shipped pipeline {pkl_path}")
            with open(pkl_path, "rb") as fp:
                pipe = pickle.load(fp)
            feat_std_pre = np.full(32, np.nan, dtype=np.float32)  # unknown without train data
        coef = pipe["logreg"].coef_[0] if "logreg" in pipe.named_steps else pipe.steps[-1][1].coef_[0]
        intercept_arr = (
            pipe["logreg"].intercept_ if "logreg" in pipe.named_steps else pipe.steps[-1][1].intercept_
        )
        intercept = float(intercept_arr[0])
        print(f"  |coef|.sum={abs(coef).sum():.3f} intercept={intercept:.4f}")

        # 2. Cross-evaluate on every (test_data) projected onto this SV
        for test_data in DATASETS:
            X_test, y_test = load_test(args.data_dir, test_data, sv_data)
            if X_test is None:
                continue
            proba = pipe.predict_proba(X_test)[:, 1]
            pred = proba > 0.5
            row = {
                "sv_data": sv_data,
                "test_data": test_data,
                "auroc":              roc_auc_score(y_test, proba),
                "aupr":               average_precision_score(y_test, proba),
                "tpr_at_fpr_5pct":    tpr_at_fpr(y_test, proba, 0.05),
                "balanced_accuracy":  balanced_accuracy_score(y_test, pred),
                "mcc":                matthews_corrcoef(y_test, pred),
                "f1_macro":           f1_score(y_test, pred, average="macro"),
                "accuracy":           accuracy_score(y_test, pred),
                "n":                  len(y_test),
            }
            metrics_rows.append(row)

        # 3. Dump per-layer weights
        wdf = pd.DataFrame({
            "layer":          np.arange(len(coef)),
            "coef":           coef,
            "abs_coef":       np.abs(coef),
            "feat_std_pre":   feat_std_pre,        # std of dot products before standardization
        })
        out_path = os.path.join(args.out_tsv_dir, f"detectrl_weights_{sv_data}.tsv")
        wdf.to_csv(out_path, sep="\t", index=False)
        print(f"  wrote {out_path}")

        # Print summary of top-8 layers
        order = np.argsort(-np.abs(coef))
        print(f"  Top 8 layers by |coef|:")
        for i in order[:8]:
            sign = "fake-leaning" if coef[i] > 0 else "human-leaning"
            print(f"    layer {i:>2}: coef={coef[i]:+.4f}  ({sign})")
        for i in range(len(coef)):
            weights_summary_rows.append({
                "sv_data": sv_data, "layer": int(i),
                "coef": float(coef[i]),
            })

    # 4. Save & print metrics matrix
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = os.path.join(args.out_tsv_dir, "detectrl_cross_eval_metrics.tsv")
    metrics_df.to_csv(metrics_path, sep="\t", index=False)
    print(f"\nwrote {metrics_path}")

    print("\n=== Cross-eval AUROC (rows = SV, cols = test data) ===")
    pivot_auroc = metrics_df.pivot(index="sv_data", columns="test_data", values="auroc")
    print(pivot_auroc.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n=== Cross-eval F1 macro ===")
    pivot_f1 = metrics_df.pivot(index="sv_data", columns="test_data", values="f1_macro")
    print(pivot_f1.to_string(float_format=lambda x: f"{x:.4f}"))

    # Pivot of per-layer weights for at-a-glance comparison
    wdf_all = pd.DataFrame(weights_summary_rows)
    pivot_w = wdf_all.pivot(index="layer", columns="sv_data", values="coef")
    weights_pivot_path = os.path.join(args.out_tsv_dir, "detectrl_weights_pivot.tsv")
    pivot_w.to_csv(weights_pivot_path, sep="\t")
    print(f"\nwrote {weights_pivot_path}")
    print("\n=== Per-layer detector weights (rows = layer, cols = SV) ===")
    print(pivot_w.to_string(float_format=lambda x: f"{x:+.3f}"))


if __name__ == "__main__":
    main()
