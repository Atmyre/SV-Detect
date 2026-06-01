"""Train a logistic regression on per-layer dot products with mean steering vectors,
and evaluate on val (COLING dev) and test (COLING held-out test_set_en_with_label).

Reads the dot-product files produced by compute_steering_vectors.py.
Test labels are read from the same jsonl that extract_activations.py used (the
test split is read in original file order, so labels align by row index).
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True,
                   help="Folder with *_dot_products_<method><suffix>.npy")
    p.add_argument("--method", default="mean", choices=["mean", "pca", "logreg"],
                   help="Which steering-vector method's dot products to load.")
    p.add_argument("--suffix", default="",
                   help="Same value as compute_steering_vectors.py --out-suffix "
                        "(e.g. '_woweak'). Default: empty.")
    p.add_argument("--test-jsonl", required=True,
                   help="Same labeled jsonl that was passed to extract_activations.py "
                        "as --test-jsonl (default: test_set_en_with_label.jsonl)")
    p.add_argument("--cs", nargs="+", type=float,
                   default=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
    p.add_argument("--penalty", default="l1", choices=["l1", "l2"])
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--layers", nargs="+", type=int, default=None,
                   help="Subset of layer indices to feed the detector. Default: all.")
    return p.parse_args()


def load_xy(data_dir: str, split: str, file_tag: str) -> tuple[np.ndarray, np.ndarray]:
    real = np.load(os.path.join(data_dir, f"real_{split}_dot_products_{file_tag}.npy"))
    fake = np.load(os.path.join(data_dir, f"fake_{split}_dot_products_{file_tag}.npy"))
    X = np.concatenate([real, fake], axis=0)
    y = np.concatenate([np.zeros(len(real)), np.ones(len(fake))], axis=0)
    return X, y


def load_test(data_dir: str, test_jsonl: str, file_tag: str) -> tuple[np.ndarray, np.ndarray]:
    X = np.load(os.path.join(data_dir, f"test_dot_products_{file_tag}.npy"))
    y = np.array(pd.read_json(test_jsonl, lines=True)["label"])
    if len(y) != len(X):
        raise ValueError(f"test X has {len(X)} rows but labels have {len(y)}")
    return X, y


def evaluate(model, X, y, threshold: float) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = proba > threshold
    return {
        "accuracy": accuracy_score(y, pred),
        "f1_macro": f1_score(y, pred, average="macro"),
        "roc_auc": roc_auc_score(y, proba),
    }


def main():
    args = parse_args()

    file_tag = f"{args.method}{args.suffix}"
    print(f"file_tag={file_tag}")
    X_train, y_train = load_xy(args.data_dir, "train", file_tag)
    X_val,   y_val   = load_xy(args.data_dir, "val",   file_tag)
    X_test,  y_test  = load_test(args.data_dir, args.test_jsonl, file_tag)
    if args.layers is not None:
        layer_idx = sorted(set(args.layers))
        print(f"restricting to layers {layer_idx} ({len(layer_idx)} of {X_train.shape[1]})")
        X_train = X_train[:, layer_idx]
        X_val   = X_val[:,   layer_idx]
        X_test  = X_test[:,  layer_idx]
    print(f"train {X_train.shape} val {X_val.shape} test {X_test.shape}")

    solver = "saga" if args.penalty == "l1" else "lbfgs"
    print(f"Sweeping C over {args.cs} with penalty={args.penalty}")
    for C in args.cs:
        model = LogisticRegression(
            solver=solver, penalty=args.penalty, C=C,
            max_iter=1000, random_state=42,
        )
        model.fit(X_train, y_train)
        val_metrics  = evaluate(model, X_val,  y_val,  args.threshold)
        test_metrics = evaluate(model, X_test, y_test, args.threshold)
        print(f"C={C:>7g}  "
              f"val acc={val_metrics['accuracy']:.4f} f1={val_metrics['f1_macro']:.4f} auc={val_metrics['roc_auc']:.4f}   "
              f"test acc={test_metrics['accuracy']:.4f} f1={test_metrics['f1_macro']:.4f} auc={test_metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
