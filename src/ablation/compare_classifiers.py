"""Run a small zoo of standard classifiers on the same cosine features used by
the logreg detector. Lets us see whether non-linear models can extract more
signal from the per-layer (or per-(layer,direction)) cosines.

Three modes (chosen by --dataset):

  coling   — load real/fake _train, _val, _test dot products for one
             (method, suffix) on a given LM. Sweep the C-equivalent for each
             classifier and report best-by-val test metrics.

  detectrl — load DetectRL train/test for one attack on its own SV.

  mirage   — load Mirage train + DIG/SIG × {generate,polish,rewrite} test
             for one method.

Saves a per-classifier metric TSV.
"""

import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, f1_score,
                             matthews_corrcoef, roc_auc_score, roc_curve)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["coling", "detectrl", "mirage"])
    p.add_argument("--data-dir", required=True)
    p.add_argument("--method", default="logreg",
                   help="SV method tag, e.g. mean / pca / logreg / logreg_l2 / logreg_None")
    p.add_argument("--suffix", default="",
                   help="COLING-only filter suffix (e.g. _woweak)")
    p.add_argument("--test-jsonl", default=None,
                   help="COLING-only: labeled jsonl for test labels")
    p.add_argument("--detectrl-attack", default=None,
                   help="DetectRL-only: which attack type's SV+train+test to use")
    p.add_argument("--out-tsv", required=True)
    p.add_argument("--max-train", type=int, default=200000,
                   help="Subsample train if larger; keeps KNN/MLP tractable.")
    p.add_argument("--skip-slow", action="store_true",
                   help="Skip KNN and MLP if train is large.")
    return p.parse_args()


# ============================================================
# Data loading
# ============================================================

def load_coling(data_dir, method, suffix, test_jsonl):
    file_tag = f"{method}{suffix}"
    def _load(split):
        real = np.load(os.path.join(data_dir, f"real_{split}_dot_products_{file_tag}.npy"))
        fake = np.load(os.path.join(data_dir, f"fake_{split}_dot_products_{file_tag}.npy"))
        X = np.concatenate([real, fake])
        y = np.concatenate([np.zeros(len(real)), np.ones(len(fake))])
        return X, y
    X_train, y_train = _load("train")
    X_val,   y_val   = _load("val")
    X_test = np.load(os.path.join(data_dir, f"test_dot_products_{file_tag}.npy"))
    y_test = np.array(pd.read_json(test_jsonl, lines=True)["label"])
    if len(y_test) != len(X_test):
        raise ValueError(f"test mismatch: X={len(X_test)} y={len(y_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def load_detectrl(data_dir, attack):
    sv = attack
    real_tr = np.load(os.path.join(data_dir, f"{sv}_train_real_dot_products_{sv}_steering_vectors_logreg.npy"))
    fake_tr = np.load(os.path.join(data_dir, f"{sv}_train_fake_dot_products_{sv}_steering_vectors_logreg.npy"))
    X_train = np.concatenate([real_tr, fake_tr])
    y_train = np.concatenate([np.zeros(len(real_tr)), np.ones(len(fake_tr))])
    real_te = np.load(os.path.join(data_dir, f"{sv}_test_real_dot_products_{sv}_steering_vectors_logreg.npy"))
    fake_te = np.load(os.path.join(data_dir, f"{sv}_test_fake_dot_products_{sv}_steering_vectors_logreg.npy"))
    X_test = np.concatenate([real_te, fake_te])
    y_test = np.concatenate([np.zeros(len(real_te)), np.ones(len(fake_te))])
    # Use 20% of train as val
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y_train))
    cut = int(0.8 * len(y_train))
    return (X_train[idx[:cut]], y_train[idx[:cut]],
            X_train[idx[cut:]], y_train[idx[cut:]],
            X_test, y_test)


def load_mirage(data_dir, method):
    real_tr = np.load(os.path.join(data_dir, f"train_real_dot_products_{method}.npy"))
    fake_tr = np.load(os.path.join(data_dir, f"train_fake_dot_products_{method}.npy"))
    X_train = np.concatenate([real_tr, fake_tr])
    y_train = np.concatenate([np.zeros(len(real_tr)), np.ones(len(fake_tr))])
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y_train))
    cut = int(0.8 * len(y_train))
    Xtr, ytr = X_train[idx[:cut]], y_train[idx[:cut]]
    Xva, yva = X_train[idx[cut:]], y_train[idx[cut:]]
    # Aggregate all 6 test cells into a single test set
    test_xs, test_ys = [], []
    for source in ["DIG", "SIG"]:
        for task in ["generate", "polish", "rewrite"]:
            real = np.load(os.path.join(
                data_dir, f"{source}_{task}_test_real_dot_products_{method}.npy"))
            fake = np.load(os.path.join(
                data_dir, f"{source}_{task}_test_fake_dot_products_{method}.npy"))
            test_xs.append(np.concatenate([real, fake]))
            test_ys.append(np.concatenate([np.zeros(len(real)), np.ones(len(fake))]))
    X_test = np.concatenate(test_xs)
    y_test = np.concatenate(test_ys)
    return Xtr, ytr, Xva, yva, X_test, y_test


# ============================================================
# Classifier zoo
# ============================================================

def build_zoo(skip_slow: bool):
    z = {}

    # 1. logreg with light C sweep (we already know this; baseline)
    z["logreg_l2_C=1"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)),
    ])
    z["logreg_l1_C=1"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, solver="saga", penalty="l1",
                                    max_iter=1000, random_state=42)),
    ])

    # 2. tree-based
    z["decision_tree_d=8"] = DecisionTreeClassifier(max_depth=8, random_state=42)
    z["decision_tree_dNone"] = DecisionTreeClassifier(random_state=42)
    z["random_forest_n=100"] = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    z["hist_gbm"] = HistGradientBoostingClassifier(max_iter=200, random_state=42)

    # 3. linear SVM (no probability output natively; use decision_function)
    z["linear_svc"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearSVC(C=1.0, max_iter=2000, random_state=42)),
    ])

    # 4. naive bayes
    z["gaussian_nb"] = GaussianNB()

    # 5. KNN — slow on big train
    if not skip_slow:
        z["knn_k=5"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
        ])

    # 6. MLP (small)
    if not skip_slow:
        z["mlp_64x32"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200,
                                  random_state=42, early_stopping=True)),
        ])

    return z


def predict_proba(model, X):
    """Return (proba_class1, pred). LinearSVC has no predict_proba — we use
    decision_function and a 0-threshold for prediction; AUROC uses raw scores."""
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)[:, 1]
    else:
        # LinearSVC inside Pipeline
        try:
            p = model.decision_function(X)
        except AttributeError:
            try:
                p = model.named_steps["clf"].decision_function(model.named_steps["scaler"].transform(X))
            except (AttributeError, KeyError):
                p = model.predict(X).astype(float)
    pred = (p > 0.5).astype(int) if (p.min() >= 0 and p.max() <= 1) else (p > 0).astype(int)
    return p, pred


def evaluate(name, model, X_test, y_test) -> dict:
    p, pred = predict_proba(model, X_test)
    fpr, tpr, _ = roc_curve(y_test, p)
    return {
        "classifier":        name,
        "accuracy":          accuracy_score(y_test, pred),
        "f1_macro":          f1_score(y_test, pred, average="macro"),
        "auroc":             roc_auc_score(y_test, p),
        "aupr":              average_precision_score(y_test, p),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "mcc":               matthews_corrcoef(y_test, pred),
        "tpr_at_fpr_5pct":   float(np.interp(0.05, fpr, tpr)),
    }


def main():
    args = parse_args()

    if args.dataset == "coling":
        if not args.test_jsonl:
            raise SystemExit("--test-jsonl required for coling")
        X_train, y_train, X_val, y_val, X_test, y_test = load_coling(
            args.data_dir, args.method, args.suffix, args.test_jsonl,
        )
    elif args.dataset == "detectrl":
        if not args.detectrl_attack:
            raise SystemExit("--detectrl-attack required")
        X_train, y_train, X_val, y_val, X_test, y_test = load_detectrl(
            args.data_dir, args.detectrl_attack,
        )
    elif args.dataset == "mirage":
        X_train, y_train, X_val, y_val, X_test, y_test = load_mirage(
            args.data_dir, args.method,
        )

    # Subsample train if requested
    if len(y_train) > args.max_train:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(y_train), args.max_train, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]
        print(f"  subsampled train to {len(y_train)}")

    print(f"shapes: train {X_train.shape}, val {X_val.shape}, test {X_test.shape}")
    print(f"label balance: train {np.bincount(y_train.astype(int))}, "
          f"test {np.bincount(y_test.astype(int))}")

    skip_slow = args.skip_slow or len(y_train) > 50000
    zoo = build_zoo(skip_slow=skip_slow)
    print(f"zoo: {list(zoo.keys())}")

    rows = []
    for name, model in zoo.items():
        t0 = time.time()
        try:
            model.fit(X_train, y_train)
            metrics_val  = evaluate(name, model, X_val,  y_val)
            metrics_test = evaluate(name, model, X_test, y_test)
            elapsed = time.time() - t0
            print(f"\n=== {name}  ({elapsed:.1f}s) ===")
            print(f"  val  acc={metrics_val['accuracy']:.4f} f1={metrics_val['f1_macro']:.4f} "
                  f"auc={metrics_val['auroc']:.4f}")
            print(f"  test acc={metrics_test['accuracy']:.4f} f1={metrics_test['f1_macro']:.4f} "
                  f"auc={metrics_test['auroc']:.4f}")
            rows.append({
                "classifier": name, "elapsed_sec": round(elapsed, 1),
                **{f"val_{k}": v for k, v in metrics_val.items() if k != "classifier"},
                **{f"test_{k}": v for k, v in metrics_test.items() if k != "classifier"},
            })
        except Exception as e:
            elapsed = time.time() - t0
            print(f"\n=== {name}  ({elapsed:.1f}s)  FAILED: {e}")
            rows.append({"classifier": name, "elapsed_sec": round(elapsed, 1), "error": str(e)})

    df = pd.DataFrame(rows)
    df.to_csv(args.out_tsv, sep="\t", index=False)
    print(f"\nwrote {args.out_tsv}")

    # Print summary sorted by test F1
    if "test_f1_macro" in df.columns:
        print("\n=== ranked by test F1 ===")
        cols = ["classifier", "test_accuracy", "test_f1_macro", "test_auroc",
                "test_mcc", "test_tpr_at_fpr_5pct", "elapsed_sec"]
        cols = [c for c in cols if c in df.columns]
        print(df.sort_values("test_f1_macro", ascending=False)[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
