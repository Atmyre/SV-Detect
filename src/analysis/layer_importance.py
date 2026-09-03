"""Layer-importance correlation between the MIRAGE-trained detector and the
DetectRL-Multi-Domain-trained detector (Exp 3 reverse-transfer setup).

Both detectors are StandardScaler + LogisticRegression fit on cosine dot-products,
where features are indexed by (layer, direction). Following paper Appendix D.1,
the per-feature importance is c_ℓ = |w_ℓ| · σ_ℓ. With StandardScaler,
LogReg's coefficients are already applied to standardized features (σ_std=1),
so c_ℓ ≡ |w_std_ℓ| — the LogReg coefficient magnitude directly.

We then aggregate across the K directions per layer (paper's D.1 protocol:
"we sum c_ℓ across the K task-specific directions") to get a 32-dim
per-layer importance vector.

The question we answer: how correlated are these two vectors?
High correlation → shared representational mechanism across benchmarks.
"""

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from src.extract.nb_pipeline import build_detector


BASE = os.environ.get("SVDETECT_BASE", ".")
DOTS = f"{BASE}/data/dots"


def _stack(reals, fakes):
    real = np.concatenate(reals, axis=0)
    fake = np.concatenate(fakes, axis=0)
    X = np.concatenate([real, fake], axis=0)
    y = np.concatenate([np.zeros(len(real), np.int8), np.ones(len(fake), np.int8)])
    return X, y


def train_mirage_3vec(method: str = "logreg"):
    """Same recipe as all_experiments.exp_mirage_3vec."""
    d = f"{DOTS}/mirage_3vec"
    tasks = ["generate", "polish", "rewrite"]
    reals = [np.load(f"{d}/train_{t}_real_dot_products_{method}.npy") for t in tasks]
    fakes = [np.load(f"{d}/train_{t}_fake_dot_products_{method}.npy") for t in tasks]
    X, y = _stack(reals, fakes)
    return build_detector().fit(X, y), X.shape[1]


def train_md_4vec(method: str = "logreg"):
    """Same recipe as all_experiments.exp_reverse_transfer."""
    d = f"{DOTS}/reverse_md4vec"
    subs = ["arxiv", "writing_prompt", "xsum", "yelp_review"]
    reals = [np.load(f"{d}/{s}_train_real_dot_products_{method}.npy") for s in subs]
    fakes = [np.load(f"{d}/{s}_train_fake_dot_products_{method}.npy") for s in subs]
    X, y = _stack(reals, fakes)
    return build_detector().fit(X, y), X.shape[1]


def train_mirage_1vec(method: str = "logreg"):
    """Exp 5: single-SV MIRAGE (union of 800 pairs)."""
    d = f"{DOTS}/mirage_1vec"
    tasks = ["generate", "polish", "rewrite"]
    reals = [np.load(f"{d}/train_{t}_real_dot_products_{method}.npy") for t in tasks]
    fakes = [np.load(f"{d}/train_{t}_fake_dot_products_{method}.npy") for t in tasks]
    X, y = _stack(reals, fakes)
    return build_detector().fit(X, y), X.shape[1]


def train_md_persubset(sub: str, method: str = "logreg"):
    """Milestone-1: per-subset MD detector (K=1 direction, single SV)."""
    d = f"{DOTS}/detectrl_md"
    real = np.load(f"{d}/{sub}_train_real_dot_products_{sub}_steering_vectors_{method}.npy")
    fake = np.load(f"{d}/{sub}_train_fake_dot_products_{sub}_steering_vectors_{method}.npy")
    X, y = _stack([real], [fake])
    return build_detector().fit(X, y), X.shape[1]


def per_layer_importance(clf, n_features: int, n_layers: int = 32) -> np.ndarray:
    """Extract per-layer importance c_l from a Pipeline(StandardScaler, LogReg).

    Feature layout (K-fastest, matching nb_pipeline.dot_products):
      For 1-vec system:  features = 32 layers × 1 direction  = 32 (K=1)
      For 3-vec system:  features = 32 layers × 3 directions = 96 (K=3, K-fastest)
      For 4-vec system:  features = 32 layers × 4 directions = 128 (K=4, K-fastest)

    Reshape coefficient vector to (n_layers, K), take |.|, sum across K.
    """
    logreg = clf.named_steps["logreg"]
    w = logreg.coef_.ravel()
    assert w.shape[0] == n_features, (w.shape, n_features)
    K = n_features // n_layers
    assert n_features == n_layers * K, (n_features, n_layers, K)
    per_lk = np.abs(w).reshape(n_layers, K)
    return per_lk.sum(axis=1)


def corr_report(name_a: str, imp_a: np.ndarray, name_b: str, imp_b: np.ndarray) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p_r, p_p = pearsonr(imp_a, imp_b)
        s_r, s_p = spearmanr(imp_a, imp_b)
    top5_a = [int(x) for x in np.argsort(imp_a)[-5:][::-1]]
    top5_b = [int(x) for x in np.argsort(imp_b)[-5:][::-1]]
    top5_shared = [int(x) for x in set(top5_a) & set(top5_b)]
    return {
        "pair": f"{name_a} vs {name_b}",
        "pearson_r": float(p_r),
        "pearson_p": float(p_p),
        "spearman_r": float(s_r),
        "spearman_p": float(s_p),
        "top5_layers_a": top5_a,
        "top5_layers_b": top5_b,
        "top5_shared":   top5_shared,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=f"{BASE}/results/layer_importance")
    p.add_argument("--method", default="logreg", choices=["logreg", "mean", "pca"])
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n=== Training detectors and extracting per-layer importance (method={args.method}) ===")

    print("[1/6] MIRAGE-3vec ...")
    clf, n_feat = train_mirage_3vec(args.method)
    imp_mirage_3vec = per_layer_importance(clf, n_feat)
    print(f"  imp shape={imp_mirage_3vec.shape} range=[{imp_mirage_3vec.min():.3f}, {imp_mirage_3vec.max():.3f}]")

    print("[2/6] MIRAGE-1vec (union 800) ...")
    clf, n_feat = train_mirage_1vec(args.method)
    imp_mirage_1vec = per_layer_importance(clf, n_feat)
    print(f"  imp shape={imp_mirage_1vec.shape}")

    print("[3/6] DetectRL-MD-4vec (reverse-transfer detector) ...")
    clf, n_feat = train_md_4vec(args.method)
    imp_md_4vec = per_layer_importance(clf, n_feat)
    print(f"  imp shape={imp_md_4vec.shape}")

    print("[4-7/6] DetectRL-MD per-subset detectors ...")
    md_persubset = {}
    for sub in ["arxiv", "writing_prompt", "xsum", "yelp_review"]:
        clf, n_feat = train_md_persubset(sub, args.method)
        md_persubset[sub] = per_layer_importance(clf, n_feat)

    all_importances = {
        "MIRAGE-3vec": imp_mirage_3vec,
        "MIRAGE-1vec (union 800)": imp_mirage_1vec,
        "MD-4vec (reverse transfer)": imp_md_4vec,
        **{f"MD-{s}": v for s, v in md_persubset.items()},
    }

    # === Correlation matrix ===
    names = list(all_importances.keys())
    n = len(names)
    pearson_mat = np.zeros((n, n))
    spearman_mat = np.zeros((n, n))
    for i, na in enumerate(names):
        for j, nb in enumerate(names):
            if i == j:
                pearson_mat[i, j] = 1.0; spearman_mat[i, j] = 1.0
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pearson_mat[i, j], _ = pearsonr(all_importances[na], all_importances[nb])
                    spearman_mat[i, j], _ = spearmanr(all_importances[na], all_importances[nb])

    dfp = pd.DataFrame(pearson_mat, index=names, columns=names)
    dfs = pd.DataFrame(spearman_mat, index=names, columns=names)
    print("\nPearson correlation matrix (per-layer importance):")
    print(dfp.to_string(float_format=lambda x: f"{x:+.3f}"))
    print("\nSpearman correlation matrix (per-layer importance):")
    print(dfs.to_string(float_format=lambda x: f"{x:+.3f}"))

    # === Focused pairs ===
    print("\n=== Detailed reports for headline pairs ===")
    key_pairs = [
        ("MIRAGE-3vec", "MD-4vec (reverse transfer)"),
        ("MIRAGE-3vec", "MIRAGE-1vec (union 800)"),
        ("MIRAGE-3vec", "MD-arxiv"),
        ("MIRAGE-3vec", "MD-xsum"),
        ("MD-4vec (reverse transfer)", "MD-arxiv"),
    ]
    reports = []
    for a, b in key_pairs:
        r = corr_report(a, all_importances[a], b, all_importances[b])
        reports.append(r)
        print(f"\n{r['pair']}")
        print(f"  Pearson  r={r['pearson_r']:+.4f}  p={r['pearson_p']:.4e}")
        print(f"  Spearman r={r['spearman_r']:+.4f}  p={r['spearman_p']:.4e}")
        print(f"  Top-5 layers A: {r['top5_layers_a']}")
        print(f"  Top-5 layers B: {r['top5_layers_b']}")
        print(f"  Shared in top-5: {r['top5_shared']}  ({len(r['top5_shared'])} of 5)")

    # === Dump per-layer importance vectors (for plotting) ===
    per_layer_df = pd.DataFrame(all_importances)
    per_layer_df.index.name = "layer"
    per_layer_df.to_csv(f"{args.out_dir}/per_layer_importance_{args.method}.csv")
    dfp.to_csv(f"{args.out_dir}/correlation_pearson_{args.method}.csv")
    dfs.to_csv(f"{args.out_dir}/correlation_spearman_{args.method}.csv")
    with open(f"{args.out_dir}/headline_pairs_{args.method}.json", "w") as f:
        json.dump(reports, f, indent=2)

    print(f"\nwrote: per_layer_importance / correlation_{{pearson,spearman}} / headline_pairs — in {args.out_dir}")


if __name__ == "__main__":
    main()
