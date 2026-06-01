"""For each (probe LM, dataset) pair where we have BOTH a dense steering vector
and the SAE-derived steering vector, compute per-layer cosine similarity between
the two directions. Reveals whether SAE-SV recovers the same direction as the
residual-stream-direct one or finds a different one.

GPT-Neo is the only LM here with both SAE and dense SVs, so we compare:
  - direct_prompt   (DetectRL)
  - generate / polish / rewrite (Mirage tasks; dense per-task SVs available)
"""

import argparse
import os

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sae-dir", required=True)
    p.add_argument("--dense-dir-detectrl", required=True)
    p.add_argument("--dense-dir-mirage", required=True)
    p.add_argument("--out-tsv", required=True)
    return p.parse_args()


def cosine_per_layer(a, b):
    """a, b shape (L, H). Return per-layer cosine similarity (L,)."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    out = np.zeros(a.shape[0])
    for layer in range(a.shape[0]):
        u = a[layer]; v = b[layer]
        out[layer] = (u @ v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)
    return out


def main():
    args = parse_args()
    rows = []

    # 1. direct_prompt — DetectRL
    sae = np.load(os.path.join(args.sae_dir, "direct_prompt_steering_vectors_logreg.npy"))
    dense = np.load(os.path.join(args.dense_dir_detectrl, "direct_prompt_steering_vectors_logreg.npy"))
    print(f"direct_prompt: SAE {sae.shape}  dense {dense.shape}")
    cos = cosine_per_layer(sae, dense)
    print(f"  per-layer cosine min={cos.min():.4f} max={cos.max():.4f} mean={cos.mean():.4f}")
    for layer, c in enumerate(cos):
        rows.append({"dataset": "DetectRL", "task": "direct_prompt",
                     "layer": layer, "cosine": float(c)})

    # 2. generate / polish / rewrite — Mirage per-task SVs
    for task in ["generate", "polish", "rewrite"]:
        sae = np.load(os.path.join(args.sae_dir, f"steering_vectors_logreg_{task}.npy"))
        dense = np.load(os.path.join(args.dense_dir_mirage, f"steering_vectors_logreg_l2_{task}.npy"))
        print(f"{task}: SAE {sae.shape}  dense {dense.shape}")
        cos = cosine_per_layer(sae, dense)
        print(f"  per-layer cosine min={cos.min():.4f} max={cos.max():.4f} mean={cos.mean():.4f}")
        for layer, c in enumerate(cos):
            rows.append({"dataset": "Mirage", "task": task,
                         "layer": layer, "cosine": float(c)})

    df = pd.DataFrame(rows)
    df.to_csv(args.out_tsv, sep="\t", index=False)
    print(f"\nwrote {args.out_tsv}")

    print("\n=== Summary table (mean cosine per task) ===")
    summary = df.groupby(["dataset", "task"])["cosine"].agg(["mean", "min", "max"])
    print(summary.to_string(float_format=lambda x: f"{x:+.4f}"))


if __name__ == "__main__":
    main()
