"""Bulk-compute cosine dot-product files needed for the DetectRL Multi-Domain
same-source + cross-source evaluation (16-cell matrix), plus optional MIRAGE
test-side projection using the same MD SVs (for the reverse-transfer experiment).

Given a set of subsets and their SVs, and a set of "eval" datasets (activations
already extracted), this script emits every {test}_{split}_{kind}_dot_products_{train}_steering_vectors_{method}.npy
in one go.

Design: this is a pure numpy job — small CPU cost, no GPU needed for dot products.
"""

import argparse
import os
from glob import glob

import numpy as np

from src.extract.nb_pipeline import dot_products, load_all_chunks


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--acts-dir", required=True,
                   help="Root activations dir. Layout: {acts-dir}/{subset}/{subset}_{split}_{kind}_activations_*.npy")
    p.add_argument("--sv-dir", required=True,
                   help="Dir with {subset}_steering_vectors_{method}.npy and/or "
                        "orthonormal_steering_vectors_{method}.npy")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--train-subsets", nargs="+", required=True,
                   help="Subsets whose SVs to use (source of the direction)")
    p.add_argument("--eval-subsets", nargs="+", required=True,
                   help="Subsets whose activations to project. Layout: acts_dir/{eval_subset}/...")
    p.add_argument("--methods", nargs="+", default=["logreg"])
    p.add_argument("--kinds", nargs="+", default=["real", "fake"])
    p.add_argument("--splits", nargs="+", default=["train", "test"])
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--orthonormal", action="store_true",
                   help="Use orthonormal_steering_vectors_{method}.npy instead of per-subset")
    p.add_argument("--label", default="",
                   help="Optional infix. Used mainly for orthonormal to avoid collisions.")
    return p.parse_args()


def sv_path_for(sv_dir: str, train_subset: str, method: str, orthonormal: bool) -> str:
    if orthonormal:
        return os.path.join(sv_dir, f"orthonormal_steering_vectors_{method}.npy")
    return os.path.join(sv_dir, f"{train_subset}_steering_vectors_{method}.npy")


def out_path(out_dir: str, eval_subset: str, split: str, kind: str,
             train_subset: str, method: str) -> str:
    return os.path.join(
        out_dir,
        f"{eval_subset}_{split}_{kind}_dot_products_{train_subset}_steering_vectors_{method}.npy",
    )


def out_path_ortho(out_dir: str, eval_subset: str, split: str, kind: str,
                   method: str, label: str) -> str:
    tag = f"{label}_" if label else ""
    return os.path.join(
        out_dir,
        f"{eval_subset}_{split}_{kind}_dot_products_{tag}orthonormal_{method}.npy",
    )


def main():
    a = parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    # Cache activations we've already loaded so we don't re-read chunks per SV.
    cache: dict[str, np.ndarray] = {}

    def acts_of(subset: str, split: str, kind: str) -> np.ndarray:
        key = f"{subset}/{split}/{kind}"
        if key in cache:
            return cache[key]
        acts_dir = os.path.join(a.acts_dir, subset)
        prefix = f"{subset}_{split}_{kind}"
        arr = load_all_chunks(acts_dir, prefix)
        cache[key] = arr
        print(f"  loaded {key}  shape={arr.shape}")
        return arr

    for method in a.methods:
        for train_sub in a.train_subsets:
            sv = np.load(sv_path_for(a.sv_dir, train_sub, method, a.orthonormal))
            print(f"[SV] {train_sub} method={method}  shape={sv.shape}")

            for eval_sub in a.eval_subsets:
                for split in a.splits:
                    # For same-source train dots we only need eval_sub == train_sub
                    if split == "train" and eval_sub != train_sub and not a.orthonormal:
                        continue
                    for kind in a.kinds:
                        if a.orthonormal:
                            out = out_path_ortho(a.out_dir, eval_sub, split, kind, method, a.label)
                        else:
                            out = out_path(a.out_dir, eval_sub, split, kind, train_sub, method)
                        if a.skip_existing and os.path.exists(out):
                            continue
                        acts = acts_of(eval_sub, split, kind)
                        d = dot_products(acts, sv)
                        np.save(out, d)
                        print(f"  saved {os.path.basename(out)} shape={d.shape}")

            # If orthonormal, we compute once (single SV), no need to iterate train_subs
            if a.orthonormal:
                break


if __name__ == "__main__":
    main()
