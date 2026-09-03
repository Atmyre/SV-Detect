"""Bulk-produce every dot-product file we need for the ARR-rebuttal experiments,
with per-activation-set caching so each ~1-2 GB activation chunk set is loaded
at most once (across all SV systems it gets projected onto).

Produces four dot-product families:

  detectrl_md/           : DetectRL MD 16-cell matrix (per-subset SVs)
  mirage_3vec/           : MIRAGE test + train projected onto 3-vec QR system
                           (reproduces paper's main MIRAGE result)
  reverse_md4vec/        : MD train + MIRAGE test projected onto MD-4vec QR system
                           (Exp 3 reverse transfer)
  mirage_1vec/           : MIRAGE train union + MIRAGE test projected onto
                           single-vector MIRAGE SV (Exp 5)

Also produces the two derived SV systems:
  detectrl_md_4vec/orthonormal_steering_vectors_{method}.npy
  mirage_1vec/all_steering_vectors_{method}.npy
"""

import argparse
import glob
import os
import re

import numpy as np

from src.extract.nb_pipeline import (compute_sv, dot_products,
                                     load_all_chunks, qr_orthonormalize)


BASE = os.environ.get("SVDETECT_BASE", ".")
ACTS_MD  = f"{BASE}/data/activations/detectrl/multi_domains"
ACTS_MIR_TR = f"{BASE}/data/activations/mirage/train"
ACTS_MIR_TE = f"{BASE}/data/activations/mirage/test"
SVS_MD   = f"{BASE}/data/svs/detectrl/multi_domains"
SVS_MIR  = f"{BASE}/data/svs/mirage"
SVS_MD4  = f"{BASE}/data/svs/detectrl_md_4vec"
SVS_MIR1 = f"{BASE}/data/svs/mirage_1vec"
DOTS     = f"{BASE}/data/dots"

MD_SUBS = ["arxiv", "writing_prompt", "xsum", "yelp_review"]
MIR_TASKS = ["generate", "polish", "rewrite"]
MIR_SCEN  = ["DIG", "SIG"]
METHODS = ["mean", "logreg", "pca"]


def build_derived_svs():
    os.makedirs(SVS_MD4, exist_ok=True)
    for m in METHODS:
        svs = [np.load(f"{SVS_MD}/{s}_steering_vectors_{m}.npy") for s in MD_SUBS]
        Q = qr_orthonormalize(svs)
        np.save(f"{SVS_MD4}/orthonormal_steering_vectors_{m}.npy", Q)
        print(f"  saved MD-4vec {m}  shape={Q.shape}")

    # MIRAGE 1-vec built from union of 3 tasks' activations
    def combine(prefix_kind):
        arrays = []
        for t in MIR_TASKS:
            files = sorted(
                glob.glob(f"{ACTS_MIR_TR}/train_{t}_{prefix_kind}_activations_*.npy"),
                key=lambda x: int(re.search(r"_(\d+)-\d+\.npy$", x).group(1)),
            )
            arrays.extend(np.load(f) for f in files)
        return np.concatenate(arrays, axis=0) if arrays else None

    real = combine("real"); fake = combine("fake")
    print("MIRAGE union train shapes:", real.shape, fake.shape)
    os.makedirs(SVS_MIR1, exist_ok=True)
    for m in METHODS:
        sv = compute_sv(fake, real, m)
        np.save(f"{SVS_MIR1}/all_steering_vectors_{m}.npy", sv)
        print(f"  saved MIRAGE-1vec {m}  shape={sv.shape}")


def _dot(activations, sv):
    return dot_products(activations, sv)


def _write(out_path, arr):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, arr)


def project_and_save(activations, sv_paths_and_out_paths, skip_existing=True):
    """Given loaded activations and a list of (sv_path, out_path), do projections."""
    for sv_path, out_path in sv_paths_and_out_paths:
        if skip_existing and os.path.exists(out_path):
            continue
        sv = np.load(sv_path)
        d = _dot(activations, sv)
        _write(out_path, d)
        print(f"    saved {os.path.basename(out_path)}  {d.shape}")


def detectrl_md_dots():
    """For MD: per-subset SVs. For each activation set, project onto every subset's SV."""
    md_dir = f"{DOTS}/detectrl_md"
    os.makedirs(md_dir, exist_ok=True)
    for eval_sub in MD_SUBS:
        for split in ("train", "test"):
            for kind in ("real", "fake"):
                prefix = f"{eval_sub}_{split}_{kind}"
                acts_dir = f"{ACTS_MD}/{eval_sub}"
                # only load if any target is missing
                targets = []
                for method in METHODS:
                    for sv_sub in MD_SUBS:
                        # For train split, only own-SV is needed (we don't cross-train)
                        if split == "train" and sv_sub != eval_sub:
                            continue
                        sv_path = f"{SVS_MD}/{sv_sub}_steering_vectors_{method}.npy"
                        out = f"{md_dir}/{eval_sub}_{split}_{kind}_dot_products_{sv_sub}_steering_vectors_{method}.npy"
                        if not os.path.exists(out):
                            targets.append((sv_path, out))
                if not targets:
                    continue
                print(f"[MD] loading acts {prefix}")
                acts = load_all_chunks(acts_dir, prefix)
                print(f"    {acts.shape}  -> {len(targets)} projections")
                project_and_save(acts, targets)


def mirage_3vec_dots():
    """MIRAGE 3-vec orthonormal system: project MIRAGE train + test onto it."""
    out_dir = f"{DOTS}/mirage_3vec"
    os.makedirs(out_dir, exist_ok=True)
    # MIRAGE train: 3 tasks × 2 kinds × 3 methods
    for task in MIR_TASKS:
        for kind in ("real", "fake"):
            prefix = f"train_{task}_{kind}"
            targets = []
            for method in METHODS:
                sv_path = f"{SVS_MIR}/orthonormal_steering_vectors_{method}.npy"
                out = f"{out_dir}/train_{task}_{kind}_dot_products_{method}.npy"
                if not os.path.exists(out):
                    targets.append((sv_path, out))
            if not targets:
                continue
            print(f"[MIR-3vec] loading acts {prefix}")
            acts = load_all_chunks(ACTS_MIR_TR, prefix)
            print(f"    {acts.shape}  -> {len(targets)} projections")
            project_and_save(acts, targets)
    # MIRAGE test: 6 sets
    for scen in MIR_SCEN:
        for task in MIR_TASKS:
            for kind in ("real", "fake"):
                prefix = f"{scen}_{task}_{kind}"
                acts_dir = f"{ACTS_MIR_TE}/{scen}/{task}"
                # Skip if this test set hasn't been extracted yet
                if not os.path.exists(acts_dir) or not glob.glob(f"{acts_dir}/{prefix}_activations_*.npy"):
                    print(f"[MIR-3vec] skip missing {prefix}")
                    continue
                targets = []
                for method in METHODS:
                    sv_path = f"{SVS_MIR}/orthonormal_steering_vectors_{method}.npy"
                    out = f"{out_dir}/{scen}_{task}_test_{kind}_dot_products_{method}.npy"
                    if not os.path.exists(out):
                        targets.append((sv_path, out))
                if not targets:
                    continue
                print(f"[MIR-3vec] loading acts {prefix}")
                acts = load_all_chunks(acts_dir, prefix)
                print(f"    {acts.shape}  -> {len(targets)} projections")
                project_and_save(acts, targets)


def reverse_md4vec_dots():
    """Reverse transfer: DetectRL MD train + MIRAGE test → MD-4vec orthonormal system."""
    out_dir = f"{DOTS}/reverse_md4vec"
    os.makedirs(out_dir, exist_ok=True)
    # MD train: 4 subsets × 2 kinds × 3 methods
    for sub in MD_SUBS:
        for kind in ("real", "fake"):
            prefix = f"{sub}_train_{kind}"
            acts_dir = f"{ACTS_MD}/{sub}"
            targets = []
            for method in METHODS:
                sv_path = f"{SVS_MD4}/orthonormal_steering_vectors_{method}.npy"
                out = f"{out_dir}/{sub}_train_{kind}_dot_products_{method}.npy"
                if not os.path.exists(out):
                    targets.append((sv_path, out))
            if not targets:
                continue
            print(f"[REV-4vec] loading acts {prefix}")
            acts = load_all_chunks(acts_dir, prefix)
            print(f"    {acts.shape}  -> {len(targets)} projections")
            project_and_save(acts, targets)
    # MIRAGE test: 6 sets → project onto MD-4vec
    for scen in MIR_SCEN:
        for task in MIR_TASKS:
            for kind in ("real", "fake"):
                prefix = f"{scen}_{task}_{kind}"
                acts_dir = f"{ACTS_MIR_TE}/{scen}/{task}"
                if not os.path.exists(acts_dir) or not glob.glob(f"{acts_dir}/{prefix}_activations_*.npy"):
                    print(f"[REV-4vec] skip missing {prefix}")
                    continue
                targets = []
                for method in METHODS:
                    sv_path = f"{SVS_MD4}/orthonormal_steering_vectors_{method}.npy"
                    out = f"{out_dir}/{scen}_{task}_test_{kind}_dot_products_{method}.npy"
                    if not os.path.exists(out):
                        targets.append((sv_path, out))
                if not targets:
                    continue
                print(f"[REV-4vec] loading acts {prefix}")
                acts = load_all_chunks(acts_dir, prefix)
                print(f"    {acts.shape}  -> {len(targets)} projections")
                project_and_save(acts, targets)


def mirage_1vec_dots():
    """Exp 5: MIRAGE train + test → single-vector MIRAGE SV."""
    out_dir = f"{DOTS}/mirage_1vec"
    os.makedirs(out_dir, exist_ok=True)
    for task in MIR_TASKS:
        for kind in ("real", "fake"):
            prefix = f"train_{task}_{kind}"
            targets = []
            for method in METHODS:
                sv_path = f"{SVS_MIR1}/all_steering_vectors_{method}.npy"
                out = f"{out_dir}/train_{task}_{kind}_dot_products_{method}.npy"
                if not os.path.exists(out):
                    targets.append((sv_path, out))
            if not targets:
                continue
            print(f"[MIR-1vec] loading acts {prefix}")
            acts = load_all_chunks(ACTS_MIR_TR, prefix)
            print(f"    {acts.shape}  -> {len(targets)} projections")
            project_and_save(acts, targets)
    for scen in MIR_SCEN:
        for task in MIR_TASKS:
            for kind in ("real", "fake"):
                prefix = f"{scen}_{task}_{kind}"
                acts_dir = f"{ACTS_MIR_TE}/{scen}/{task}"
                if not os.path.exists(acts_dir) or not glob.glob(f"{acts_dir}/{prefix}_activations_*.npy"):
                    print(f"[MIR-1vec] skip missing {prefix}")
                    continue
                targets = []
                for method in METHODS:
                    sv_path = f"{SVS_MIR1}/all_steering_vectors_{method}.npy"
                    out = f"{out_dir}/{scen}_{task}_test_{kind}_dot_products_{method}.npy"
                    if not os.path.exists(out):
                        targets.append((sv_path, out))
                if not targets:
                    continue
                print(f"[MIR-1vec] loading acts {prefix}")
                acts = load_all_chunks(acts_dir, prefix)
                print(f"    {acts.shape}  -> {len(targets)} projections")
                project_and_save(acts, targets)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stages", nargs="+",
                   default=["derived_svs", "md", "mir_3vec", "reverse", "mir_1vec"])
    args = p.parse_args()

    stages = set(args.stages)
    if "derived_svs" in stages:
        print("\n=== Building derived SVs (MD-4vec + MIRAGE-1vec) ===")
        build_derived_svs()
    if "md" in stages:
        print("\n=== DetectRL MD 16-cell dots ===")
        detectrl_md_dots()
    if "mir_3vec" in stages:
        print("\n=== MIRAGE 3-vec dots ===")
        mirage_3vec_dots()
    if "reverse" in stages:
        print("\n=== Reverse transfer (MD-4vec) dots ===")
        reverse_md4vec_dots()
    if "mir_1vec" in stages:
        print("\n=== MIRAGE 1-vec dots (Exp 5) ===")
        mirage_1vec_dots()
    print("\nDONE")


if __name__ == "__main__":
    main()
