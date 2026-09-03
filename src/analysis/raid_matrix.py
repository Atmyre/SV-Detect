"""RAID-faithful cross-source matrix for SV-Detect. Same protocol as the
paper's DetectRL Multi-Domain 16-cell but for RAID's 24 configs.

For each (train_cfg, eval_cfg) pair:
  1. Fit a per-config SV on train_cfg train activations (mean-diff / logreg / pca)
  2. Compute train dot products (only for own-SV), fit StandardScaler+LogReg
  3. Project eval_cfg test activations onto train_cfg SV, score with classifier
  4. Report AUROC + F1

Reports:
  - 24 × 24 AUROC matrix (rows = train_cfg, cols = eval_cfg)
  - In-domain diagonal mean (matches RepreGuard's 96.34 ID)
  - Off-diagonal mean (matches RepreGuard's 93.49 OOD)
  - Per-LLM-type in-domain and cross-source means
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

from src.analysis.padben_eval import tpr_at_fpr
from src.extract.nb_pipeline import (build_detector, compute_sv, dot_products,
                                     load_all_chunks)


BASE = os.environ.get("SVDETECT_BASE", ".")

LLM_TYPES = ["gpt2", "mistral", "mistral-chat", "llama-chat", "mpt", "mpt-chat"]
DECODINGS = ["greedy", "sampling"]
REP_PENS = ["no", "yes"]


def all_configs():
    return [f"{L}_{D}_{R}" for L in LLM_TYPES for D in DECODINGS for R in REP_PENS]


def load_train_acts(cfg, acts_root, max_per_class=None, seed=42):
    d = f"{acts_root}/{cfg}"
    real = load_all_chunks(d, f"{cfg}_train_real")
    fake = load_all_chunks(d, f"{cfg}_train_fake")
    if max_per_class is not None:
        rng = np.random.default_rng(seed)
        if len(real) > max_per_class:
            real = real[rng.permutation(len(real))[:max_per_class]]
        if len(fake) > max_per_class:
            fake = fake[rng.permutation(len(fake))[:max_per_class]]
    return real, fake


def load_test_acts(cfg, acts_root):
    d = f"{acts_root}/{cfg}"
    real = load_all_chunks(d, f"{cfg}_test_real")
    fake = load_all_chunks(d, f"{cfg}_test_fake")
    return real, fake


def metrics(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    return {
        "AUROC": float(auc(fpr, tpr)),
        "TPR@1%FPR": tpr_at_fpr(fpr, tpr, 0.01),
        "TPR@5%FPR": tpr_at_fpr(fpr, tpr, 0.05),
        "TPR@10%FPR": tpr_at_fpr(fpr, tpr, 0.10),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="logreg", choices=["logreg", "mean", "pca"])
    p.add_argument("--acts-root", default=f"{BASE}/data/activations/raid",
                   help="use raid or raid_llama activations root")
    p.add_argument("--out-dir", default=f"{BASE}/results/raid_matrix")
    p.add_argument("--configs", nargs="+", default=None)
    p.add_argument("--tag", default="gptneo",
                   help="Filename tag for saved artefacts (e.g. gptneo or llama)")
    p.add_argument("--max-train-per-class", type=int, default=1800,
                   help="Subsample RAID train to keep SV fit tractable. "
                        "1800 matches paper's DetectRL scale; 0 = use full ~6544.")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    configs = args.configs or all_configs()
    max_pc = args.max_train_per_class if args.max_train_per_class > 0 else None

    # === Step 1: fit per-config SVs and per-config classifiers on train ===
    print(f"[step 1] fitting {len(configs)} SVs (method={args.method}, max_per_class={max_pc}) + per-config classifiers")
    svs = {}
    train_dots = {}  # (cfg) -> (X_train, y_train)
    clfs = {}
    for cfg in configs:
        try:
            real, fake = load_train_acts(cfg, args.acts_root, max_per_class=max_pc)
        except FileNotFoundError as e:
            print(f"  [skip] {cfg}: missing train acts ({e})")
            continue
        sv = compute_sv(fake, real, args.method)
        svs[cfg] = sv
        d_real = dot_products(real, sv); d_fake = dot_products(fake, sv)
        X = np.concatenate([d_real, d_fake], axis=0)
        y = np.concatenate([np.zeros(len(d_real), np.int8), np.ones(len(d_fake), np.int8)])
        clf = build_detector().fit(X, y)
        clfs[cfg] = clf
        train_dots[cfg] = (X, y)
        print(f"  [{cfg}] SV shape={sv.shape}, train N={len(X)}")

    good_configs = [c for c in configs if c in svs]

    # === Step 2: score every (train, eval) pair ===
    print(f"\n[step 2] scoring {len(good_configs)} × {len(good_configs)} = {len(good_configs)**2} cells")
    matrix_auroc = pd.DataFrame(index=good_configs, columns=good_configs, dtype=float)
    per_cell = {}

    for train_cfg in good_configs:
        sv = svs[train_cfg]; clf = clfs[train_cfg]
        for eval_cfg in good_configs:
            try:
                real, fake = load_test_acts(eval_cfg, args.acts_root)
            except FileNotFoundError:
                print(f"  [skip] eval {eval_cfg}: missing test acts")
                continue
            d_real = dot_products(real, sv); d_fake = dot_products(fake, sv)
            X = np.concatenate([d_real, d_fake], axis=0)
            y = np.concatenate([np.zeros(len(d_real), np.int8), np.ones(len(d_fake), np.int8)])
            probs = clf.predict_proba(X)[:, 1]
            m = metrics(y, probs)
            matrix_auroc.loc[train_cfg, eval_cfg] = m["AUROC"]
            per_cell[f"{train_cfg}->{eval_cfg}"] = m

    # === Step 3: aggregates ===
    print("\n=== 24x24 AUROC matrix (train↓ / eval→): ===")
    print(matrix_auroc.to_string(float_format=lambda x: f"{x:.3f}"))

    diag = np.diag(matrix_auroc.values.astype(float))
    off_diag = matrix_auroc.values.astype(float).copy()
    np.fill_diagonal(off_diag, np.nan)
    id_mean = np.nanmean(diag)
    ood_mean = np.nanmean(off_diag)

    print(f"\nID (diagonal, in-domain) mean AUROC: {id_mean:.4f} ({len(diag)} cells)")
    print(f"OOD (off-diagonal) mean AUROC:        {ood_mean:.4f}")

    # Per-LLM-type breakdown (mean over decoding × rep_penalty combos, within same llm_type)
    llm_of = lambda c: c.rsplit("_", 2)[0]
    per_llm_id, per_llm_ood = {}, {}
    for L in LLM_TYPES:
        llm_configs = [c for c in good_configs if llm_of(c) == L]
        # In-domain: diagonal cells where train == eval and both in this llm_type
        id_cells = [matrix_auroc.loc[c, c] for c in llm_configs if not np.isnan(matrix_auroc.loc[c, c])]
        # Same-LLM cross-source: train in this llm, eval in this llm, train != eval
        ood_cells = []
        for a in llm_configs:
            for b in llm_configs:
                if a != b and not np.isnan(matrix_auroc.loc[a, b]):
                    ood_cells.append(matrix_auroc.loc[a, b])
        per_llm_id[L] = np.mean(id_cells) if id_cells else np.nan
        per_llm_ood[L] = np.mean(ood_cells) if ood_cells else np.nan

    print("\nPer-LLM-type (mean AUROC):")
    for L in LLM_TYPES:
        print(f"  {L:15s}  ID={per_llm_id[L]:.4f}   within-LLM OOD={per_llm_ood[L]:.4f}")

    # dump
    matrix_auroc.to_csv(f"{args.out_dir}/raid_auroc_matrix_{args.tag}_{args.method}.csv")
    with open(f"{args.out_dir}/raid_matrix_full_{args.tag}_{args.method}.json", "w") as f:
        json.dump({
            "id_mean": id_mean,
            "ood_mean": ood_mean,
            "per_llm_id": per_llm_id,
            "per_llm_ood": per_llm_ood,
            "cells": per_cell,
        }, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"\nwrote {args.out_dir}/raid_*_{args.tag}_{args.method}.{{csv,json}}")


if __name__ == "__main__":
    main()
