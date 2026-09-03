"""Read every dot-product file we've produced and compute rebuttal-relevant
metrics. Prints tables and dumps JSON per experiment.

Experiments:
  md_16cell:      Milestone-1 sanity check (paper Table 8)
  mirage_3vec:    Paper's main MIRAGE result (unified classifier on 3-vec SV)
  reverse_transfer: Exp 3 (SV-Detect on MD, test on MIRAGE)
  mirage_1vec:    Exp 5 (single-vector MIRAGE compared to 3-vec)
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from src.extract.nb_pipeline import build_detector, metrics


BASE = os.environ.get("SVDETECT_BASE", ".")
DOTS = f"{BASE}/data/dots"
OUT  = f"{BASE}/results"

MD_SUBS   = ["arxiv", "writing_prompt", "xsum", "yelp_review"]
MIR_TASKS = ["generate", "polish", "rewrite"]
MIR_SCEN  = ["DIG", "SIG"]


def _load(path):
    return np.load(path)


def _stack(real, fake):
    X = np.concatenate([real, fake], axis=0)
    y = np.concatenate([np.zeros(len(real), np.int8), np.ones(len(fake), np.int8)])
    return X, y


def exp_md_16cell(method: str = "logreg") -> dict:
    """Table-8-like matrix for DetectRL Multi-Domain (per-subset SVs)."""
    d = f"{DOTS}/detectrl_md"
    auroc = pd.DataFrame(index=MD_SUBS, columns=MD_SUBS, dtype=float)
    f1    = pd.DataFrame(index=MD_SUBS, columns=MD_SUBS, dtype=float)
    per_cell = {}
    for sv_sub in MD_SUBS:
        real = _load(f"{d}/{sv_sub}_train_real_dot_products_{sv_sub}_steering_vectors_{method}.npy")
        fake = _load(f"{d}/{sv_sub}_train_fake_dot_products_{sv_sub}_steering_vectors_{method}.npy")
        Xtr, ytr = _stack(real, fake)
        clf = build_detector().fit(Xtr, ytr)
        for ev_sub in MD_SUBS:
            r = _load(f"{d}/{ev_sub}_test_real_dot_products_{sv_sub}_steering_vectors_{method}.npy")
            f = _load(f"{d}/{ev_sub}_test_fake_dot_products_{sv_sub}_steering_vectors_{method}.npy")
            Xte, yte = _stack(r, f)
            probs = clf.predict_proba(Xte)[:, 1]
            m = metrics(yte, probs)
            per_cell[f"{sv_sub}->{ev_sub}"] = m
            auroc.loc[sv_sub, ev_sub] = m["AUROC"]
            f1.loc[sv_sub, ev_sub]    = m["F1"]
    print(f"[MD 16-cell, method={method}]\nAUROC:")
    print(auroc.to_string(float_format=lambda x: f"{x:.4f}"))
    print("F1:")
    print(f1.to_string(float_format=lambda x: f"{x:.4f}"))
    return {"auroc": auroc.to_dict(), "f1": f1.to_dict(), "cells": per_cell}


def _unified_train_from(dots_dir: str, tasks: list[str], method: str) -> tuple:
    real_arrs = [_load(f"{dots_dir}/train_{t}_real_dot_products_{method}.npy") for t in tasks]
    fake_arrs = [_load(f"{dots_dir}/train_{t}_fake_dot_products_{method}.npy") for t in tasks]
    real = np.concatenate(real_arrs, axis=0)
    fake = np.concatenate(fake_arrs, axis=0)
    return _stack(real, fake)


def _test_on_mirage(clf, dots_dir: str, method: str) -> dict:
    out = {}
    for scen in MIR_SCEN:
        for task in MIR_TASKS:
            r = _load(f"{dots_dir}/{scen}_{task}_test_real_dot_products_{method}.npy")
            f = _load(f"{dots_dir}/{scen}_{task}_test_fake_dot_products_{method}.npy")
            X, y = _stack(r, f)
            probs = clf.predict_proba(X)[:, 1]
            m = metrics(y, probs)
            out[f"{scen}_{task}"] = m
            print(f"  {scen}_{task:9s}  AUROC={m['AUROC']:.4f}  F1={m['F1']:.4f}  Acc={m['Balanced Accuracy']:.4f}")
    return out


def exp_mirage_3vec(method: str = "logreg") -> dict:
    """Paper's main MIRAGE result — train on 800 MIRAGE pairs projected onto 3-vec system,
    evaluate on 6 MIRAGE test files."""
    d = f"{DOTS}/mirage_3vec"
    Xtr, ytr = _unified_train_from(d, MIR_TASKS, method)
    print(f"[MIRAGE 3-vec, method={method}]  train X={Xtr.shape} pos={ytr.sum()}")
    clf = build_detector().fit(Xtr, ytr)
    per_test = _test_on_mirage(clf, d, method)
    return {"per_test": per_test}


def exp_reverse_transfer(method: str = "logreg") -> dict:
    """Exp 3: Train unified classifier on DetectRL Multi-Domain 4-subset dots (projected onto
    MD-4vec orthonormal system), evaluate on 6 MIRAGE test files."""
    d = f"{DOTS}/reverse_md4vec"
    real_arrs = [_load(f"{d}/{s}_train_real_dot_products_{method}.npy") for s in MD_SUBS]
    fake_arrs = [_load(f"{d}/{s}_train_fake_dot_products_{method}.npy") for s in MD_SUBS]
    Xtr, ytr = _stack(np.concatenate(real_arrs, axis=0), np.concatenate(fake_arrs, axis=0))
    print(f"[REVERSE TRANSFER, method={method}] train X={Xtr.shape} pos={ytr.sum()}")
    clf = build_detector().fit(Xtr, ytr)
    per_test = _test_on_mirage(clf, d, method)
    return {"per_test": per_test}


def exp_mirage_1vec(method: str = "logreg") -> dict:
    """Exp 5: Train on MIRAGE 800 pairs projected onto SINGLE union SV (no QR),
    evaluate on 6 MIRAGE test files. Compare to exp_mirage_3vec."""
    d = f"{DOTS}/mirage_1vec"
    Xtr, ytr = _unified_train_from(d, MIR_TASKS, method)
    print(f"[MIRAGE 1-vec, method={method}]  train X={Xtr.shape} pos={ytr.sum()}")
    clf = build_detector().fit(Xtr, ytr)
    per_test = _test_on_mirage(clf, d, method)
    return {"per_test": per_test}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiments", nargs="+",
                   default=["md_16cell", "mirage_3vec", "reverse_transfer", "mirage_1vec"])
    p.add_argument("--methods", nargs="+", default=["logreg", "mean", "pca"])
    args = p.parse_args()
    os.makedirs(OUT, exist_ok=True)

    results = {}
    fns = {
        "md_16cell":        exp_md_16cell,
        "mirage_3vec":      exp_mirage_3vec,
        "reverse_transfer": exp_reverse_transfer,
        "mirage_1vec":      exp_mirage_1vec,
    }
    for exp in args.experiments:
        results[exp] = {}
        for method in args.methods:
            print(f"\n============= {exp} / {method} =============")
            try:
                results[exp][method] = fns[exp](method)
            except FileNotFoundError as e:
                print(f"[!! MISSING] {e}")
                results[exp][method] = {"error": str(e)}

    # dump results per experiment
    for exp, per_method in results.items():
        with open(f"{OUT}/{exp}.json", "w") as f:
            json.dump(per_method, f, indent=2, default=lambda x: x if not isinstance(x, pd.DataFrame) else x.to_dict())
        print(f"wrote {OUT}/{exp}.json")


if __name__ == "__main__":
    main()
