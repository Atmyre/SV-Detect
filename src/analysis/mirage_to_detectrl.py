"""MIRAGE-trained SV-Detect → DetectRL transfer, comparing 3 SV construction variants:
  - polish-only-500 (single SV fit on 500 polish pairs — paper's ablation row)
  - union-of-800 1-vec  (single SV fit on all 800 pairs — our Exp 5)
  - 3-vec orthonormal   (paper's main system)

Evaluates on any DetectRL activation collection (Multi-Domain / Multi-LLM / Multi-Attack)
by taking test activations from --acts-root/<subset>/<subset>_test_{real,fake}_*.npy.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

from src.analysis.padben_eval import (DETECTORS, train_classifier_from_dots,
                                       train_classifier_inline, tpr_at_fpr)
from src.extract.nb_pipeline import dot_products, load_all_chunks


def metrics(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    return {
        "AUROC": float(auc(fpr, tpr)),
        "TPR@1%FPR": tpr_at_fpr(fpr, tpr, 0.01),
        "TPR@5%FPR": tpr_at_fpr(fpr, tpr, 0.05),
        "TPR@10%FPR": tpr_at_fpr(fpr, tpr, 0.10),
    }


def load_classifier(det_name: str):
    det = DETECTORS[det_name]
    if det.get("inline_dots"):
        return train_classifier_inline(det), det["sv_path"]
    else:
        return train_classifier_from_dots(det["train_dots_root"], "logreg",
                                          det["train_tasks"]), det["sv_path"]


def score_subset(acts_dir: str, prefix_real: str, prefix_fake: str,
                 sv_path: str, clf):
    sv = np.load(sv_path)
    real = load_all_chunks(acts_dir, prefix_real)
    fake = load_all_chunks(acts_dir, prefix_fake)
    d_real = dot_products(real, sv)
    d_fake = dot_products(fake, sv)
    X = np.concatenate([d_real, d_fake], axis=0)
    y = np.concatenate([np.zeros(len(d_real), np.int8),
                       np.ones(len(d_fake), np.int8)])
    probs = clf.predict_proba(X)[:, 1]
    return metrics(y, probs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--acts-root", required=True,
                   help="Root dir; expects <acts-root>/<subset>/<subset>_test_{real,fake}_*.npy")
    p.add_argument("--subsets", nargs="+", required=True)
    p.add_argument("--detectors", nargs="+",
                   default=["mirage_polish_only", "mirage_1vec", "mirage_3vec"])
    p.add_argument("--split", default="test",
                   help="Split to evaluate — 'test' by default, could also be 'train'")
    p.add_argument("--out-file", required=True)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)

    print(f"\n=== Training classifiers ===")
    detector_state = {}
    for det_name in args.detectors:
        print(f"  training {det_name}...")
        clf, sv_path = load_classifier(det_name)
        detector_state[det_name] = (clf, sv_path)

    all_results = {}
    for det_name in args.detectors:
        clf, sv_path = detector_state[det_name]
        print(f"\n=== Detector: {det_name} ===")
        for sub in args.subsets:
            acts_dir = f"{args.acts_root}/{sub}"
            try:
                m = score_subset(acts_dir, f"{sub}_{args.split}_real",
                                 f"{sub}_{args.split}_fake",
                                 sv_path, clf)
            except FileNotFoundError as e:
                print(f"  [skip] {sub}: {e}")
                continue
            key = f"{det_name}/{sub}"
            all_results[key] = m
            print(f"  {sub:25s} AUC={m['AUROC']:.4f} T1={m['TPR@1%FPR']:.3f} "
                  f"T5={m['TPR@5%FPR']:.3f} T10={m['TPR@10%FPR']:.3f}")

    print("\n=== Summary (AUROC × TPR@5%FPR × 100) ===")
    rows = []
    for det_name in args.detectors:
        for sub in args.subsets:
            key = f"{det_name}/{sub}"
            if key in all_results:
                m = all_results[key]
                rows.append({"detector": det_name, "subset": sub,
                             "AUROC": m["AUROC"], "TPR@5%FPR": m["TPR@5%FPR"] * 100})
    df = pd.DataFrame(rows)
    if not df.empty:
        piv = df.pivot(index="detector", columns="subset", values="AUROC")
        print("\nAUROC:")
        print(piv.to_string(float_format=lambda x: f"{x:.4f}"))
        piv_tpr = df.pivot(index="detector", columns="subset", values="TPR@5%FPR")
        print("\nTPR@5%FPR (%):")
        print(piv_tpr.to_string(float_format=lambda x: f"{x:6.2f}"))

        print("\nPer-detector means (across subsets):")
        for det_name in args.detectors:
            sub = df[df.detector == det_name]
            print(f"  {det_name:22s} AUROC mean {sub['AUROC'].mean():.4f}  TPR@5% mean {sub['TPR@5%FPR'].mean():.2f}")

    with open(args.out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nwrote {args.out_file}")


if __name__ == "__main__":
    main()
