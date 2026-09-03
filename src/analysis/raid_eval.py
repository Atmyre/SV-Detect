"""Evaluate pretrained SV-Detect detectors on RAID (Dugan et al., ACL 2024).

RAID has 24 test configurations = 6 LLM types × 2 decoding strategies × 2 rep-penalty settings.
Each config has ~1000 paired records (human_text + direct_prompt = 2000 sentences).

Detectors evaluated (all pretrained, no RAID-specific training):
  - MIRAGE-3vec (paper's main MIRAGE detector)
  - MIRAGE-1vec (union-800 single SV)
  - MD-4vec (Exp 3 reverse-transfer detector)

Metrics per (detector, config): AUROC, TPR@1%/5%/10%FPR.
Aggregates: per-LLM-type mean, overall mean.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

from src.analysis.padben_eval import (DETECTORS, train_classifier_from_dots,
                                       train_classifier_md4vec,
                                       train_classifier_inline, tpr_at_fpr)
from src.extract.nb_pipeline import dot_products, load_all_chunks


BASE = os.environ.get("SVDETECT_BASE", ".")

LLM_TYPES = ["gpt2", "mistral", "mistral-chat", "llama-chat", "mpt", "mpt-chat"]
DECODINGS = ["greedy", "sampling"]
REP_PENS = ["no", "yes"]


def all_configs():
    return [f"{L}_{D}_{R}" for L in LLM_TYPES for D in DECODINGS for R in REP_PENS]


def metrics_from_scores(y_true, scores):
    fpr, tpr, _ = roc_curve(y_true, scores)
    return {
        "AUROC": float(auc(fpr, tpr)),
        "TPR@1%FPR": tpr_at_fpr(fpr, tpr, 0.01),
        "TPR@5%FPR": tpr_at_fpr(fpr, tpr, 0.05),
        "TPR@10%FPR": tpr_at_fpr(fpr, tpr, 0.10),
    }


def score_config(cfg: str, sv: np.ndarray, clf) -> tuple[np.ndarray, np.ndarray]:
    acts_dir = f"{BASE}/data/activations/raid/{cfg}"
    real = load_all_chunks(acts_dir, f"{cfg}_test_real")
    fake = load_all_chunks(acts_dir, f"{cfg}_test_fake")
    d_real = dot_products(real, sv)
    d_fake = dot_products(fake, sv)
    X = np.concatenate([d_real, d_fake], axis=0)
    y = np.concatenate([np.zeros(len(d_real), np.int8), np.ones(len(d_fake), np.int8)])
    probs = clf.predict_proba(X)[:, 1]
    return y, probs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--detectors", nargs="+", default=["mirage_3vec", "mirage_1vec", "md_4vec"])
    p.add_argument("--out-dir", default=f"{BASE}/results/raid")
    p.add_argument("--configs", nargs="+", default=None)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    configs = args.configs or all_configs()
    all_results = {}

    for det_name in args.detectors:
        det = DETECTORS[det_name]
        print(f"\n=========== DETECTOR: {det_name} ===========")
        if det.get("inline_dots"):
            clf = train_classifier_inline(det)
        elif det_name == "md_4vec":
            clf = train_classifier_md4vec(det["train_dots_root"], "logreg", det["md_subsets"])
        else:
            clf = train_classifier_from_dots(det["train_dots_root"], "logreg", det["train_tasks"])
        sv = np.load(det["sv_path"])
        print(f"  clf trained, SV shape={sv.shape}")

        for cfg in configs:
            try:
                y, probs = score_config(cfg, sv, clf)
                m = metrics_from_scores(y, probs)
                key = f"{det_name}/{cfg}"
                all_results[key] = m
                print(f"  {cfg:40s} AUC={m['AUROC']:.4f} T1={m['TPR@1%FPR']:.3f} "
                      f"T5={m['TPR@5%FPR']:.3f} T10={m['TPR@10%FPR']:.3f}")
            except FileNotFoundError as e:
                print(f"  {cfg:40s} [skip missing acts]")
                continue

    # Save & summarise
    with open(f"{args.out_dir}/raid_full_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    rows = []
    for det_name in args.detectors:
        for cfg in configs:
            key = f"{det_name}/{cfg}"
            if key in all_results:
                L, D, R = cfg.rsplit("_", 2)
                rows.append({"detector": det_name, "llm_type": L, "decoding": D,
                             "rep_penalty": R, "config": cfg,
                             **all_results[key]})
    df = pd.DataFrame(rows)
    df.to_csv(f"{args.out_dir}/raid_summary.csv", index=False)

    print("\n=========== Per-LLM-type AUROC ===========")
    pivot = df.pivot_table(index="llm_type", columns="detector", values="AUROC", aggfunc="mean")
    print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n=========== Overall means ===========")
    for det_name in args.detectors:
        sub = df[df.detector == det_name]
        print(f"  {det_name}: AUROC {sub['AUROC'].mean():.4f} "
              f"(min {sub['AUROC'].min():.4f}, max {sub['AUROC'].max():.4f}, "
              f"n_configs={len(sub)})")

    print(f"\nwrote {args.out_dir}/raid_full_results.json + raid_summary.csv")


if __name__ == "__main__":
    main()
