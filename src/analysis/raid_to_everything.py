"""Test a RAID-trained SV-Detect on DetectRL / MIRAGE / PADBen. Completes the
3-benchmark cross-transfer story alongside:
  - MIRAGE-trained → DetectRL/MIRAGE/PADBen  (Exp 2 + Exp 7 mirage_3vec / mirage_1vec)
  - DetectRL-trained → MIRAGE/DetectRL/PADBen (Exp 3 + Exp 7 md_4vec)

To keep things tractable and analogous to md_4vec, we:
  1. Fit one per-LLM_TYPE SV (6 SVs, aggregated across 4 decoding×rep_penalty configs)
  2. QR-orthonormalize into a (32, hidden, 6) system → 32*6=192 features per sample
  3. Train one StandardScaler+LogReg classifier on aggregated RAID train dot products
  4. Project every other benchmark's test activations onto the RAID-6vec system, score

Reports AUROC per test set:
  - DetectRL Multi-Domain (4 subsets, in-domain test)
  - MIRAGE 6 test files (DIG/SIG × generate/polish/rewrite)
  - PADBen tasks 1-5 (single-sentence exhaustive + sentence-pair)
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

from src.analysis.padben_eval import (tpr_at_fpr, _exhaustive_json_for_task,
                                       _pair_json_for_task, load_pair_records,
                                       load_and_score_task)
from src.extract.nb_pipeline import (build_detector, compute_sv, dot_products,
                                     load_all_chunks, qr_orthonormalize)


BASE = os.environ.get("SVDETECT_BASE", ".")

RAID_LLMS = ["gpt2", "mistral", "mistral-chat", "llama-chat", "mpt", "mpt-chat"]
RAID_DECODINGS = ["greedy", "sampling"]
RAID_REP_PENS = ["no", "yes"]


def raid_configs_for_llm(L):
    return [f"{L}_{D}_{R}" for D in RAID_DECODINGS for R in RAID_REP_PENS]


def all_raid_configs():
    return [f"{L}_{D}_{R}" for L in RAID_LLMS for D in RAID_DECODINGS for R in RAID_REP_PENS]


def concat_train_acts(cfgs, acts_root, max_per_class=None, seed=42):
    reals, fakes = [], []
    for c in cfgs:
        d = f"{acts_root}/{c}"
        try:
            reals.append(load_all_chunks(d, f"{c}_train_real"))
            fakes.append(load_all_chunks(d, f"{c}_train_fake"))
        except FileNotFoundError:
            print(f"    [skip missing train] {c}")
    real = np.concatenate(reals, axis=0)
    fake = np.concatenate(fakes, axis=0)
    if max_per_class is not None:
        rng = np.random.default_rng(seed)
        if len(real) > max_per_class:
            real = real[rng.permutation(len(real))[:max_per_class]]
        if len(fake) > max_per_class:
            fake = fake[rng.permutation(len(fake))[:max_per_class]]
    return real, fake


def build_raid_6vec_system(method: str, acts_root: str, max_per_class_per_llm: int | None = None):
    """Fit per-llm_type SV on RAID train data (union of 4 configs per llm),
    QR-orthonormalize into (L, H, 6) system, then compute training features."""
    svs = []
    train_reals, train_fakes = [], []
    for L in RAID_LLMS:
        cfgs = raid_configs_for_llm(L)
        r, f = concat_train_acts(cfgs, acts_root, max_per_class=max_per_class_per_llm)
        print(f"    [{L}] real={r.shape}, fake={f.shape}")
        sv = compute_sv(f, r, method)
        svs.append(sv)
        train_reals.append(r); train_fakes.append(f)
    Q = qr_orthonormalize(svs)  # (L, H, 6)
    print(f"  RAID-6vec orthonormal system shape: {Q.shape}")

    # For classifier training, use the union of ALL RAID train activations projected
    # onto Q. Features: (N, L*6) = (N, 192).
    all_real = np.concatenate(train_reals, axis=0)
    all_fake = np.concatenate(train_fakes, axis=0)
    print(f"  Total RAID train: real={all_real.shape}, fake={all_fake.shape}")
    d_real = dot_products(all_real, Q); d_fake = dot_products(all_fake, Q)
    X = np.concatenate([d_real, d_fake], axis=0)
    y = np.concatenate([np.zeros(len(d_real), np.int8), np.ones(len(d_fake), np.int8)])
    clf = build_detector().fit(X, y)
    print(f"  classifier trained on {len(X)} samples, {X.shape[1]} features")
    return Q, clf


def metrics(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    return {
        "AUROC": float(auc(fpr, tpr)),
        "TPR@1%FPR": tpr_at_fpr(fpr, tpr, 0.01),
        "TPR@5%FPR": tpr_at_fpr(fpr, tpr, 0.05),
        "TPR@10%FPR": tpr_at_fpr(fpr, tpr, 0.10),
    }


def _detectrl_md_prefixes():
    """(subset_name, acts_dir, real_prefix, fake_prefix) for DetectRL Multi-Domain test sets."""
    subs = ["arxiv", "writing_prompt", "xsum", "yelp_review"]
    for s in subs:
        yield s, f"{BASE}/data/activations/detectrl/multi_domains/{s}", \
              f"{s}_test_real", f"{s}_test_fake"


def _mirage_prefixes():
    for scen in ["DIG", "SIG"]:
        for task in ["generate", "polish", "rewrite"]:
            yield f"{scen}_{task}", f"{BASE}/data/activations/mirage/test/{scen}/{task}", \
                  f"{scen}_{task}_real", f"{scen}_{task}_fake"


def eval_on(scen_name, acts_dir, real_prefix, fake_prefix, sv, clf):
    try:
        r = load_all_chunks(acts_dir, real_prefix)
        f = load_all_chunks(acts_dir, fake_prefix)
    except FileNotFoundError as e:
        print(f"  [skip missing] {scen_name}: {e}")
        return None
    d_r = dot_products(r, sv); d_f = dot_products(f, sv)
    X = np.concatenate([d_r, d_f], axis=0)
    y = np.concatenate([np.zeros(len(d_r), np.int8), np.ones(len(d_f), np.int8)])
    probs = clf.predict_proba(X)[:, 1]
    return metrics(y, probs)


def eval_padben(sv, clf, tasks=(1, 2, 3, 4, 5)):
    """PADBen has one activation dir per task with label0/label1 prefixes. Use
    the same reconstruction logic as padben_eval.load_and_score_task."""
    from src.analysis.padben_eval import load_and_score_task, eval_single_sentence_task, eval_sentence_pair
    # We need a hack: load_and_score_task uses its DETECTORS structure with sv_path.
    # We'll manually inline: extract task acts, project, score.
    results = {}
    for task_id in tasks:
        d = f"{BASE}/data/activations/padben/task{task_id}"
        try:
            a0 = load_all_chunks(d, f"task{task_id}_label0")
            a1 = load_all_chunks(d, f"task{task_id}_label1")
        except FileNotFoundError:
            continue
        d0 = dot_products(a0, sv); d1 = dot_products(a1, sv)
        s0 = clf.predict_proba(d0)[:, 1]
        s1 = clf.predict_proba(d1)[:, 1]
        # Rebuild scores in original PADBen row order
        with open(_exhaustive_json_for_task(task_id)) as fp:
            rows = json.load(fp)
        scores = np.empty(len(rows), dtype=np.float64)
        labels = np.empty(len(rows), dtype=np.int8)
        i0 = i1 = 0
        for i, r in enumerate(rows):
            if r["label"] == 0:
                scores[i] = s0[i0]; labels[i] = 0; i0 += 1
            else:
                scores[i] = s1[i1]; labels[i] = 1; i1 += 1
        # Single-sentence exhaustive
        results[f"task{task_id}/exhaustive"] = metrics(labels, scores)
        # Sentence-pair
        pair_rows = load_pair_records(_pair_json_for_task(task_id))
        sent_to_score = dict(zip([r["sentence"] for r in rows], scores))
        diffs, y_bin = [], []
        for r in pair_rows:
            s1_str, s2_str = r["sentence_pair"]
            l1, l2 = r["label_pair"]
            if s1_str in sent_to_score and s2_str in sent_to_score:
                diffs.append(sent_to_score[s2_str] - sent_to_score[s1_str])
                y_bin.append(1 if (l1 == 0 and l2 == 1) else 0)
        if diffs:
            results[f"task{task_id}/sentence-pair"] = metrics(np.array(y_bin, dtype=np.int8), np.array(diffs))
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="logreg", choices=["logreg", "mean", "pca"])
    p.add_argument("--acts-root", default=f"{BASE}/data/activations/raid")
    p.add_argument("--out-dir", default=f"{BASE}/results/raid_to_everything")
    p.add_argument("--tag", default="gptneo")
    p.add_argument("--max-train-per-class-per-llm", type=int, default=2000,
                   help="Cap the per-llm_type train pool to make SV fit tractable. "
                        "2000 = ~12k across 6 llms, matches DetectRL Multi-LLM scale.")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"=== Building RAID-6vec system (method={args.method}, cap per llm={args.max_train_per_class_per_llm}) ===")
    max_pc = args.max_train_per_class_per_llm if args.max_train_per_class_per_llm > 0 else None
    Q, clf = build_raid_6vec_system(args.method, args.acts_root, max_pc)

    all_results = {}

    print("\n=== DetectRL Multi-Domain test ===")
    for name, acts_dir, rp, fp in _detectrl_md_prefixes():
        m = eval_on(name, acts_dir, rp, fp, Q, clf)
        if m:
            all_results[f"DetectRL_MD/{name}"] = m
            print(f"  {name:20s} AUC={m['AUROC']:.4f} T1={m['TPR@1%FPR']:.3f} "
                  f"T5={m['TPR@5%FPR']:.3f} T10={m['TPR@10%FPR']:.3f}")

    print("\n=== MIRAGE (DIG + SIG × G/P/R) ===")
    for name, acts_dir, rp, fp in _mirage_prefixes():
        m = eval_on(name, acts_dir, rp, fp, Q, clf)
        if m:
            all_results[f"MIRAGE/{name}"] = m
            print(f"  {name:20s} AUC={m['AUROC']:.4f} T1={m['TPR@1%FPR']:.3f} "
                  f"T5={m['TPR@5%FPR']:.3f} T10={m['TPR@10%FPR']:.3f}")

    print("\n=== PADBen (tasks 1-5) ===")
    padben = eval_padben(Q, clf)
    for key, m in padben.items():
        all_results[f"PADBen/{key}"] = m
        print(f"  {key:25s} AUC={m['AUROC']:.4f} T1={m['TPR@1%FPR']:.3f} "
              f"T5={m['TPR@5%FPR']:.3f} T10={m['TPR@10%FPR']:.3f}")

    with open(f"{args.out_dir}/raid_to_everything_{args.tag}_{args.method}.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nwrote {args.out_dir}/raid_to_everything_{args.tag}_{args.method}.json")


if __name__ == "__main__":
    main()
