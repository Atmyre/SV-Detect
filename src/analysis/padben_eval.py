"""Evaluate a pretrained SV-Detect on PADBen — all 5 tasks × all 5 setups.

Faithful to PADBen paper (arXiv 2511.00416) Section 5.1:
  1. Single-Sentence Exhaustive         (all samples, balanced 50-50)
  2. Single-Sentence Sampling (30-70)   (30% positive, 70% negative)
  3. Single-Sentence Sampling (50-50)   (balanced random subsample)
  4. Single-Sentence Sampling (80-20)   (80% positive, 20% negative)
  5. Sentence-Pair Recognition          (pairwise, random-order; per-pair score = s(s2)-s(s1))

Metrics per (task, setup) cell: AUROC, TPR@1%FPR, TPR@5%FPR, TPR@10%FPR
(matches PADBen Tables 3-4 exactly).

Uses precomputed activations from GPT-Neo-2.7B (SV-Detect paper backbone),
projected onto one of our already-trained SV systems:
  - MIRAGE-3vec-orthonormal + logreg classifier (paper's main MIRAGE detector)
  - MIRAGE-1vec (union-800, Exp 5 setup)
  - MD-4vec-orthonormal (Exp 3 reverse-transfer detector)

No PADBen-specific training. Detector is applied as pretrained.
"""

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

from src.extract.nb_pipeline import (build_detector, dot_products,
                                     load_all_chunks)


BASE = os.environ.get("SVDETECT_BASE", ".")
PADBEN_ROOT = f"{BASE}/data/PADBen/single-sentence"
PADBEN_SP_ROOT = f"{BASE}/data/PADBen/sentence-pair"
ACTS = f"{BASE}/data/activations/padben"

METRICS_TPR_AT_FPR = [0.01, 0.05, 0.10]


def load_sentences_and_labels(json_path: str) -> tuple[list[str], np.ndarray]:
    with open(json_path) as f:
        rows = json.load(f)
    return [r["sentence"] for r in rows], np.array([r["label"] for r in rows], dtype=np.int8)


def load_pair_records(json_path: str) -> list[dict]:
    with open(json_path) as f:
        return json.load(f)


def tpr_at_fpr(fpr: np.ndarray, tpr: np.ndarray, target: float) -> float:
    """Interpolated TPR at the given FPR target. Matches PADBen's reporting."""
    if fpr[0] > target:
        return 0.0
    idx = np.searchsorted(fpr, target, side="right") - 1
    if idx >= len(fpr) - 1:
        return float(tpr[-1])
    x0, x1 = fpr[idx], fpr[idx + 1]
    y0, y1 = tpr[idx], tpr[idx + 1]
    if x1 == x0:
        return float(y0)
    return float(y0 + (target - x0) / (x1 - x0) * (y1 - y0))


def metrics_from_scores(y_true: np.ndarray, scores: np.ndarray) -> dict:
    fpr, tpr, _ = roc_curve(y_true, scores)
    return {
        "AUROC": float(auc(fpr, tpr)),
        "TPR@1%FPR": tpr_at_fpr(fpr, tpr, 0.01),
        "TPR@5%FPR": tpr_at_fpr(fpr, tpr, 0.05),
        "TPR@10%FPR": tpr_at_fpr(fpr, tpr, 0.10),
        "n_pos": int(y_true.sum()),
        "n_neg": int((1 - y_true).sum()),
    }


def sv_scores(dots: np.ndarray, clf) -> np.ndarray:
    """Detector probability of LGT class (label 1)."""
    return clf.predict_proba(dots)[:, 1]


def load_and_score_task(task_id: int, sv_path: str, clf):
    """Project extracted label0/label1 activations onto SV, get scores.
    Returns (scores_by_original_index, labels_by_original_index).

    IMPORTANT: PADBen's exhaustive JSON has interleaved label 0/1 samples. We
    extracted label0 and label1 as SEPARATE prefixes (order preserved *within*
    each label). To reconstruct scores in original file order, we need to
    re-merge by walking through the original JSON row-by-row and drawing from
    the correct queue.
    """
    task_acts_dir = f"{ACTS}/task{task_id}"
    sv = np.load(sv_path)
    acts0 = load_all_chunks(task_acts_dir, f"task{task_id}_label0")
    acts1 = load_all_chunks(task_acts_dir, f"task{task_id}_label1")
    d0 = dot_products(acts0, sv)
    d1 = dot_products(acts1, sv)
    s0 = sv_scores(d0, clf)
    s1 = sv_scores(d1, clf)

    # Reconstruct original ordering
    json_path = _exhaustive_json_for_task(task_id)
    with open(json_path) as f:
        rows = json.load(f)
    scores = np.empty(len(rows), dtype=np.float64)
    labels = np.empty(len(rows), dtype=np.int8)
    i0 = i1 = 0
    for i, r in enumerate(rows):
        if r["label"] == 0:
            scores[i] = s0[i0]; labels[i] = 0; i0 += 1
        else:
            scores[i] = s1[i1]; labels[i] = 1; i1 += 1
    assert i0 == len(s0) and i1 == len(s1), (i0, len(s0), i1, len(s1))
    return scores, labels, rows


TASK_JSON_STEM = {
    1: "task1_paraphrase_source_without_context",
    2: "task2_general_text_authorship_detection",
    3: "task3_ai_text_laundering_detection",
    4: "task4_iterative_paraphrase_depth_detection",
    5: "task5_original_vs_deep_paraphrase_attack",
}


def _exhaustive_json_for_task(task_id: int) -> str:
    stem = TASK_JSON_STEM[task_id]
    return f"{PADBEN_ROOT}/exhaustive_method/task{task_id}/{stem}.json"


def _sampling_json_for_task(task_id: int, variant: str) -> str:
    stem = TASK_JSON_STEM[task_id]
    return f"{PADBEN_ROOT}/sampling_method/{variant}/task{task_id}/dynamic_{stem}.json"


def _pair_json_for_task(task_id: int) -> str:
    stem = TASK_JSON_STEM[task_id]
    return f"{PADBEN_SP_ROOT}/task{task_id}/{stem}_sentence_pair.json"


def eval_single_sentence_task(task_id: int, scores: np.ndarray, labels: np.ndarray,
                              rows: list, setup: str) -> dict:
    """For 'exhaustive' return metrics over all scores. For sampling setups
    ('30-70' / '50-50' / '80-20') subsample by matching sentence *content* to
    PADBen's own subset file (safer than idx — some PADBen variants renumber)."""
    if setup == "exhaustive":
        return metrics_from_scores(labels, scores)
    variant_path = _sampling_json_for_task(task_id, setup)
    with open(variant_path) as f:
        sub_rows = json.load(f)
    sent_to_pos = {r["sentence"]: i for i, r in enumerate(rows)}
    hits, misses = [], 0
    for r in sub_rows:
        if r["sentence"] in sent_to_pos:
            hits.append(sent_to_pos[r["sentence"]])
        else:
            misses += 1
    if misses:
        print(f"      [warn] {misses}/{len(sub_rows)} sampling rows didn't match exhaustive by sentence")
    sub_scores = scores[np.array(hits)]
    sub_labels = np.array([sub_rows[i]["label"] for i, r in enumerate(sub_rows)
                           if r["sentence"] in sent_to_pos], dtype=np.int8)
    return metrics_from_scores(sub_labels, sub_scores)


def eval_sentence_pair(task_id: int, scores: np.ndarray, rows: list) -> dict:
    """Sentence-pair setup: for each labeled pair (label_pair ∈ {(0,1), (1,0)}),
    compute detector score-difference s(s2) - s(s1), then AUROC where the
    binary target is 1 if label_pair == (0,1) else 0.
    """
    pair_json = _pair_json_for_task(task_id)
    pair_rows = load_pair_records(pair_json)
    # Build sentence -> score lookup from `rows`+`scores` (need sentence content
    # since sentence-pair rows use fresh idxs that may or may not match).
    sent_to_score = {}
    for i, r in enumerate(rows):
        sent_to_score[r["sentence"]] = scores[i]

    y_bin = []
    diffs = []
    n_missing = 0
    for r in pair_rows:
        s1, s2 = r["sentence_pair"]
        l1, l2 = r["label_pair"]
        if s1 not in sent_to_score or s2 not in sent_to_score:
            n_missing += 1
            continue
        diff = sent_to_score[s2] - sent_to_score[s1]
        target = 1 if (l1 == 0 and l2 == 1) else 0
        y_bin.append(target)
        diffs.append(diff)
    y_bin = np.array(y_bin, dtype=np.int8)
    diffs = np.array(diffs)
    m = metrics_from_scores(y_bin, diffs)
    m["n_missing"] = n_missing
    return m


def train_classifier_from_dots(train_dots_root: str, method: str,
                               tasks: list[str] | None = None) -> "sklearn.pipeline.Pipeline":
    """Load train dot-products for a given detector, train Pipeline."""
    if tasks is None:
        tasks = ["generate", "polish", "rewrite"]
    real_arrs = [np.load(f"{train_dots_root}/train_{t}_real_dot_products_{method}.npy") for t in tasks]
    fake_arrs = [np.load(f"{train_dots_root}/train_{t}_fake_dot_products_{method}.npy") for t in tasks]
    real = np.concatenate(real_arrs, axis=0)
    fake = np.concatenate(fake_arrs, axis=0)
    X = np.concatenate([real, fake], axis=0)
    y = np.concatenate([np.zeros(len(real), np.int8), np.ones(len(fake), np.int8)])
    clf = build_detector().fit(X, y)
    return clf


def train_classifier_inline(det_cfg: dict) -> "sklearn.pipeline.Pipeline":
    """Compute train dots on-the-fly by projecting raw activations onto the SV.
    Used for detectors where we haven't precomputed the train dot-product file."""
    sv = np.load(det_cfg["sv_path"])
    acts_dir = det_cfg["train_acts_dir"]
    reals, fakes = [], []
    for _task, (real_pref, fake_pref) in det_cfg["train_prefixes"].items():
        r_acts = load_all_chunks(acts_dir, real_pref)
        f_acts = load_all_chunks(acts_dir, fake_pref)
        reals.append(dot_products(r_acts, sv))
        fakes.append(dot_products(f_acts, sv))
    X, y = _stack(reals, fakes)
    return build_detector().fit(X, y)


def _stack(reals, fakes):
    real = np.concatenate(reals, axis=0)
    fake = np.concatenate(fakes, axis=0)
    X = np.concatenate([real, fake], axis=0)
    y = np.concatenate([np.zeros(len(real), np.int8), np.ones(len(fake), np.int8)])
    return X, y


def train_classifier_md4vec(dots_root: str, method: str, subsets: list[str]) -> "sklearn.pipeline.Pipeline":
    real_arrs = [np.load(f"{dots_root}/{s}_train_real_dot_products_{method}.npy") for s in subsets]
    fake_arrs = [np.load(f"{dots_root}/{s}_train_fake_dot_products_{method}.npy") for s in subsets]
    X = np.concatenate([*real_arrs, *fake_arrs], axis=0)
    y = np.concatenate([
        np.zeros(sum(len(a) for a in real_arrs), np.int8),
        np.ones(sum(len(a) for a in fake_arrs), np.int8),
    ])
    return build_detector().fit(X, y)


DETECTORS = {
    "mirage_3vec": {
        "sv_path": f"{BASE}/data/svs/mirage/orthonormal_steering_vectors_logreg.npy",
        "train_dots_root": f"{BASE}/data/dots/mirage_3vec",
        "train_tasks": ["generate", "polish", "rewrite"],
    },
    "mirage_1vec": {
        "sv_path": f"{BASE}/data/svs/mirage_1vec/all_steering_vectors_logreg.npy",
        "train_dots_root": f"{BASE}/data/dots/mirage_1vec",
        "train_tasks": ["generate", "polish", "rewrite"],
    },
    "md_4vec": {
        "sv_path": f"{BASE}/data/svs/detectrl_md_4vec/orthonormal_steering_vectors_logreg.npy",
        "train_dots_root": f"{BASE}/data/dots/reverse_md4vec",
        "md_subsets": ["arxiv", "writing_prompt", "xsum", "yelp_review"],
    },
    # Polish-only-500: single SV fit on just the 500 polish pairs (matches paper's
    # SV-Detect (polish-only) row). Uses inline dot-product computation from raw
    # activations, since we don't have precomputed dots against this SV yet.
    "mirage_polish_only": {
        "sv_path": f"{BASE}/data/svs/mirage/polish_steering_vectors_logreg.npy",
        "train_acts_dir": f"{BASE}/data/activations/mirage/train",
        "train_prefixes": {"polish": ("train_polish_real", "train_polish_fake")},
        "inline_dots": True,
    },
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--detectors", nargs="+", default=list(DETECTORS.keys()))
    p.add_argument("--tasks", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--out-dir", default=f"{BASE}/results/padben")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
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
        print("  classifier trained")

        for task_id in args.tasks:
            print(f"  === Task {task_id} ===")
            try:
                scores, labels, rows = load_and_score_task(task_id, det["sv_path"], clf)
            except Exception as e:
                print(f"    [!! failed] {e}")
                continue
            print(f"    scored {len(scores)} sentences, mean(score)={scores.mean():.4f}")

            for setup in ["exhaustive", "30-70", "50-50", "80-20"]:
                m = eval_single_sentence_task(task_id, scores, labels, rows, setup)
                key = f"{det_name}/task{task_id}/single-{setup}"
                all_results[key] = m
                print(f"    {setup:11s} AUC={m['AUROC']:.4f} "
                      f"T1={m['TPR@1%FPR']:.3f} T5={m['TPR@5%FPR']:.3f} T10={m['TPR@10%FPR']:.3f}")

            m = eval_sentence_pair(task_id, scores, rows)
            key = f"{det_name}/task{task_id}/sentence-pair"
            all_results[key] = m
            print(f"    sent-pair   AUC={m['AUROC']:.4f} "
                  f"T1={m['TPR@1%FPR']:.3f} T5={m['TPR@5%FPR']:.3f} T10={m['TPR@10%FPR']:.3f} "
                  f"(missing={m.get('n_missing', 0)})")

    with open(f"{args.out_dir}/padben_full_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n=========== SUMMARY (AUROC) ===========")
    rows = []
    for det_name in args.detectors:
        for task_id in args.tasks:
            row = {"detector": det_name, "task": task_id}
            for setup in ["exhaustive", "30-70", "50-50", "80-20", "sentence-pair"]:
                key = f"{det_name}/task{task_id}/single-{setup}" if setup != "sentence-pair" else f"{det_name}/task{task_id}/sentence-pair"
                if key in all_results:
                    row[setup] = all_results[key]["AUROC"]
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(f"{args.out_dir}/padben_auroc_summary.csv", index=False)
    print(df.to_string(index=False))
    print(f"\nwrote {args.out_dir}/padben_full_results.json + padben_auroc_summary.csv")


if __name__ == "__main__":
    main()
