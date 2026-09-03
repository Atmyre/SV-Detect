"""Score a trained RoBERTa-fair detector on all 5 PADBen tasks, then compute
metrics under all 5 setups (single-sentence exhaustive/30-70/50-50/80-20 +
sentence-pair). Faithful to PADBen paper Section 5.1.

Outputs one JSON per (model, task, setup) block plus a wide summary CSV.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.analysis.padben_eval import (TASK_JSON_STEM, tpr_at_fpr,
                                      _exhaustive_json_for_task,
                                      _sampling_json_for_task,
                                      _pair_json_for_task,
                                      load_pair_records)


BASE = os.environ.get("SVDETECT_BASE", ".")


class SentenceDataset(Dataset):
    def __init__(self, sentences, tokenizer, max_length):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, i):
        enc = self.tokenizer(self.sentences[i], truncation=True,
                             padding="max_length", max_length=self.max_length,
                             return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0)}


@torch.no_grad()
def score_sentences(model, tokenizer, sentences, batch_size, max_length, device):
    ds = SentenceDataset(sentences, tokenizer, max_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    scores = []
    model.eval()
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        probs = torch.softmax(logits, dim=-1)[:, 1].detach().float().cpu().numpy()
        scores.append(probs)
    return np.concatenate(scores)


def metrics_from_scores(y_true, scores):
    fpr, tpr, _ = roc_curve(y_true, scores)
    return {
        "AUROC": float(auc(fpr, tpr)),
        "TPR@1%FPR": tpr_at_fpr(fpr, tpr, 0.01),
        "TPR@5%FPR": tpr_at_fpr(fpr, tpr, 0.05),
        "TPR@10%FPR": tpr_at_fpr(fpr, tpr, 0.10),
        "n_pos": int(y_true.sum()),
        "n_neg": int((1 - y_true).sum()),
    }


def eval_single_task(task_id, sent_to_score, sent_to_label, setup):
    if setup == "exhaustive":
        path = _exhaustive_json_for_task(task_id)
    else:
        path = _sampling_json_for_task(task_id, setup)
    with open(path) as f:
        rows = json.load(f)
    y = np.array([r["label"] for r in rows], dtype=np.int8)
    s = np.array([sent_to_score.get(r["sentence"], np.nan) for r in rows])
    mask = ~np.isnan(s)
    if mask.sum() < len(s):
        print(f"      [warn] {(~mask).sum()}/{len(s)} scored=nan (sentence missing?)")
    return metrics_from_scores(y[mask], s[mask])


def eval_sentence_pair(task_id, sent_to_score):
    pair_rows = load_pair_records(_pair_json_for_task(task_id))
    y_bin = []
    diffs = []
    n_missing = 0
    for r in pair_rows:
        s1, s2 = r["sentence_pair"]
        l1, l2 = r["label_pair"]
        sc1 = sent_to_score.get(s1); sc2 = sent_to_score.get(s2)
        if sc1 is None or sc2 is None:
            n_missing += 1
            continue
        diffs.append(sc2 - sc1)
        y_bin.append(1 if (l1 == 0 and l2 == 1) else 0)
    m = metrics_from_scores(np.array(y_bin, dtype=np.int8), np.array(diffs))
    m["n_missing"] = n_missing
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True,
                   help="Path to a HuggingFace-formatted saved RoBERTa model dir "
                        "(e.g. results/roberta_fair/roberta-base/model)")
    p.add_argument("--model-label", required=True,
                   help="A short label for this model, used in output filenames")
    p.add_argument("--tasks", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--out-dir", default=f"{BASE}/results/padben_roberta")
    args = p.parse_args()

    device = "cuda"
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"loading model {args.model_path}")
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path).to(device)

    all_results = {}
    for task_id in args.tasks:
        print(f"\n=== Task {task_id} ===")
        with open(_exhaustive_json_for_task(task_id)) as f:
            rows = json.load(f)
        sentences = [r["sentence"] for r in rows]
        labels = np.array([r["label"] for r in rows], dtype=np.int8)
        print(f"  scoring {len(sentences)} sentences with batch_size={args.batch_size}")
        scores = score_sentences(model, tok, sentences, args.batch_size,
                                 args.max_length, device)
        sent_to_score = dict(zip(sentences, scores))
        sent_to_label = dict(zip(sentences, labels))

        for setup in ["exhaustive", "30-70", "50-50", "80-20"]:
            m = eval_single_task(task_id, sent_to_score, sent_to_label, setup)
            key = f"{args.model_label}/task{task_id}/single-{setup}"
            all_results[key] = m
            print(f"    {setup:11s} AUC={m['AUROC']:.4f} "
                  f"T1={m['TPR@1%FPR']:.3f} T5={m['TPR@5%FPR']:.3f} T10={m['TPR@10%FPR']:.3f}")

        m = eval_sentence_pair(task_id, sent_to_score)
        key = f"{args.model_label}/task{task_id}/sentence-pair"
        all_results[key] = m
        print(f"    sent-pair   AUC={m['AUROC']:.4f} "
              f"T1={m['TPR@1%FPR']:.3f} T5={m['TPR@5%FPR']:.3f} T10={m['TPR@10%FPR']:.3f} "
              f"(missing={m.get('n_missing', 0)})")

    out_path = f"{args.out_dir}/{args.model_label}_padben.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
