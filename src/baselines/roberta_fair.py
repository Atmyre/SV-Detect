"""RoBERTa-fair baseline for Exp 1 of the ARR rebuttal.

Trains RoBERTa (base or large) on the *same* augmented MIRAGE training set that
SV-Detect uses (500 polish + 150 generate + 150 rewrite = 800 pairs = 1,600
examples), then evaluates on DetectRL Multi-Domain / Multi-LLM / Multi-Attack.
This isolates the effect of the steering-vector representation from any
advantage the SV-Detect pipeline may derive from its non-standard training set.
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (auc, average_precision_score, balanced_accuracy_score,
                             f1_score, matthews_corrcoef, precision_recall_curve,
                             roc_curve)
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)


MIRAGE_TRAIN_FILES = {
    "polish":   "ai_detection_500_polish.raw_data.json",
    "generate": "xsum_generation_gpt-3.5-turbo.raw_data.json",
    "rewrite":  "xsum_rewrite_gpt-3.5-turbo.raw_data.json",
}


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_mirage_train(root: str, subset: str = "all") -> tuple[list[str], list[int]]:
    """subset ∈ {'all', 'polish', 'generate', 'rewrite'}.
    'all' = 800 pairs (500 polish + 150 generate + 150 rewrite), matches SV-Detect augmented.
    'polish' = 500 pairs, matches SV-Detect (polish-only) row."""
    files = ([MIRAGE_TRAIN_FILES[subset]] if subset in MIRAGE_TRAIN_FILES
             else list(MIRAGE_TRAIN_FILES.values()))
    texts, labels = [], []
    for fname in files:
        with open(os.path.join(root, fname)) as f:
            d = json.load(f)
        texts.extend(d["original"]);  labels.extend([0] * len(d["original"]))
        texts.extend(d["rewritten"]); labels.extend([1] * len(d["rewritten"]))
    return texts, labels


def load_detectrl_split(json_path: str) -> tuple[list[str], list[int]]:
    with open(json_path) as f:
        rows = json.load(f)
    texts  = [r["text"] for r in rows]
    labels = [0 if r["label"] == "human" else 1 for r in rows]
    return texts, labels


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tokenizer(
            self.texts[i],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[i], dtype=torch.long),
        }


def train(model, tokenizer, train_texts, train_labels, epochs, batch_size, lr,
          max_length, device, log_every=20):
    ds = TextDataset(train_texts, train_labels, tokenizer, max_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(loader) * epochs
    sched = get_linear_schedule_with_warmup(optim, int(0.06 * total_steps), total_steps)

    model.train()
    step = 0
    for ep in range(epochs):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); sched.step(); optim.zero_grad()
            step += 1
            if step % log_every == 0:
                print(f"epoch={ep} step={step}/{total_steps} loss={loss.item():.4f}")
        print(f"[epoch {ep} done]")


@torch.no_grad()
def eval_probs(model, tokenizer, texts, batch_size, max_length, device):
    model.eval()
    ds = TextDataset(texts, [0] * len(texts), tokenizer, max_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    scores = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        logits = model(**batch).logits
        probs = torch.softmax(logits, dim=-1)[:, 1].detach().float().cpu().numpy()
        scores.append(probs)
    return np.concatenate(scores)


def _tpr_at_fpr5(fpr, tpr):
    valid = np.where(fpr <= 0.05)[0]
    if not valid.size:
        return 0.0
    i = valid[-1]
    if i == len(fpr) - 1:
        return float(tpr[i])
    ratio = (0.05 - fpr[i]) / (fpr[i + 1] - fpr[i])
    return float(tpr[i] + ratio * (tpr[i + 1] - tpr[i]))


def metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    fpr, tpr, thresholds = roc_curve(labels, scores)
    prec, rec, _ = precision_recall_curve(labels, scores)
    denom = (prec + rec)
    f1_arr = np.divide(2 * prec * rec, denom, out=np.zeros_like(denom), where=denom > 0)
    return {
        "AUROC": float(auc(fpr, tpr)),
        "AUPR":  float(average_precision_score(labels, scores)),
        "TPR@FPR=5%": _tpr_at_fpr5(fpr, tpr),
        "Balanced Accuracy": float(np.max((tpr + 1 - fpr) / 2)),
        "MCC": float(np.max([matthews_corrcoef(labels, scores > t) for t in thresholds])),
        "F1": float(np.max(f1_arr)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="roberta-base")
    p.add_argument("--mirage-train-dir", required=True)
    p.add_argument("--detectrl-benchmark-dir", required=True,
                   help="Path to Benchmark_Data root (contains Multi_Domain/, Multi_LLM/, Multi_Attack/)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--train-subset", default="all",
                   choices=["all", "polish", "generate", "rewrite"],
                   help="'all' = 800 pairs (paper's augmented); 'polish' = 500-only (matches SV-Detect polish-only)")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    seed_all(args.seed)
    device = "cuda"
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Train ----
    print(f"loading MIRAGE training set from {args.mirage_train_dir}")
    tr_texts, tr_labels = load_mirage_train(args.mirage_train_dir, args.train_subset)
    print(f"training samples: {len(tr_texts)} "
          f"(human={sum(1 for x in tr_labels if x==0)}, "
          f"llm={sum(1 for x in tr_labels if x==1)})")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2).to(device)

    train(model, tok, tr_texts, tr_labels, args.epochs, args.batch_size,
          args.lr, args.max_length, device)

    # Save model checkpoint
    model.save_pretrained(os.path.join(args.out_dir, "model"))
    tok.save_pretrained(os.path.join(args.out_dir, "model"))

    # ---- Eval on every DetectRL test set ----
    results = {}
    subsets = {
        "Multi_Domain": [
            ("arxiv",          "multi_domains_arxiv_test.json"),
            ("writing_prompt", "multi_domains_writing_prompt_test.json"),
            ("xsum",           "multi_domains_xsum_test.json"),
            ("yelp_review",    "multi_domains_yelp_review_test.json"),
        ],
        "Multi_LLM": [
            ("ChatGPT",       "multi_llms_ChatGPT_test.json"),
            ("Claude-instant","multi_llms_Claude-instant_test.json"),
            ("Google-PaLM",   "multi_llms_Google-PaLM_test.json"),
            ("Llama-2-70b",   "multi_llms_Llama-2-70b_test.json"),
        ],
        "Multi_Attack": [
            ("direct_prompt",                "Direct_Prompt/direct_prompt_test.json"),
            ("prompt_attacks_llm",           "Prompt_Attacks/prompt_attacks_llm_test.json"),
            ("prompt_few_shot_llm",          "Prompt_Attacks/prompt_few_shot_llm_test.json"),
            ("prompt_ICO_llm",               "Prompt_Attacks/prompt_ICO_llm_test.json"),
            ("paraphrase_attacks_llm",       "Paraphrase_Attacks/paraphrase_attacks_llm_test.json"),
            ("paraphrase_back_translation_llm","Paraphrase_Attacks/paraphrase_back_translation_llm_test.json"),
            ("paraphrase_dipper_llm",        "Paraphrase_Attacks/paraphrase_dipper_llm_test.json"),
            ("paraphrase_polish_llm",        "Paraphrase_Attacks/paraphrase_polish_llm_test.json"),
            ("perturbation_attacks_llm",     "Perturbation_Attacks/perturbation_attacks_llm_test.json"),
            ("perturbation_character_llm",   "Perturbation_Attacks/perturbation_character_llm_test.json"),
            ("perturbation_sent_llm",        "Perturbation_Attacks/perturbation_sent_llm_test.json"),
            ("perturbation_word_llm",        "Perturbation_Attacks/perturbation_word_llm_test.json"),
            ("data_mixing_attacks",          "Data_Mixing/data_mixing_attacks_test.json"),
            ("llm_centered_mixing",          "Data_Mixing/llm_centered_mixing_test.json"),
            ("multi_llm_mixing",             "Data_Mixing/multi_llm_mixing_test.json"),
        ],
    }
    dir_map = {"Multi_Domain": "Multi_Domain", "Multi_LLM": "Multi_LLM", "Multi_Attack": ""}

    for section, entries in subsets.items():
        for label, rel_path in entries:
            if section in ("Multi_Domain", "Multi_LLM"):
                full = os.path.join(args.detectrl_benchmark_dir, section, rel_path)
            else:
                full = os.path.join(args.detectrl_benchmark_dir, rel_path)
            if not os.path.exists(full):
                print(f"[skip] missing {full}")
                continue
            texts, labels = load_detectrl_split(full)
            scores = eval_probs(model, tok, texts, args.batch_size, args.max_length, device)
            m = metrics(np.array(labels), scores)
            results[f"{section}::{label}"] = m
            print(f"{section}::{label} -> AUROC={m['AUROC']:.4f} F1={m['F1']:.4f}")

    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", os.path.join(args.out_dir, "results.json"))


if __name__ == "__main__":
    main()
