"""End-to-end port of the DetectRL + MIRAGE notebooks to a cluster-friendly
Python module. Everything matches the notebook logic 1:1:
  * frozen GPT-Neo-2.7B, fp16, mean-pooled decoder-block activations,
    max_seq_length=2048
  * per-subset (DetectRL) or per-task (MIRAGE) steering vectors
  * mean-difference / logreg / PCA construction
  * cosine-similarity dot products
  * StandardScaler + LogisticRegression(C=1, solver=liblinear, penalty=l2) head

Adds cluster-only affordances the notebook lacked:
  * chunked, resumable activation extraction
  * CLI subcommands so slurm arrays can parallelise per subset
"""

import argparse
import gc
import json
import os
import pickle
import re
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (auc, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, precision_recall_curve,
                             roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

LLM_NAME_DEFAULT = "EleutherAI/gpt-neo-2.7B"
NUM_LAYERS_DEFAULT = 32
HIDDEN_SIZE_DEFAULT = 2560
MAX_SEQ_LENGTH_DEFAULT = 2048

CHUNK_SIZE = 500  # samples per resumable .npy chunk
CHUNK_RE = re.compile(r"_activations_(\d+)-(\d+)\.npy$")


# --------------------------------------------------------------------------- #
# Model + activation extraction
# --------------------------------------------------------------------------- #

def load_model(llm_name: str = LLM_NAME_DEFAULT, hf_token: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(llm_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        llm_name, token=hf_token, dtype=torch.float16, device_map="cuda"
    )
    model.eval()
    return model, tokenizer


def get_decoder_blocks(model: torch.nn.Module):
    """Match `record_activations(..., layer_type='decoder_block')` from the notebook.
    Works for GPT-Neo/GPT-2/GPT-J (`model.transformer.h`) and Llama-style
    (`model.model.layers`)."""
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    raise RuntimeError(f"Unknown decoder layout for {type(model).__name__}")


class ActivationRecorder:
    """Per-layer mean-over-tokens hidden state, in fp32."""

    def __init__(self, blocks):
        self.blocks = blocks
        self.layer_means = []
        self.handles = []

    def __enter__(self):
        self.layer_means = [None] * len(self.blocks)

        def make_hook(idx):
            def hook(_m, _i, output):
                hidden = output[0] if isinstance(output, tuple) else output
                self.layer_means[idx] = hidden.mean(dim=1).flatten().to(torch.float32)
            return hook

        for i, block in enumerate(self.blocks):
            self.handles.append(block.register_forward_hook(make_hook(i)))
        return self

    def __exit__(self, *_exc):
        for h in self.handles:
            h.remove()
        self.handles = []

    def stack(self) -> np.ndarray:
        return torch.stack(self.layer_means).cpu().numpy()


def encode_one(tokenizer, text: str, max_seq_length: int):
    return tokenizer(
        text, return_tensors="pt", max_length=max_seq_length,
        truncation=True, padding=False,
    ).to("cuda")


def _existing_offset(out_dir: str, prefix: str) -> int:
    offset = 0
    for path in glob(os.path.join(out_dir, f"{prefix}_activations_*.npy")):
        m = CHUNK_RE.search(os.path.basename(path))
        if m and int(m.group(2)) > offset:
            offset = int(m.group(2))
    return offset


def extract_split(texts, prefix: str, out_dir: str, model, tokenizer, blocks,
                  max_seq_length: int):
    """Extract mean-pooled activations for one list of strings, chunked/resumable.

    Output shape: (N, num_layers, hidden_size). Files:
        {out_dir}/{prefix}_activations_{start}-{end}.npy
    """
    os.makedirs(out_dir, exist_ok=True)
    start = _existing_offset(out_dir, prefix)
    if start >= len(texts):
        print(f"[{prefix}] already complete ({len(texts)} samples)")
        return
    if start > 0:
        print(f"[{prefix}] resuming from {start}")

    samples = []
    chunk_start = start
    for text in tqdm(texts[start:], desc=prefix, initial=start, total=len(texts)):
        with ActivationRecorder(blocks) as rec:
            inputs = encode_one(tokenizer, text, max_seq_length)
            with torch.no_grad():
                model(**inputs)
            samples.append(rec.stack())

        if len(samples) == CHUNK_SIZE:
            chunk_end = chunk_start + len(samples)
            path = os.path.join(out_dir, f"{prefix}_activations_{chunk_start}-{chunk_end}.npy")
            np.save(path, np.array(samples, dtype=np.float32))
            print(f"saved {path}")
            samples = []
            chunk_start = chunk_end

    if samples:
        chunk_end = chunk_start + len(samples)
        path = os.path.join(out_dir, f"{prefix}_activations_{chunk_start}-{chunk_end}.npy")
        np.save(path, np.array(samples, dtype=np.float32))
        print(f"saved {path}")


def load_all_chunks(out_dir: str, prefix: str) -> np.ndarray:
    files = sorted(
        glob(os.path.join(out_dir, f"{prefix}_activations_*.npy")),
        key=lambda p: int(CHUNK_RE.search(os.path.basename(p)).group(1)),
    )
    if not files:
        raise FileNotFoundError(f"No chunks for {prefix} in {out_dir}")
    return np.concatenate([np.load(f) for f in files], axis=0)


# --------------------------------------------------------------------------- #
# Steering-vector construction (mirrors notebook exactly)
# --------------------------------------------------------------------------- #

def sv_mean(fake: np.ndarray, real: np.ndarray) -> np.ndarray:
    """(L, H) direction: mean(fake) - mean(real), row-normalized."""
    v = fake.astype(np.float32).mean(0) - real.astype(np.float32).mean(0)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _sv_logreg_layer(fake_l: np.ndarray, real_l: np.ndarray) -> np.ndarray:
    mean_acts = np.stack([fake_l, real_l]).mean(axis=0)
    Xf = fake_l - mean_acts
    Xr = real_l - mean_acts
    X = np.concatenate([Xf, Xr], axis=0)
    y = np.concatenate([np.ones(len(Xf), np.int8), np.zeros(len(Xr), np.int8)])
    clf = LogisticRegression(
        fit_intercept=False, l1_ratio=0.0, solver="liblinear",
        random_state=42, C=1.0, max_iter=100,
    ).fit(X, y)
    v = clf.coef_.squeeze(0).astype(np.float32)
    return v / np.linalg.norm(v)


def _sv_pca_layer(fake_l: np.ndarray, real_l: np.ndarray) -> np.ndarray:
    diff = fake_l - real_l
    pca = PCA(n_components=1, random_state=42).fit(diff)
    return pca.components_[0].astype(np.float32)


def sv_layerwise(fake: np.ndarray, real: np.ndarray, method: str) -> np.ndarray:
    """(L, H) per-layer steering vector via mean / logreg / pca."""
    fake_l = fake.transpose(1, 0, 2).astype(np.float32)
    real_l = real.transpose(1, 0, 2).astype(np.float32)
    L = fake_l.shape[0]
    out = np.zeros((L, fake_l.shape[2]), dtype=np.float32)
    for l in tqdm(range(L), desc=f"SV[{method}]"):
        if method == "logreg":
            out[l] = _sv_logreg_layer(fake_l[l], real_l[l])
        elif method == "pca":
            out[l] = _sv_pca_layer(fake_l[l], real_l[l])
        else:
            raise ValueError(method)
    return out


def compute_sv(fake_acts: np.ndarray, real_acts: np.ndarray, method: str) -> np.ndarray:
    if method == "mean":
        return sv_mean(fake_acts, real_acts)
    return sv_layerwise(fake_acts, real_acts, method)


def qr_orthonormalize(sv_list: list[np.ndarray]) -> np.ndarray:
    """Given K per-task SVs each (L, H), stack to (L, H, K) and orthonormalize per layer.
    Returns (L, H, K)."""
    stacked = np.stack(sv_list).transpose(1, 2, 0)  # (L, H, K)
    Q, _ = np.linalg.qr(stacked)
    return Q


# --------------------------------------------------------------------------- #
# Dot products (cosine)
# --------------------------------------------------------------------------- #

def dot_products(activations: np.ndarray, sv: np.ndarray) -> np.ndarray:
    """activations: (N, L, H). sv: (L, H) OR (L, H, K).
    Returns (N, L) if single-direction SV, (N, L*K) if K-direction SV (K-fastest)."""
    N, L, H = activations.shape
    acts_f = activations.astype(np.float32)
    # per-sample per-layer L2-normalize activation, then dot with (already unit) SV
    norms = np.linalg.norm(acts_f, axis=-1, keepdims=True)
    acts_n = acts_f / norms
    if sv.ndim == 2:
        # (L, H) -> broadcast to (1, L, H); output (N, L)
        return np.einsum("nlh,lh->nl", acts_n, sv.astype(np.float32))
    elif sv.ndim == 3:
        # (L, H, K) -> output (N, L, K) -> flatten to (N, L*K), K-fastest to match notebook
        out = np.einsum("nlh,lhk->nlk", acts_n, sv.astype(np.float32))
        return out.reshape(N, L * sv.shape[-1])
    raise ValueError(f"sv.ndim={sv.ndim}")


# --------------------------------------------------------------------------- #
# Classifier + metrics (mirror notebook exactly)
# --------------------------------------------------------------------------- #

def build_detector() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            l1_ratio=0.0, solver="liblinear", random_state=42,
            C=1.0, max_iter=100,
        )),
    ])


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
        "AUROC": auc(fpr, tpr),
        "AUPR":  auc(rec, prec),
        "TPR@FPR=5%": _tpr_at_fpr5(fpr, tpr),
        "Balanced Accuracy": float(np.max((tpr + 1 - fpr) / 2)),
        "MCC": float(np.max([matthews_corrcoef(labels, scores > t) for t in thresholds])),
        "F1": float(np.max(f1_arr)),
    }


# --------------------------------------------------------------------------- #
# High-level driver used both from CLI and from evaluation scripts
# --------------------------------------------------------------------------- #

def load_texts_detectrl(json_path: str, kind: str) -> list[str]:
    """DetectRL JSON is a list of {'text','label',...}. kind ∈ {'human','llm'}."""
    with open(json_path) as fp:
        rows = json.load(fp)
    return [r["text"] for r in rows if r["label"] == kind]


def load_texts_mirage(json_path: str, kind: str) -> list[str]:
    """MIRAGE test JSON is a list of paired {'original','rewritten',...}.
    kind ∈ {'human','llm'}."""
    with open(json_path) as fp:
        rows = json.load(fp)
    key = "original" if kind == "human" else "rewritten"
    return [r[key] for r in rows]


def load_texts_mirage_train(json_path: str, kind: str) -> list[str]:
    """MIRAGE train JSON is a dict {'original': [...], 'rewritten': [...]}."""
    with open(json_path) as fp:
        d = json.load(fp)
    return d["original"] if kind == "human" else d["rewritten"]


def load_texts_repreguard(json_path: str, kind: str) -> list[str]:
    """RepreGuard JSON is a list of paired records with fields 'human_text' + 'direct_prompt'.
    kind ∈ {'human', 'llm'}.
    """
    with open(json_path) as fp:
        rows = json.load(fp)
    key = "human_text" if kind == "human" else "direct_prompt"
    return [r[key] for r in rows]


def load_texts_padben(json_path: str, kind: str) -> list[str]:
    """PADBen single-sentence JSON: list of {idx, sentence, label∈{0,1}}.
    kind ∈ {'label0', 'label1'} — extracts the "human-ish" (0) or "LLM-ish" (1) subset,
    preserving the original in-file order so it can be re-joined with idx later.

    Note: label semantics vary by task per the PADBen paper (Fig 2). Task 1: 0=human-paraphr,
    1=LLM-paraphr; Task 2: 0=human orig, 1=LLM gen; Task 5: 0=human, 1=deep-paraphr-LLM.
    We treat them as generic binary — the SV-Detect classifier applies the same way.
    """
    with open(json_path) as fp:
        rows = json.load(fp)
    target = 0 if kind == "label0" else 1
    return [r["sentence"] for r in rows if r["label"] == target]


def load_texts_detectrl_subsampled(json_path: str, kind: str, task: str, seed: int = 42
                                    ) -> list[str]:
    """Notebook-fidelity subsampling for DetectRL *train* fake data:
      * multi_domains  -> 50 per (llm_type, data_type)
      * multi_llms     -> 200 per (llm_type, data_type)
      * multi_attacks  -> 10192 total
    Real data + all test data: no subsampling.
    """
    with open(json_path) as fp:
        rows = json.load(fp)
    if kind == "human":
        return [r["text"] for r in rows if r["label"] == "human"]

    llm_rows = [r for r in rows if r["label"] == "llm"]
    if task == "multi_attacks":
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(llm_rows))[:10192]
        return [llm_rows[i]["text"] for i in idx]

    per_cell = {"multi_domains": 50, "multi_llms": 200}[task]
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in llm_rows:
        buckets[(r["llm_type"], r["data_type"])].append(r["text"])
    rng = np.random.default_rng(seed)
    out = []
    for k in sorted(buckets):
        b = buckets[k]
        idx = rng.permutation(len(b))[: min(per_cell, len(b))]
        out.extend([b[i] for i in idx])
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _cmd_extract(args):
    """Extract activations for one (input, kind) → one prefix."""
    kind = args.kind  # 'human' or 'llm'
    loader = {
        "detectrl":       load_texts_detectrl,
        "detectrl_train": lambda p, k: load_texts_detectrl_subsampled(p, k, args.task, args.seed),
        "mirage":         load_texts_mirage,
        "mirage_train":   load_texts_mirage_train,
        "repreguard":     load_texts_repreguard,
        "padben":         load_texts_padben,
    }[args.loader]
    texts = loader(args.input, kind)
    print(f"[{args.prefix}] {len(texts)} texts")
    model, tok = load_model(args.llm, hf_token=os.environ.get("HF_TOKEN"))
    blocks = get_decoder_blocks(model)
    extract_split(texts, args.prefix, args.out_dir, model, tok, blocks,
                  args.max_seq_length)


def _cmd_sv(args):
    """Fit one per-subset SV from real/fake activations (mean/logreg/pca)."""
    real = load_all_chunks(args.acts_dir, args.real_prefix)
    fake = load_all_chunks(args.acts_dir, args.fake_prefix)
    print(f"real {real.shape} fake {fake.shape}")
    for m in args.methods:
        v = compute_sv(fake, real, m)
        os.makedirs(args.out_dir, exist_ok=True)
        out = os.path.join(args.out_dir, f"{args.subset}_steering_vectors_{m}.npy")
        np.save(out, v)
        print(f"saved {out}  shape={v.shape}")


def _cmd_orthonormal(args):
    """Combine per-task SVs into a QR-orthonormal (L, H, K) system (MIRAGE)."""
    for m in args.methods:
        svs = [np.load(os.path.join(args.sv_dir, f"{t}_steering_vectors_{m}.npy"))
               for t in args.tasks]
        Q = qr_orthonormalize(svs)
        out = os.path.join(args.sv_dir, f"orthonormal_steering_vectors_{m}.npy")
        np.save(out, Q)
        print(f"saved {out}  shape={Q.shape}")


def _cmd_dots(args):
    """Project activations onto (one, or K-dim) SV → cosine features."""
    acts = load_all_chunks(args.acts_dir, args.acts_prefix)
    sv = np.load(args.sv_path)
    d = dot_products(acts, sv)
    os.makedirs(args.out_dir, exist_ok=True)
    np.save(args.out_path, d)
    print(f"saved {args.out_path}  shape={d.shape}")


def _cmd_train(args):
    """Train the StandardScaler + LogReg detector on {train_real, train_fake} dots."""
    real = np.load(args.train_real)
    fake = np.load(args.train_fake)
    X = np.concatenate([real, fake], axis=0)
    y = np.concatenate([np.zeros(len(real), np.int8), np.ones(len(fake), np.int8)])
    clf = build_detector().fit(X, y)
    os.makedirs(os.path.dirname(args.out_pkl), exist_ok=True)
    with open(args.out_pkl, "wb") as f:
        pickle.dump(clf, f, protocol=5)
    print(f"saved {args.out_pkl}")


def _cmd_evaluate(args):
    """Score {test_real, test_fake} dots with a trained detector, dump metrics row."""
    with open(args.clf) as f:
        clf = pickle.load(open(args.clf, "rb"))
    real = np.load(args.test_real); fake = np.load(args.test_fake)
    X = np.concatenate([real, fake], axis=0)
    y = np.concatenate([np.zeros(len(real), np.int8), np.ones(len(fake), np.int8)])
    probs = clf.predict_proba(X)[:, 1]
    m = metrics(y, probs)
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({"label": args.label, **m}, f, indent=2)
    print(json.dumps({"label": args.label, **m}, indent=2))


def build_argparser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract")
    e.add_argument("--llm", default=LLM_NAME_DEFAULT)
    e.add_argument("--input", required=True, help="Path to JSON (or dict-JSON for mirage_train)")
    e.add_argument("--loader", required=True,
                   choices=["detectrl", "detectrl_train", "mirage", "mirage_train", "repreguard", "padben"])
    e.add_argument("--kind", required=True, choices=["human", "llm", "label0", "label1"])
    e.add_argument("--prefix", required=True, help="Filename prefix, e.g. arxiv_train_real")
    e.add_argument("--out-dir", required=True)
    e.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH_DEFAULT)
    e.add_argument("--task", default=None,
                   help="Only for --loader detectrl_train: multi_domains / multi_llms / multi_attacks")
    e.add_argument("--seed", type=int, default=42)
    e.set_defaults(func=_cmd_extract)

    s = sub.add_parser("sv")
    s.add_argument("--acts-dir", required=True)
    s.add_argument("--real-prefix", required=True)
    s.add_argument("--fake-prefix", required=True)
    s.add_argument("--methods", nargs="+", default=["logreg"], choices=["mean", "logreg", "pca"])
    s.add_argument("--subset", required=True)
    s.add_argument("--out-dir", required=True)
    s.set_defaults(func=_cmd_sv)

    o = sub.add_parser("orthonormal")
    o.add_argument("--sv-dir", required=True)
    o.add_argument("--tasks", nargs="+", required=True)
    o.add_argument("--methods", nargs="+", default=["logreg"])
    o.set_defaults(func=_cmd_orthonormal)

    d = sub.add_parser("dots")
    d.add_argument("--acts-dir", required=True)
    d.add_argument("--acts-prefix", required=True)
    d.add_argument("--sv-path", required=True)
    d.add_argument("--out-dir", required=True)
    d.add_argument("--out-path", required=True)
    d.set_defaults(func=_cmd_dots)

    t = sub.add_parser("train")
    t.add_argument("--train-real", required=True)
    t.add_argument("--train-fake", required=True)
    t.add_argument("--out-pkl", required=True)
    t.set_defaults(func=_cmd_train)

    v = sub.add_parser("evaluate")
    v.add_argument("--clf", required=True)
    v.add_argument("--test-real", required=True)
    v.add_argument("--test-fake", required=True)
    v.add_argument("--out-json", required=True)
    v.add_argument("--label", default="")
    v.set_defaults(func=_cmd_evaluate)

    return p


def main():
    args = build_argparser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
