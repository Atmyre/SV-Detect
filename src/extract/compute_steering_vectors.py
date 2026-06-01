"""Compute steering vectors from saved activations and project all splits.

Inputs (read from --data-dir, layout produced by extract_activations.py):
    real_train_activations_*.npy, fake_train_activations_*.npy,
    real_val_activations_*.npy,   fake_val_activations_*.npy,
    test_activations_*.npy
Each file has shape (n_samples, num_layers, hidden_size).

Outputs (written to --data-dir, file_tag = "<method><suffix>"):
    steering_vectors_<file_tag>.npy                (num_layers, hidden_size)
    {real,fake}_{train,val}_dot_products_<file_tag>.npy   (n_samples, num_layers)
    test_dot_products_<file_tag>.npy                       (n_samples, num_layers)

Methods (--method, repeatable):
    mean    : (mean(fake) - mean(real)) / ||·||  per layer  [streamed, no full load]
    pca     : top-1 PC of paired (fake - real) differences per layer
    logreg  : LogisticRegression(C=0.01, solver=liblinear, fit_intercept=False)
              fit on centered real+fake; normalized coef as SV direction

Optional `--exclude-models` or `--keep-models` masks training samples by their
`model` column when computing the steering vector. Activations were saved in
jsonl row order within each class, so we read --train-jsonl, build per-class
boolean masks, and feed them to the loaders. Dot products are computed for
every sample regardless of the filter.
"""

import argparse
import os
import re
from glob import glob

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.utils.extmath import randomized_svd
from tqdm import tqdm


CHUNK_RE = re.compile(r"_activations_(\d+)-(\d+)\.npy$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--method", nargs="+", default=["mean"],
                   choices=["mean", "pca", "logreg"],
                   help="Which steering-vector method(s) to compute. Repeatable.")
    p.add_argument("--exclude-models", nargs="*", default=[],
                   help="Skip train samples whose `model` is in this list. "
                        "Mutually exclusive with --keep-models.")
    p.add_argument("--keep-models", nargs="*", default=[],
                   help="Only keep train samples whose `model` is in this list.")
    p.add_argument("--train-jsonl", default=None,
                   help="Required if --exclude-models or --keep-models is set.")
    p.add_argument("--out-suffix", default="",
                   help="Append to file_tag after the method (e.g. '_woweak').")
    p.add_argument("--logreg-c", type=float, default=0.01,
                   help="C for the logreg method (matches original 0.01).")
    p.add_argument("--logreg-max-samples", type=int, default=50000,
                   help="Subsample this many samples per class before fitting logreg. "
                        "At C=0.01 the SV direction stabilizes well below the full dataset, "
                        "and liblinear scales poorly to >100k×2560+. 0 = use full data.")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip a method if its steering_vectors_<file_tag>.npy already exists.")
    return p.parse_args()


def chunk_files(data_dir: str, prefix: str) -> list:
    paths = glob(os.path.join(data_dir, f"{prefix}_activations_*.npy"))

    def offset(p: str) -> int:
        m = CHUNK_RE.search(os.path.basename(p))
        if not m:
            raise ValueError(f"unrecognized chunk filename: {p}")
        return int(m.group(1))

    return sorted(paths, key=offset)


def streaming_mean(data_dir: str, prefix: str, mask: np.ndarray | None = None) -> np.ndarray:
    """Mean over samples, streaming chunk-by-chunk. Returns (num_layers, hidden_size).

    Chunks may be saved as fp16; we accumulate in fp32 to avoid precision loss on
    sums of hundreds of thousands of samples."""
    files = chunk_files(data_dir, prefix)
    if not files:
        raise FileNotFoundError(f"No {prefix}_activations_*.npy in {data_dir}")
    total = 0
    running = None
    cursor = 0
    for f in files:
        arr = np.load(f)
        n = arr.shape[0]
        if mask is not None:
            sub = mask[cursor:cursor + n]
            arr = arr[sub]
            cursor += n
        if arr.shape[0] == 0:
            continue
        s = arr.astype(np.float32, copy=False).sum(axis=0)
        running = s if running is None else running + s
        total += arr.shape[0]
    if total == 0:
        raise RuntimeError(f"mask kept zero samples for prefix {prefix}")
    return running / total


def load_all_train(data_dir: str, prefix: str, mask: np.ndarray | None = None) -> np.ndarray:
    """Concat all chunks for one prefix, optionally masked. Returns (N, L, H)."""
    files = chunk_files(data_dir, prefix)
    if not files:
        raise FileNotFoundError(f"No {prefix}_activations_*.npy in {data_dir}")
    parts = []
    cursor = 0
    for f in tqdm(files, desc=f"load {prefix}"):
        arr = np.load(f)
        n = arr.shape[0]
        if mask is not None:
            arr = arr[mask[cursor:cursor + n]]
            cursor += n
        parts.append(arr)
    return np.concatenate(parts, axis=0)


def compute_sv_mean(data_dir: str,
                    real_mask: np.ndarray | None,
                    fake_mask: np.ndarray | None) -> np.ndarray:
    real_mean = streaming_mean(data_dir, "real_train", mask=real_mask)
    fake_mean = streaming_mean(data_dir, "fake_train", mask=fake_mask)
    sv = fake_mean - real_mean
    return sv / np.linalg.norm(sv, axis=-1, keepdims=True)


def compute_sv_pca(real_full: np.ndarray, fake_full: np.ndarray) -> np.ndarray:
    """Top-1 PC of paired (fake - real) differences per layer.
    Pairs are zipped in order, truncated to min(N_real, N_fake).

    Uses randomized_svd instead of IncrementalPCA: top-1 only, O(N*H) instead of
    O(N*H^2). On Llama-2 (H=4096) this drops per-layer PCA from ~10 min to ~2 s."""
    n = min(len(real_full), len(fake_full))
    L = real_full.shape[1]
    out = np.zeros((L, real_full.shape[2]), dtype=np.float32)
    for layer in tqdm(range(L), desc="pca"):
        diffs = (fake_full[:n, layer, :] - real_full[:n, layer, :]).astype(np.float32)
        diffs -= diffs.mean(axis=0)  # center, matches PCA
        # rank-1 randomized SVD; right singular vector is the top PC direction.
        _, _, vt = randomized_svd(diffs, n_components=1, random_state=42)
        v = vt[0]
        out[layer] = v / np.linalg.norm(v)
    return out


def compute_sv_logreg(real_full: np.ndarray, fake_full: np.ndarray, C: float,
                      max_per_class: int = 0) -> np.ndarray:
    """Per layer: fit LogReg on centered real+fake, return normalized coef.

    `max_per_class > 0` subsamples each class to that many rows before fitting.
    Subsampling indices are drawn once (not per-layer) so the SV directions across
    layers see the same documents."""
    rng = np.random.default_rng(42)
    n_r, n_f = len(real_full), len(fake_full)
    if max_per_class > 0:
        n_r_keep = min(n_r, max_per_class)
        n_f_keep = min(n_f, max_per_class)
        real_idx = rng.choice(n_r, n_r_keep, replace=False)
        fake_idx = rng.choice(n_f, n_f_keep, replace=False)
    else:
        real_idx = slice(None)
        fake_idx = slice(None)
        n_r_keep, n_f_keep = n_r, n_f

    print(f"logreg: using {n_r_keep}+{n_f_keep} = {n_r_keep + n_f_keep} samples "
          f"(of {n_r}+{n_f})")

    L = real_full.shape[1]
    out = np.zeros((L, real_full.shape[2]), dtype=np.float32)
    y = np.concatenate([np.zeros(n_r_keep, dtype=np.int8), np.ones(n_f_keep, dtype=np.int8)])
    for layer in tqdm(range(L), desc="logreg"):
        real_l = real_full[real_idx, layer, :].astype(np.float32, copy=False)
        fake_l = fake_full[fake_idx, layer, :].astype(np.float32, copy=False)
        joint_mean = (real_l.sum(0) + fake_l.sum(0)) / (n_r_keep + n_f_keep)
        X = np.concatenate([real_l - joint_mean, fake_l - joint_mean])
        clf = LogisticRegression(
            solver="liblinear", C=C, fit_intercept=False, random_state=42,
        )
        clf.fit(X, y)
        v = clf.coef_[0]
        out[layer] = v / np.linalg.norm(v)
    return out


def get_dot_products(activations: np.ndarray, steering_vectors: np.ndarray) -> np.ndarray:
    """activations (N, L, H), steering_vectors (L, H) → (N, L). Activations are
    L2-normalized per (sample, layer) before dotting. fp16 chunks are upcast to
    fp32 first so the norms don't lose precision on long Llama-style hidden dims."""
    activations = activations.astype(np.float32, copy=False)
    norms = np.linalg.norm(activations, axis=-1, keepdims=True)
    normed = activations / np.clip(norms, 1e-12, None)
    return np.sum(normed * steering_vectors, axis=-1)


def project_split(data_dir: str, prefix: str, steering_vectors: np.ndarray, file_tag: str):
    files = chunk_files(data_dir, prefix)
    if not files:
        print(f"[skip] no files for prefix {prefix}")
        return
    parts = []
    for f in tqdm(files, desc=f"dot {prefix} {file_tag}"):
        parts.append(get_dot_products(np.load(f), steering_vectors))
    out = np.concatenate(parts, axis=0)
    name = f"{prefix}_dot_products_{file_tag}.npy"
    np.save(os.path.join(data_dir, name), out)
    print(f"saved {name} shape={out.shape}")


def build_masks(train_jsonl: str,
                exclude_models: list,
                keep_models: list) -> tuple[np.ndarray, np.ndarray]:
    print(f"loading train jsonl for model filter: {train_jsonl}")
    df = pd.read_json(train_jsonl, lines=True)
    real_models = df.loc[df["label"] == 0, "model"]
    fake_models = df.loc[df["label"] == 1, "model"]
    if keep_models:
        keep = set(keep_models)
        real_keep = real_models.isin(keep).to_numpy()
        fake_keep = fake_models.isin(keep).to_numpy()
    else:
        excl = set(exclude_models)
        real_keep = (~real_models.isin(excl)).to_numpy()
        fake_keep = (~fake_models.isin(excl)).to_numpy()
    print(f"real rows: kept {real_keep.sum()}/{len(real_keep)}")
    print(f"fake rows: kept {fake_keep.sum()}/{len(fake_keep)}")
    dropped_real = real_models[~real_keep].value_counts()
    dropped_fake = fake_models[~fake_keep].value_counts()
    if len(dropped_real):
        print(f"dropped real models: {dict(dropped_real)}")
    if len(dropped_fake):
        print(f"dropped fake models: {dict(dropped_fake)}")
    return real_keep, fake_keep


def main():
    args = parse_args()
    if args.exclude_models and args.keep_models:
        raise SystemExit("--exclude-models and --keep-models are mutually exclusive")

    real_mask = fake_mask = None
    if args.exclude_models or args.keep_models:
        if not args.train_jsonl:
            raise SystemExit("--exclude-models / --keep-models requires --train-jsonl")
        real_mask, fake_mask = build_masks(
            args.train_jsonl, args.exclude_models, args.keep_models,
        )

    methods = list(args.method)
    needs_full = any(m in {"pca", "logreg"} for m in methods)
    real_full = fake_full = None
    if needs_full:
        print("Loading full train activations into memory...")
        real_full = load_all_train(args.data_dir, "real_train", mask=real_mask)
        fake_full = load_all_train(args.data_dir, "fake_train", mask=fake_mask)
        print(f"real_full: {real_full.shape}; fake_full: {fake_full.shape}; "
              f"~{(real_full.nbytes + fake_full.nbytes) / 1e9:.1f} GB")

    for method in methods:
        file_tag = f"{method}{args.out_suffix}"
        sv_path = os.path.join(args.data_dir, f"steering_vectors_{file_tag}.npy")
        if args.skip_existing and os.path.exists(sv_path):
            print(f"\n=== method={method}  file_tag={file_tag} (SKIP — SV already on disk) ===")
            sv = np.load(sv_path)
        else:
            print(f"\n=== method={method}  file_tag={file_tag} ===")
            if method == "mean":
                sv = compute_sv_mean(args.data_dir, real_mask, fake_mask)
            elif method == "pca":
                sv = compute_sv_pca(real_full, fake_full)
            elif method == "logreg":
                sv = compute_sv_logreg(
                    real_full, fake_full,
                    C=args.logreg_c, max_per_class=args.logreg_max_samples,
                )
            else:
                raise AssertionError(method)
            np.save(sv_path, sv)
            print(f"saved {sv_path} shape={sv.shape}")

        for prefix in ["real_train", "fake_train", "real_val", "fake_val", "test"]:
            dp_path = os.path.join(args.data_dir, f"{prefix}_dot_products_{file_tag}.npy")
            if args.skip_existing and os.path.exists(dp_path):
                print(f"[skip] {os.path.basename(dp_path)} already on disk")
                continue
            project_split(args.data_dir, prefix, sv, file_tag)


if __name__ == "__main__":
    main()
