"""Extract per-sample, per-layer mean activations from a causal LM on the
COLING-2025 MGT-en dataset.

For every example we record the output of each decoder block, average over the
sequence dimension, and stack into a (num_layers, hidden_size) array. Activations
are written to disk in chunks of `BATCH_SIZE` samples so the job is resumable.

Splits handled (each reads from a local jsonl with `text` and `label` columns):
  * `real_train`, `fake_train`  ->  --train-jsonl  (default: en_train.jsonl)
  * `real_val`,   `fake_val`    ->  --dev-jsonl    (default: en_dev.jsonl)
  * `test`                       ->  --test-jsonl   (default: test_set_en_with_label.jsonl)
"""

import argparse
import os
import re
from glob import glob

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


BATCH_SIZE = 5000  # samples per saved .npy chunk; lower = more frequent checkpoints

CHUNK_RE = re.compile(r"_activations_(\d+)-(\d+)\.npy$")


DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
NP_DTYPES = {"float32": np.float32, "float16": np.float16}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--llm", default="EleutherAI/gpt-neo-2.7B")
    p.add_argument("--dtype", default="float32", choices=list(DTYPES),
                   help="Model weights dtype on GPU. Token averaging happens in fp32 "
                        "regardless; --save-dtype controls the on-disk format.")
    p.add_argument("--save-dtype", default="float32", choices=list(NP_DTYPES),
                   help="Numpy dtype for saved activations. fp16 halves disk usage; "
                        "downstream code casts back to fp32 when reading.")
    p.add_argument("--data-root", default="./coling_data",
                   help="Folder with COLING jsonls (used as default for --*-jsonl)")
    p.add_argument("--train-jsonl", default=None)
    p.add_argument("--dev-jsonl",   default=None)
    p.add_argument("--test-jsonl",  default=None)
    p.add_argument("--out-dir", default=None,
                   help="Output dir; defaults to ./data/<llm>/COLING_2025_MGT_en/")
    p.add_argument("--max-seq-length", type=int, default=4096)
    p.add_argument("--splits", nargs="+",
                   default=["real_train", "fake_train", "real_val", "fake_val", "test"],
                   help="Which splits to compute. Useful for resuming or parallel jobs.")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    return p.parse_args()


def load_model(llm_name: str, hf_token: str | None, dtype: str = "float32"):
    tokenizer = AutoTokenizer.from_pretrained(llm_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        llm_name, token=hf_token, torch_dtype=DTYPES[dtype],
    )
    model.to("cuda")
    model.eval()
    return model, tokenizer


def get_decoder_blocks(model: torch.nn.Module) -> list[torch.nn.Module]:
    """Return the list of decoder blocks for the most common HF causal LMs."""
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)               # GPT-Neo, GPT-2, GPT-J
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)                # Llama, Qwen, Gemma
    raise RuntimeError(f"Unknown decoder layout for {type(model).__name__}")


class ActivationRecorder:
    """Forward hooks that capture per-layer mean-over-tokens activations.

    Replaces `steering_vectors.record_activations` so we don't need that pip dep.
    """

    def __init__(self, blocks: list[torch.nn.Module]):
        self.blocks = blocks
        self.handles: list = []
        self.layer_means: list = []

    def __enter__(self):
        self.layer_means = [None] * len(self.blocks)

        def make_hook(idx: int):
            def hook(_module, _input, output):
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
        text,
        return_tensors="pt",
        max_length=max_seq_length,
        truncation=True,
        padding=False,
    ).to("cuda")


def existing_resume_offset(out_dir: str, prefix: str) -> int:
    """Resume offset = max end-of-range across already-saved chunks for this prefix.

    Tolerates mixed chunk sizes (e.g. older 20k-row chunks alongside newer 5k-row ones)
    by reading actual filename ranges instead of assuming a fixed BATCH_SIZE.
    """
    offset = 0
    for path in glob(os.path.join(out_dir, f"{prefix}_activations_*.npy")):
        m = CHUNK_RE.search(os.path.basename(path))
        if not m:
            continue
        end = int(m.group(2))
        if end > offset:
            offset = end
    return offset


def get_split_activations(texts, prefix: str, out_dir: str,
                           model, tokenizer, recorder_blocks, max_seq_length: int,
                           save_dtype: str = "float32"):
    """Compute activations for `texts`, saving in BATCH_SIZE chunks. Resumable."""
    start = existing_resume_offset(out_dir, prefix)
    if start >= len(texts):
        print(f"[{prefix}] already complete ({len(texts)} samples)")
        return
    if start > 0:
        print(f"[{prefix}] resuming from sample {start}")

    np_dtype = NP_DTYPES[save_dtype]
    samples: list = []
    chunk_start = start
    for text in tqdm(texts[start:], desc=prefix, initial=start, total=len(texts)):
        with ActivationRecorder(recorder_blocks) as rec:
            inputs = encode_one(tokenizer, text, max_seq_length)
            with torch.no_grad():
                model(**inputs)
            samples.append(rec.stack())

        if len(samples) == BATCH_SIZE:
            chunk_end = chunk_start + len(samples)
            path = f"{out_dir}/{prefix}_activations_{chunk_start}-{chunk_end}.npy"
            np.save(path, np.array(samples, dtype=np_dtype))
            print(f"saved {path}")
            samples = []
            chunk_start = chunk_end

    if samples:
        chunk_end = chunk_start + len(samples)
        path = f"{out_dir}/{prefix}_activations_{chunk_start}-{chunk_end}.npy"
        np.save(path, np.array(samples, dtype=np_dtype))
        print(f"saved {path}")


def load_jsonl(path: str) -> pd.DataFrame:
    print(f"loading {path}")
    return pd.read_json(path, lines=True)


def main():
    args = parse_args()
    out_dir = args.out_dir or (
        f"./data/{args.llm.split('/')[-1]}/COLING_2025_MGT_en"
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output dir: {out_dir}")

    train_path = args.train_jsonl or os.path.join(args.data_root, "en_train.jsonl")
    dev_path   = args.dev_jsonl   or os.path.join(args.data_root, "en_dev.jsonl")
    test_path  = args.test_jsonl  or os.path.join(args.data_root, "test_set_en_with_label.jsonl")

    print(f"Loading {args.llm} (dtype={args.dtype}, save_dtype={args.save_dtype})")
    model, tokenizer = load_model(args.llm, args.hf_token, dtype=args.dtype)
    blocks = get_decoder_blocks(model)
    print(f"num_layers={len(blocks)} hidden_size={model.config.hidden_size}")

    # GPT-Neo and other absolute-positional-embedding models have a hard cap;
    # exceeding it triggers a CUDA index-out-of-bounds in the position-embed lookup.
    model_cap = getattr(model.config, "max_position_embeddings", None)
    if model_cap is not None and args.max_seq_length > model_cap:
        print(f"Capping max_seq_length {args.max_seq_length} -> {model_cap} (model limit)")
        args.max_seq_length = model_cap

    split_texts: dict = {}
    if "real_train" in args.splits or "fake_train" in args.splits:
        train_df = load_jsonl(train_path)
        split_texts["real_train"] = train_df.loc[train_df["label"] == 0, "text"].tolist()
        split_texts["fake_train"] = train_df.loc[train_df["label"] == 1, "text"].tolist()
        print(f"train: real={len(split_texts.get('real_train',[]))} "
              f"fake={len(split_texts.get('fake_train',[]))}")
    if "real_val" in args.splits or "fake_val" in args.splits:
        dev_df = load_jsonl(dev_path)
        split_texts["real_val"] = dev_df.loc[dev_df["label"] == 0, "text"].tolist()
        split_texts["fake_val"] = dev_df.loc[dev_df["label"] == 1, "text"].tolist()
        print(f"dev: real={len(split_texts.get('real_val',[]))} "
              f"fake={len(split_texts.get('fake_val',[]))}")
    if "test" in args.splits:
        test_df = load_jsonl(test_path)
        split_texts["test"] = test_df["text"].tolist()
        print(f"test: {len(split_texts['test'])} (labels saved in jsonl for logreg eval)")

    for prefix in args.splits:
        if prefix not in split_texts:
            continue
        print(f"--- {prefix}: {len(split_texts[prefix])} samples ---")
        get_split_activations(
            split_texts[prefix], prefix, out_dir,
            model, tokenizer, blocks, args.max_seq_length,
            save_dtype=args.save_dtype,
        )


if __name__ == "__main__":
    main()
