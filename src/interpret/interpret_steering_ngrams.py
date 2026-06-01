"""Corpus-scan interpretation: for each layer, find token positions in real
texts where the residual stream is most aligned with (or anti-aligned with) the
steering vector, and return the surrounding token window. This reveals
multi-token patterns (e.g. `<td>`, `<ref>`, `<\\div>`) that single-token
logit-lens projections miss.

For each (text, position t, layer L):
    cosine_t = (residual[t, L] / ||residual[t, L]||) @ sv[L]

We keep the global top-K positive and top-K negative across all (text, position)
pairs, then decode a window of `--context-tokens` tokens ending at position t.

Output: per layer, two lists (top + and top -) of context windows with cosines.
"""

import argparse
import heapq
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--llm", required=True)
    p.add_argument("--sv-path", required=True)
    p.add_argument("--texts-jsonl", required=True,
                   help="Source jsonl with `text` and `label` columns")
    p.add_argument("--n-per-label", type=int, default=100,
                   help="How many texts to scan per label (0 + 1).")
    p.add_argument("--layers", nargs="*", type=int, default=None,
                   help="Layer indices to interpret. Default: 16, 20, 24, 28, 31.")
    p.add_argument("--top-k", type=int, default=20,
                   help="How many top positive / top negative windows per layer.")
    p.add_argument("--context-tokens", type=int, default=8,
                   help="Number of tokens (ending at position t) to show as context.")
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bfloat16", choices=list(DTYPES))
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--out-tsv", default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def get_decoder_blocks(model: torch.nn.Module) -> list:
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    raise RuntimeError(f"Unknown decoder layout for {type(model).__name__}")


def fmt_context(tokenizer, token_ids: list) -> str:
    s = tokenizer.decode(token_ids)
    return s.replace("\n", "\\n").replace("\t", "\\t")


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    print(f"Loading {args.llm} on {args.device} (dtype={args.dtype})")
    tokenizer = AutoTokenizer.from_pretrained(args.llm, token=args.hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        args.llm, token=args.hf_token, torch_dtype=DTYPES[args.dtype],
    )
    model.to(args.device).eval()
    blocks = get_decoder_blocks(model)
    H = model.config.hidden_size

    sv_all = np.load(args.sv_path)
    if sv_all.ndim != 2 or sv_all.shape[1] != H:
        raise ValueError(f"SV shape {sv_all.shape} doesn't match (L, {H})")
    L = sv_all.shape[0]
    layers = args.layers if args.layers else [16, 20, 24, 28, L - 1]
    layers = sorted(set(l for l in layers if 0 <= l < L))
    print(f"scanning layers {layers}")

    sv_t = {l: torch.tensor(sv_all[l], dtype=torch.float32, device=args.device)
            for l in layers}

    print(f"loading {args.texts_jsonl}")
    df = pd.read_json(args.texts_jsonl, lines=True)
    sampled_rows = []
    for label in [0, 1]:
        sub = df[df["label"] == label]
        idx = rng.choice(len(sub), size=min(args.n_per_label, len(sub)), replace=False)
        sampled_rows.append(sub.iloc[idx].assign(_label=label))
    samples = pd.concat(sampled_rows).reset_index(drop=True)
    print(f"sampled {len(samples)} texts")

    # Per layer, maintain heaps of (cosine, ...) and (-cosine, ...) for top-K each.
    # heap entries: (cosine_value, text_id, position, list_of_token_ids_for_context, label)
    top_pos = {l: [] for l in layers}   # min-heap of size K (smallest at top)
    top_neg = {l: [] for l in layers}   # max-heap via negated values

    def push_pos(l, val, tid, pos, ctx, lbl):
        if len(top_pos[l]) < args.top_k:
            heapq.heappush(top_pos[l], (val, tid, pos, ctx, lbl))
        else:
            heapq.heappushpop(top_pos[l], (val, tid, pos, ctx, lbl))

    def push_neg(l, val, tid, pos, ctx, lbl):
        # We want the most negative cosines, so heap on -val (smallest -val = most negative val).
        if len(top_neg[l]) < args.top_k:
            heapq.heappush(top_neg[l], (-val, tid, pos, ctx, lbl))
        else:
            heapq.heappushpop(top_neg[l], (-val, tid, pos, ctx, lbl))

    # Capture all requested layers in one forward pass
    captured = {}

    def make_hook(layer_idx: int):
        def hook(_m, _inp, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = hidden[0].detach()  # (T, H), batch=1
        return hook

    handles = []
    for l in layers:
        handles.append(blocks[l].register_forward_hook(make_hook(l)))

    try:
        for tid, row in tqdm(samples.iterrows(), total=len(samples), desc="scan"):
            text = row["text"]
            lbl = int(row["_label"])
            inputs = tokenizer(
                text, return_tensors="pt",
                max_length=args.max_seq_length, truncation=True, padding=False,
            ).to(args.device)
            captured.clear()
            with torch.no_grad():
                model(**inputs)
            token_ids = inputs["input_ids"][0].tolist()

            for l in layers:
                acts = captured[l].float()  # (T, H)
                norms = acts.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                cosines = (acts / norms) @ sv_t[l]    # (T,)
                # vectorized: pull top-K positive and bottom-K from this text
                k = min(args.top_k, cosines.numel())
                top = torch.topk(cosines, k)
                bot = torch.topk(cosines, k, largest=False)

                for val, pos in zip(top.values.tolist(), top.indices.tolist()):
                    ctx_start = max(0, pos - args.context_tokens + 1)
                    ctx = token_ids[ctx_start:pos + 1]
                    push_pos(l, val, tid, pos, ctx, lbl)
                for val, pos in zip(bot.values.tolist(), bot.indices.tolist()):
                    ctx_start = max(0, pos - args.context_tokens + 1)
                    ctx = token_ids[ctx_start:pos + 1]
                    push_neg(l, val, tid, pos, ctx, lbl)
    finally:
        for h in handles:
            h.remove()

    rows_out = []
    for l in layers:
        pos_sorted = sorted(top_pos[l], key=lambda x: -x[0])    # most positive first
        neg_sorted = sorted(top_neg[l], key=lambda x: -x[0])    # most negative first (heap stores -val, so highest -val = most negative)
        print(f"\n=== layer {l} ===")
        print(f"  --- top {args.top_k} +sv (fake-leaning) ---")
        for cos, tid, pos, ctx, lbl in pos_sorted:
            ctx_str = fmt_context(tokenizer, ctx)
            lbl_name = "real" if lbl == 0 else "fake"
            print(f"    cos={cos:+.3f} [{lbl_name} txt={tid:>3} pos={pos:>4}]  …{ctx_str!r}")
            rows_out.append((l, "+sv", cos, lbl_name, tid, pos, ctx_str))
        print(f"  --- top {args.top_k} -sv (human-leaning) ---")
        for nval, tid, pos, ctx, lbl in neg_sorted:
            cos = -nval
            ctx_str = fmt_context(tokenizer, ctx)
            lbl_name = "real" if lbl == 0 else "fake"
            print(f"    cos={cos:+.3f} [{lbl_name} txt={tid:>3} pos={pos:>4}]  …{ctx_str!r}")
            rows_out.append((l, "-sv", cos, lbl_name, tid, pos, ctx_str))

    if args.out_tsv:
        with open(args.out_tsv, "w") as fp:
            fp.write("layer\tdirection\tcosine\tlabel\ttext_id\tposition\tcontext\n")
            for r in rows_out:
                fp.write("\t".join(str(x) for x in r) + "\n")
        print(f"\nwrote {args.out_tsv}")


if __name__ == "__main__":
    main()
