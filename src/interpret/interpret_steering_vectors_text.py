"""Contextual logit-lens interpretation: how does adding ±α·sv to a real
text's residual stream shift the next-token distribution?

For each provided text and each requested layer:
  1. Forward the text through the LM, hook each decoder block to capture the
     last-token residual at every layer.
  2. Compute next-token logits three ways:
       logits_orig   = LM_head( LN_f( act ) )
       logits_plus   = LM_head( LN_f( act + α·sv ) )
       logits_minus  = LM_head( LN_f( act − α·sv ) )
  3. Δ_plus  = logits_plus  − logits_orig   →  top-k tokens = those *boosted*
                                                by the +sv ("fake") direction
     Δ_minus = logits_minus − logits_orig   →  top-k tokens = those *boosted*
                                                by the −sv ("human") direction

`alpha` scales the (unit-norm) steering vector. Default 10; activation norms
on Llama-2 are typically 30-100, so α=10 shifts the residual ~10-30%.
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

DEFAULT_TEXTS = [
    "The quick brown fox jumps over the lazy dog. The next sentence",
    "It is important to note that this nuanced topic requires",
    "yo guys whats good lmao i was at the store and",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--llm", required=True, help="HF model id, e.g. meta-llama/Llama-2-7b-hf")
    p.add_argument("--sv-path", required=True,
                   help="Path to steering_vectors_*.npy of shape (num_layers, hidden)")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--alpha", type=float, default=10.0,
                   help="Scale factor on the unit-norm SV when adding to residual.")
    p.add_argument("--layers", nargs="*", type=int, default=None,
                   help="Layer indices to interpret. Default: a sparse grid (every 4th).")
    p.add_argument("--texts", nargs="*", default=None, help="Inline texts to use.")
    p.add_argument("--texts-jsonl", default=None,
                   help="Optional jsonl with `text` (and optionally `label`) — first "
                        "--n-texts rows from each label are used.")
    p.add_argument("--n-texts", type=int, default=2,
                   help="When using --texts-jsonl, how many examples per label.")
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bfloat16", choices=list(DTYPES))
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--out-tsv", default=None)
    return p.parse_args()


def get_decoder_blocks(model: torch.nn.Module) -> list:
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    raise RuntimeError(f"Unknown decoder layout for {type(model).__name__}")


def get_final_norm(model: torch.nn.Module):
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    raise RuntimeError(f"Unknown final-norm location for {type(model).__name__}")


def fmt_token(tokenizer, idx: int) -> str:
    s = tokenizer.decode([idx])
    s = s.replace("\n", "\\n").replace("\t", "\\t")
    return f"{s!r}"


def collect_last_token_residuals(model, tokenizer, text: str, max_seq_length: int,
                                  device: str) -> dict:
    """Run a forward pass; return {layer_idx: (1, H) tensor} of last-token residuals."""
    blocks = get_decoder_blocks(model)
    residuals = {}

    def make_hook(idx: int):
        def hook(_m, _inp, output):
            hidden = output[0] if isinstance(output, tuple) else output
            residuals[idx] = hidden[:, -1, :].detach()
        return hook

    handles = [b.register_forward_hook(make_hook(i)) for i, b in enumerate(blocks)]
    try:
        inputs = tokenizer(
            text, return_tensors="pt",
            max_length=max_seq_length, truncation=True, padding=False,
        ).to(device)
        with torch.no_grad():
            model(**inputs)
    finally:
        for h in handles:
            h.remove()
    return residuals


def project(act: torch.Tensor, final_norm, lm_head) -> torch.Tensor:
    """act: (1, H). Returns logits (V,) in fp32 on CPU."""
    with torch.no_grad():
        normed = final_norm(act).to(lm_head.weight.dtype)
        return lm_head(normed).squeeze(0).float().cpu()


def main():
    args = parse_args()

    print(f"Loading {args.llm} on {args.device} (dtype={args.dtype})")
    tokenizer = AutoTokenizer.from_pretrained(args.llm, token=args.hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        args.llm, token=args.hf_token, torch_dtype=DTYPES[args.dtype],
    )
    model.to(args.device).eval()

    final_norm = get_final_norm(model)
    lm_head = model.lm_head
    H = model.config.hidden_size
    print(f"hidden={H}  vocab={lm_head.out_features}")

    sv_all = np.load(args.sv_path)
    L = sv_all.shape[0]
    layers = args.layers if args.layers else list(range(0, L, max(1, L // 8)))
    print(f"projecting layers {layers} from {args.sv_path}")

    # Resolve texts to use
    if args.texts:
        texts = list(args.texts)
        text_meta = [("inline", "?", t) for t in texts]
    elif args.texts_jsonl:
        df = pd.read_json(args.texts_jsonl, lines=True)
        texts, text_meta = [], []
        for label in [0, 1]:
            sub = df[df["label"] == label].head(args.n_texts) if "label" in df.columns else df.head(args.n_texts)
            for _, row in sub.iterrows():
                texts.append(row["text"])
                lbl_name = "real" if label == 0 else "fake"
                text_meta.append((args.texts_jsonl, lbl_name, row["text"]))
    else:
        texts = DEFAULT_TEXTS
        text_meta = [("default", "?", t) for t in texts]

    sv_t = torch.tensor(sv_all, dtype=DTYPES[args.dtype], device=args.device)

    rows = []
    for ti, (src, lbl, text) in enumerate(text_meta):
        print(f"\n################ text {ti} (src={src}, label={lbl}) ################")
        snippet = text[:120].replace("\n", " ")
        print(f"  text[:120] = {snippet!r}{'...' if len(text) > 120 else ''}")

        residuals = collect_last_token_residuals(
            model, tokenizer, text, args.max_seq_length, args.device,
        )

        for layer in layers:
            act = residuals[layer]                          # (1, H)
            sv = sv_t[layer]                                # (H,)
            logits_orig  = project(act,                       final_norm, lm_head)
            logits_plus  = project(act + args.alpha * sv,    final_norm, lm_head)
            logits_minus = project(act - args.alpha * sv,    final_norm, lm_head)
            d_plus  = logits_plus  - logits_orig
            d_minus = logits_minus - logits_orig

            top_plus  = d_plus.topk(args.top_k)
            top_minus = d_minus.topk(args.top_k)

            tp_tokens = [fmt_token(tokenizer, i) for i in top_plus.indices.tolist()]
            tm_tokens = [fmt_token(tokenizer, i) for i in top_minus.indices.tolist()]

            print(f"  --- layer {layer:>2} ---")
            print(f"    +α·sv (fake)  Δlogit↑: " + ", ".join(tp_tokens))
            print(f"    −α·sv (human) Δlogit↑: " + ", ".join(tm_tokens))

            if args.out_tsv:
                for token, val in zip(tp_tokens, top_plus.values.tolist()):
                    rows.append((ti, lbl, layer, "+sv", token, val))
                for token, val in zip(tm_tokens, top_minus.values.tolist()):
                    rows.append((ti, lbl, layer, "-sv", token, val))

    if args.out_tsv:
        with open(args.out_tsv, "w") as fp:
            fp.write("text_id\tlabel\tlayer\tdirection\ttoken\tdelta_logit\n")
            for r in rows:
                fp.write("\t".join(str(x) for x in r[:-1]) + f"\t{r[-1]:.4f}\n")
        print(f"\nwrote {args.out_tsv}")


if __name__ == "__main__":
    main()
