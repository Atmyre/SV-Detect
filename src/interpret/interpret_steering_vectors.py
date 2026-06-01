"""Logit-lens interpretation of steering vectors.

For each layer's steering vector `sv ∈ R^H`, apply the model's final layer-norm
and project through the LM head to get token logits:

    logits = LM_head(LN_f(sv))             # shape (vocab,)

Top-k tokens by logit value = tokens promoted by the +sv direction (i.e. the
"fake / machine-generated" side, since sv = mean(fake) - mean(real)).
Bottom-k tokens = tokens promoted by the -sv direction (the "real / human" side).

Usage:
    python interpret_steering_vectors.py \\
        --llm meta-llama/Llama-2-7b-hf \\
        --sv-path data/Llama-2-7b-hf/COLING_2025_MGT_en/steering_vectors_logreg.npy \\
        --top-k 15
"""

import argparse
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--llm", required=True, help="HF model id, e.g. meta-llama/Llama-2-7b-hf")
    p.add_argument("--sv-path", required=True,
                   help="Path to steering_vectors_*.npy of shape (num_layers, hidden)")
    p.add_argument("--top-k", type=int, default=15)
    p.add_argument("--layers", nargs="*", type=int, default=None,
                   help="Subset of layer indices to interpret (default: all)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bfloat16", choices=list(DTYPES))
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--out-tsv", default=None,
                   help="Optional TSV path to also dump top/bottom tokens per layer")
    return p.parse_args()


def get_final_norm(model: torch.nn.Module):
    """Return the model's final layer-norm module (transformer.ln_f or model.norm)."""
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f          # GPT-Neo / GPT-2 / GPT-J
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm                # Llama / Qwen / Gemma
    raise RuntimeError(f"Unknown final-norm location for {type(model).__name__}")


def project_layer(sv: torch.Tensor, final_norm, lm_head) -> torch.Tensor:
    """sv: (H,). Returns logits (V,) in fp32 on CPU."""
    with torch.no_grad():
        normed = final_norm(sv.unsqueeze(0))                     # (1, H)
        normed = normed.to(lm_head.weight.dtype)
        logits = lm_head(normed).squeeze(0).float().cpu()        # (V,)
    return logits


def fmt_token(tokenizer, idx: int) -> str:
    s = tokenizer.decode([idx])
    s = s.replace("\n", "\\n").replace("\t", "\\t")
    return f"{s!r}"


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
    V = lm_head.out_features
    print(f"hidden={H}  vocab={V}")

    sv_all = np.load(args.sv_path)
    if sv_all.ndim != 2 or sv_all.shape[1] != H:
        raise ValueError(
            f"steering_vectors shape {sv_all.shape} does not match (L, {H})"
        )
    L = sv_all.shape[0]
    layers = args.layers if args.layers else list(range(L))
    print(f"projecting {len(layers)} of {L} layers from {args.sv_path}")

    rows = []
    for layer in layers:
        sv = torch.tensor(sv_all[layer], dtype=DTYPES[args.dtype], device=args.device)
        logits = project_layer(sv, final_norm, lm_head)
        top = torch.topk(logits, args.top_k)
        bot = torch.topk(logits, args.top_k, largest=False)
        top_tokens = [fmt_token(tokenizer, i) for i in top.indices.tolist()]
        bot_tokens = [fmt_token(tokenizer, i) for i in bot.indices.tolist()]

        print(f"\n=== layer {layer:>2} ===")
        print(f"  +sv (fake-leaning)  top-{args.top_k}: " + ", ".join(top_tokens))
        print(f"  -sv (human-leaning) top-{args.top_k}: " + ", ".join(bot_tokens))

        if args.out_tsv:
            for token, val, side in zip(top_tokens, top.values.tolist(), ["+"] * len(top_tokens)):
                rows.append((layer, side, token, val))
            for token, val, side in zip(bot_tokens, bot.values.tolist(), ["-"] * len(bot_tokens)):
                rows.append((layer, side, token, val))

    if args.out_tsv:
        with open(args.out_tsv, "w") as fp:
            fp.write("layer\tside\ttoken\tlogit\n")
            for layer, side, token, val in rows:
                fp.write(f"{layer}\t{side}\t{token}\t{val:.4f}\n")
        print(f"\nwrote {args.out_tsv}")


if __name__ == "__main__":
    main()
