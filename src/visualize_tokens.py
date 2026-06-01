"""Per-token "fakeness" visualization for the FTD steering paper.

For each sampled text (real / fake), forward-pass GPT-Neo-2.7B, capture
per-token residuals at the auto-selected best layer, project each token's
residual onto the precomputed steering vector (logreg_l2 method), then render
a PNG showing tokens colored by their dot-product score and an overall
"FAKE"/"REAL" classification banner.

Usage on the cluster:
    cd ${PROJECT_ROOT:-$(pwd)}
    python visualize_tokens.py \
        --sv-path data/gpt-neo-2.7B/MIRAGE/steering_vectors_logreg_l2_generate.npy \
        --raw-json data/gpt-neo-2.7B/MIRAGE/raw_texts/DIG/generate.json \
        --out-dir teaser_pngs \
        --n-per-class 10
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--llm", default="EleutherAI/gpt-neo-2.7B")
    p.add_argument("--sv-path", required=True)
    p.add_argument("--raw-json", required=True,
                   help="Path to a MIRAGE raw_texts/{DIG,SIG}/{generate,polish,rewrite}.json")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-per-class", type=int, default=10)
    p.add_argument("--max-tokens", type=int, default=200,
                   help="Truncate each text to this many tokens for the figure.")
    p.add_argument("--max-words", type=int, default=0,
                   help="If >0, truncate each text to this many whitespace-"
                        "delimited words BEFORE tokenization. 0 = disabled.")
    p.add_argument("--layer", type=int, default=None,
                   help="If set, use this layer. Otherwise auto-pick by AUROC.")
    p.add_argument("--auc-probe-n", type=int, default=80,
                   help="When auto-picking a layer, score this many real+fake.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="float16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--width", type=int, default=760,
                   help="PNG width in pixels (narrow = fits half a page).")
    p.add_argument("--scale-pct", type=float, default=70.0,
                   help="Percentile used to normalize per-token scores. "
                        "Lower = more tokens reach full color saturation.")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    return p.parse_args()


DTYPES = {"float32": torch.float32, "float16": torch.float16,
          "bfloat16": torch.bfloat16}


# ---------------------------------------------------------------------------
# Model + hooks
# ---------------------------------------------------------------------------

class ResidualGrabber:
    """Hook every decoder block to capture per-token residual stream output."""

    def __init__(self, model):
        self.model = model
        self.handles = []
        self.cache: list[torch.Tensor] = []
        self._setup()

    def _setup(self):
        try:
            blocks = self.model.transformer.h            # GPT-Neo
        except AttributeError:
            try:
                blocks = self.model.model.layers         # Llama
            except AttributeError as e:
                raise RuntimeError("Don't know how to find decoder blocks") from e

        def make_hook(idx):
            def hook(_module, _inputs, output):
                # output may be a tuple; the first element is the hidden state
                hidden = output[0] if isinstance(output, tuple) else output
                # store on CPU to keep VRAM down across long sequences
                self.cache.append(hidden.detach().to(torch.float32).cpu())
            return hook

        self.cache = []
        for i, blk in enumerate(blocks):
            self.handles.append(blk.register_forward_hook(make_hook(i)))

    def reset(self):
        self.cache = []

    def close(self):
        for h in self.handles:
            h.remove()


# ---------------------------------------------------------------------------
# Auto layer selection
# ---------------------------------------------------------------------------

@torch.no_grad()
def per_token_residuals(model, tokenizer, text: str, max_length: int,
                        device: str, grabber: ResidualGrabber
                        ) -> Tuple[List[str], torch.Tensor]:
    """Run text through the model; return per-token display strings and
    (n_layers, T, hidden)."""
    # Fast tokenizer gives offset_mapping, which lets us extract each token's
    # visible substring directly from the source text — sidesteps every
    # BPE-byte-encoding artifact (apostrophes, ellipses, smart quotes).
    enc = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=max_length, return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(device) for k, v in enc.items()}
    grabber.reset()
    model(**enc)
    layers = torch.stack([c.squeeze(0) for c in grabber.cache], dim=0)
    display = []
    for (start, end) in offsets:
        if end > start:
            display.append(text[start:end])
        else:
            # Special tokens (no character span) — keep an empty placeholder
            # so the score array stays aligned.
            display.append("")
    return display, layers


def _split_newlines(display_tokens):
    """Split tokens whose decoded form contains newlines so the renderer can
    emit explicit line breaks."""
    out = []
    for t in display_tokens:
        if "\n" not in t:
            out.append(t)
            continue
        parts = t.split("\n")
        for i, p in enumerate(parts):
            if p:
                out.append(p)
            if i < len(parts) - 1:
                out.append("\n")
    return out


@torch.no_grad()
def auto_pick_layer(model, tokenizer, sv: np.ndarray, samples: list,
                    n_probe: int, max_length: int, device: str,
                    grabber: ResidualGrabber) -> int:
    """Sample n_probe real + n_probe fake; pick layer with best mean-pool AUROC."""
    sv_t = torch.from_numpy(sv)            # (L, H)
    L = sv_t.shape[0]
    real = [s for s in samples if s["label"] == 0][:n_probe]
    fake = [s for s in samples if s["label"] == 1][:n_probe]
    probe = real + fake
    y = np.array([0] * len(real) + [1] * len(fake))
    scores_per_layer = np.zeros((len(probe), L), dtype=np.float32)
    for i, s in enumerate(probe):
        _, resids = per_token_residuals(model, tokenizer, s["text"],
                                        max_length, device, grabber)
        # mean over tokens -> (L, H), then dot with SV -> (L,)
        mean = resids.mean(dim=1)             # (L, H)
        scores_per_layer[i] = (mean * sv_t).sum(dim=-1).numpy()
    aurocs = np.array([roc_auc_score(y, scores_per_layer[:, l])
                       for l in range(L)])
    best = int(np.argmax(aurocs))
    print(f"AUROC per layer: best={best} (AUROC={aurocs[best]:.4f})")
    print("Layer AUROCs:", " ".join(f"{l}:{a:.3f}" for l, a in enumerate(aurocs)))
    return best


# ---------------------------------------------------------------------------
# Color mapping
# ---------------------------------------------------------------------------

def lerp_color(c0, c1, t):
    return tuple(int(round(c0[i] + (c1[i] - c0[i]) * t)) for i in range(3))


WHITE = (255, 255, 255)
RED   = (213,  84,  82)     # "fake" extreme (background)
BLUE  = ( 82, 130, 213)     # "real" extreme (background)

# Saturated text colors. Anchored at a near-black "neutral" so weakly-scored
# tokens stay readable rather than washing out toward white. Extremes are
# pushed to vivid saturation so the figure stays punchy when scaled down to
# a paper teaser column.
TEXT_NEUTRAL = ( 40,  40,  40)
TEXT_RED     = (200,  10,  20)
TEXT_BLUE    = ( 10,  45, 200)
COLOR_GAMMA  = 0.55   # <1 boosts weak colors so mid-range tokens stay visible


def score_to_color(z: float) -> tuple:
    """Background color map (white ↔ red/blue) for the legend / banner."""
    z = float(max(-1.0, min(1.0, z)))
    if z >= 0:
        return lerp_color(WHITE, RED, z)
    return lerp_color(WHITE, BLUE, -z)


def score_to_text_color(z: float) -> tuple:
    """Foreground (text) color map: dark grey ↔ saturated red/blue.
    A gamma curve (<1) pushes mid-strength tokens further toward the
    saturated end so the figure stays readable at small print sizes."""
    z = float(max(-1.0, min(1.0, z)))
    sign = 1 if z >= 0 else -1
    t = (abs(z)) ** COLOR_GAMMA       # boost weak signals
    if sign > 0:
        return lerp_color(TEXT_NEUTRAL, TEXT_RED, t)
    return lerp_color(TEXT_NEUTRAL, TEXT_BLUE, t)


# ---------------------------------------------------------------------------
# PIL rendering
# ---------------------------------------------------------------------------

def _matplotlib_font(name):
    """Find a font file bundled with matplotlib (always available in this env)."""
    try:
        import matplotlib
        base = os.path.join(os.path.dirname(matplotlib.__file__),
                            "mpl-data", "fonts", "ttf")
        return os.path.join(base, name)
    except Exception:
        return None


FONT_CANDIDATES_REG = [
    _matplotlib_font("DejaVuSans.ttf"),
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
]
FONT_CANDIDATES_BOLD = [
    _matplotlib_font("DejaVuSans-Bold.ttf"),
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
]
FONT_CANDIDATES_ITALIC = [
    _matplotlib_font("DejaVuSans-Oblique.ttf"),
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
]


def load_font(candidates, size):
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def detokenize_for_display(tokens: List[str]) -> List[str]:
    """Convert subword tokens to display strings, preserving BPE spaces."""
    out = []
    for t in tokens:
        # GPT-Neo uses Ġ for leading space (GPT-2 style BPE)
        if t.startswith("Ġ"):
            out.append(" " + t[1:])
        elif t.startswith("Ċ"):
            out.append("\n")
        else:
            out.append(t)
    return out


def _layout_text(display_tokens: List[str], scores: np.ndarray,
                 font, scale_pct: float, width: int, pad: int,
                 y_start: int):
    """Compute one block's token layout. Returns (layout, y_bottom, z_array, band_h, line_step, band_pad_y)."""
    ascent, descent = font.getmetrics()
    glyph_h = ascent + descent
    band_pad_y = 3
    band_h = glyph_h + 2 * band_pad_y
    line_gap = 8
    line_step = band_h + line_gap
    line_w_limit = width - pad

    s = np.asarray(scores, dtype=np.float32)
    scale = max(1e-6, np.percentile(np.abs(s), scale_pct))
    z = s / scale

    layout = []
    dummy = Image.new("RGB", (10, 10), "white")
    measure = ImageDraw.Draw(dummy)
    x = pad
    y_band = y_start
    for tok_str, z_i in zip(display_tokens, z):
        if tok_str == "\n":
            x = pad
            y_band += line_step
            continue
        bbox = measure.textbbox((0, 0), tok_str, font=font)
        w = bbox[2] - bbox[0]
        if x + w > line_w_limit and tok_str.strip():
            x = pad
            y_band += line_step
        layout.append((x, y_band, w, tok_str, z_i))
        x += w
    return layout, y_band + band_h, band_pad_y


def _draw_banner(draw, x0, y0, x1, y1, color, msg, font, anchor_pad):
    draw.rectangle((x0, y0, x1, y1), fill=color)
    draw.text((x0 + anchor_pad, (y0 + y1) // 2), msg, font=font,
              fill="white", anchor="lm")


def _draw_colorbar(draw, width, pad, cbar_top, small_font):
    bar_w = min(280, width - 2 * pad)
    bar_x = (width - bar_w) // 2
    bar_h = 12
    for i in range(bar_w):
        z_i = -1.0 + 2.0 * i / (bar_w - 1)
        col = score_to_text_color(z_i)
        draw.rectangle((bar_x + i, cbar_top, bar_x + i + 1, cbar_top + bar_h),
                       fill=col)
    draw.text((bar_x, cbar_top + bar_h + 4), "more real", font=small_font,
              fill=TEXT_BLUE)
    draw.text((bar_x + bar_w, cbar_top + bar_h + 4), "more fake",
              font=small_font, fill=TEXT_RED, anchor="rt")


def render_text(display_tokens: List[str], scores: np.ndarray,
                pred_label: str, pred_score: float, true_label: str,
                out_path: Path,
                width: int = 760, pad: int = 18, font_size: int = 15,
                scale_pct: float = 80.0):
    """Single-example render: banner + colored text + colorbar."""
    font = load_font(FONT_CANDIDATES_REG, font_size)
    bold = load_font(FONT_CANDIDATES_BOLD, font_size + 2)
    small = load_font(FONT_CANDIDATES_REG, font_size - 2)

    banner_h = 36
    layout, text_bottom, band_pad_y = _layout_text(
        display_tokens, scores, font, scale_pct, width, pad,
        y_start=banner_h + pad)

    cbar_top = text_bottom + 16
    total_h = cbar_top + 36 + pad

    img = Image.new("RGB", (width, total_h), "white")
    draw = ImageDraw.Draw(img)

    banner_color = RED if pred_label.upper() == "FAKE" else BLUE
    msg = (f"Predicted: {pred_label.upper()}   "
           f"score {pred_score:+.2f}   "
           f"(truth: {true_label})")
    _draw_banner(draw, 0, 0, width, banner_h, banner_color, msg, bold,
                 anchor_pad=pad)

    for tx, ty_band, tw, tok_str, z_i in layout:
        text_y = ty_band + band_pad_y
        draw.text((tx, text_y), tok_str, font=font,
                  fill=score_to_text_color(z_i))

    _draw_colorbar(draw, width, pad, cbar_top, small)
    img.save(out_path)


def _draw_compact_label(draw, x, y, pred, truth, bold_font, italic_font):
    """Small colored dot + 'Predicted: FAKE  (truth: FAKE)' label.

    Returns the y position just below the label."""
    pred_up = pred.upper()
    label_color = RED if pred_up == "FAKE" else BLUE
    dot_r = 7
    dot_cy = y + 11
    draw.ellipse((x, dot_cy - dot_r, x + 2 * dot_r, dot_cy + dot_r),
                 fill=label_color)
    tx = x + 2 * dot_r + 10
    label_text = f"Predicted: {pred_up}"
    draw.text((tx, y), label_text, font=bold_font, fill=label_color)
    bbox = draw.textbbox((tx, y), label_text, font=bold_font)
    sub_x = bbox[2] + 12
    draw.text((sub_x, y + 1), f"(ground truth: {truth})",
              font=italic_font, fill=(120, 120, 120))
    return y + 24


def _truncate_to_words(tokens, scores, keep_ratio=2 / 3):
    """Keep ~`keep_ratio` of the words (cutting on word boundaries) and append
    a trailing '…' token. Word boundaries are detected by leading-space tokens,
    which is how GPT-style BPE tokenizers mark them."""
    tokens = list(tokens)
    scores = np.asarray(scores, dtype=np.float32)
    word_starts = [
        i for i, t in enumerate(tokens)
        if (t.startswith(" ") or t == "\n" or i == 0) and t.strip()
    ]
    if len(word_starts) < 4:
        return tokens, scores
    target = max(1, int(round(len(word_starts) * keep_ratio)))
    if target >= len(word_starts):
        return tokens, scores
    cut_idx = word_starts[target]
    # Drop any trailing newline / whitespace-only tokens we'd otherwise leave
    # dangling at the cut.
    while cut_idx > 0 and not tokens[cut_idx - 1].strip():
        cut_idx -= 1
    truncated_tokens = tokens[:cut_idx] + ["…"]
    truncated_scores = np.concatenate(
        [scores[:cut_idx], np.array([0.0], dtype=np.float32)]
    )
    return truncated_tokens, truncated_scores


def render_pair(fake_tokens, fake_scores, fake_pred, fake_score, fake_truth,
                real_tokens, real_scores, real_pred, real_score, real_truth,
                out_path: Path,
                width: int = 760, pad: int = 22, font_size: int = 19,
                scale_pct: float = 80.0,
                keep_word_ratio: float = 2 / 3):
    """Teaser-style pair: fake on top, hairline separator, real below.
    Compact colored-dot labels instead of full-width banners. No colorbar."""
    fake_tokens, fake_scores = _truncate_to_words(
        fake_tokens, fake_scores, keep_word_ratio)
    real_tokens, real_scores = _truncate_to_words(
        real_tokens, real_scores, keep_word_ratio)

    font = load_font(FONT_CANDIDATES_REG, font_size)
    bold = load_font(FONT_CANDIDATES_BOLD, font_size - 2)
    italic = load_font(FONT_CANDIDATES_ITALIC, font_size - 4)

    label_block_h = 24       # height the compact label takes
    after_label_gap = 8      # space between label and text
    inter_gap = 18           # space between fake-end and real-label
    border_pad = 14          # inner padding from the rounded border

    body_y_start = pad + border_pad + label_block_h + after_label_gap

    fake_layout, fake_bottom, band_pad_y = _layout_text(
        fake_tokens, fake_scores, font, scale_pct, width, pad + border_pad,
        y_start=body_y_start)

    sep_y = fake_bottom + inter_gap // 2
    real_label_y = sep_y + inter_gap // 2 + 2
    real_body_y = real_label_y + label_block_h + after_label_gap

    real_layout, real_bottom, _ = _layout_text(
        real_tokens, real_scores, font, scale_pct, width, pad + border_pad,
        y_start=real_body_y)

    total_h = real_bottom + border_pad + pad

    img = Image.new("RGB", (width, total_h), "white")
    draw = ImageDraw.Draw(img)

    # Pronounced rounded border around the whole figure
    border_color = (110, 120, 140)
    border_width = 3
    try:
        draw.rounded_rectangle(
            (pad // 2, pad // 2, width - pad // 2, total_h - pad // 2),
            radius=14, outline=border_color, width=border_width,
        )
    except AttributeError:
        draw.rectangle(
            (pad // 2, pad // 2, width - pad // 2, total_h - pad // 2),
            outline=border_color, width=border_width,
        )

    # Fake label + tokens
    _draw_compact_label(draw, pad + border_pad, pad + border_pad,
                        fake_pred, fake_truth, bold, italic)
    for tx, ty_band, tw, tok_str, z_i in fake_layout:
        text_y = ty_band + band_pad_y
        draw.text((tx, text_y), tok_str, font=font,
                  fill=score_to_text_color(z_i))

    # Hairline separator
    draw.line(
        (pad + border_pad + 40, sep_y, width - pad - border_pad - 40, sep_y),
        fill=(220, 220, 225), width=1,
    )

    # Real label + tokens
    _draw_compact_label(draw, pad + border_pad, real_label_y,
                        real_pred, real_truth, bold, italic)
    for tx, ty_band, tw, tok_str, z_i in real_layout:
        text_y = ty_band + band_pad_y
        draw.text((tx, text_y), tok_str, font=font,
                  fill=score_to_text_color(z_i))

    img.save(out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_samples(json_path: str, max_words: int = 0) -> list:
    """Auto-detect schema; return [{text, label, source}, ...] with
    label 0 = real / human, 1 = fake / LLM-generated.

    Supported schemas:
      * MIRAGE  ({original, rewritten, dataset, ...})
      * DetectRL ({text, label="human"|"llm", data_type, llm_type, ...})
    """
    with open(json_path) as f:
        data = json.load(f)

    def trim(s: str) -> str:
        if max_words <= 0:
            return s
        words = s.split()
        if len(words) <= max_words:
            return s
        return " ".join(words[:max_words])

    if not data:
        return []

    first = data[0]
    out = []
    if "original" in first or "rewritten" in first:
        for d in data:
            if d.get("original"):
                out.append({"text": trim(d["original"]), "label": 0,
                            "source": d.get("dataset", "?")})
            if d.get("rewritten"):
                out.append({"text": trim(d["rewritten"]), "label": 1,
                            "source": d.get("dataset", "?")})
    elif "text" in first and "label" in first:
        for d in data:
            label = d["label"].lower()
            if label in ("human", "real", "0"):
                lab_int = 0
                source = d.get("data_type", "?")
            elif label in ("llm", "machine", "fake", "ai", "1"):
                lab_int = 1
                source = d.get("llm_type", d.get("data_type", "?"))
            else:
                continue
            out.append({"text": trim(d["text"]), "label": lab_int,
                        "source": source})
    else:
        raise ValueError(
            f"Unrecognized schema in {json_path}; first keys: {list(first)}"
        )
    return out


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sv = np.load(args.sv_path)
    print(f"SV shape: {sv.shape}")

    samples = load_samples(args.raw_json, max_words=args.max_words)
    random.shuffle(samples)
    print(f"Loaded {len(samples)} samples from {args.raw_json}")

    tok = AutoTokenizer.from_pretrained(args.llm, token=args.hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        args.llm, torch_dtype=DTYPES[args.dtype], token=args.hf_token,
    ).to(args.device).eval()

    grabber = ResidualGrabber(model)
    try:
        if args.layer is None:
            best_layer = auto_pick_layer(model, tok, sv, samples,
                                         args.auc_probe_n, args.max_tokens,
                                         args.device, grabber)
        else:
            best_layer = args.layer
        print(f"Using layer {best_layer}")

        sv_layer = torch.from_numpy(sv[best_layer])    # (H,)

        # Now pick n_per_class fresh samples (excluding the AUROC probe)
        probe_count = args.auc_probe_n if args.layer is None else 0
        pool = samples[probe_count:]
        real_pool = [s for s in pool if s["label"] == 0][:args.n_per_class]
        fake_pool = [s for s in pool if s["label"] == 1][:args.n_per_class]

        # Classification threshold: use the median dot-product of the probe set
        # if available, otherwise zero. The SV is unit-norm but mean residuals
        # aren't zero-centered, so a non-zero threshold is more honest.
        threshold = 0.0
        if args.layer is None:
            probe = samples[:probe_count]
            y_probe = np.array([s["label"] for s in probe])
            scores = []
            for s in probe:
                _, resids = per_token_residuals(model, tok, s["text"],
                                                args.max_tokens, args.device,
                                                grabber)
                mean = resids[best_layer].mean(dim=0)
                scores.append(float((mean * sv_layer).sum()))
            scores = np.array(scores)
            # threshold = midpoint between class means
            threshold = float((scores[y_probe == 0].mean()
                               + scores[y_probe == 1].mean()) / 2)
            print(f"Decision threshold = {threshold:.4f}")

        # First, score every sample so we can render singles and pairs.
        per_class_info = {"real": [], "fake": []}
        for klass_name, klass_samples, true_label in (
            ("real", real_pool, "REAL"),
            ("fake", fake_pool, "FAKE"),
        ):
            for i, s in enumerate(klass_samples):
                tokens_display, resids = per_token_residuals(
                    model, tok, s["text"], args.max_tokens, args.device,
                    grabber)
                layer_resid = resids[best_layer]
                per_tok = (layer_resid * sv_layer).sum(dim=-1)
                per_tok_np = per_tok.numpy()
                mean_score = float(per_tok_np.mean())
                pred = "FAKE" if mean_score > threshold else "REAL"
                # Expand tokens with embedded newlines into separate "\n"
                # markers and replicate their per-token score.
                display, scores_expanded = [], []
                for t, sc in zip(tokens_display, per_tok_np - threshold):
                    if "\n" in t:
                        parts = t.split("\n")
                        for j, p in enumerate(parts):
                            if p:
                                display.append(p); scores_expanded.append(sc)
                            if j < len(parts) - 1:
                                display.append("\n"); scores_expanded.append(0.0)
                    else:
                        display.append(t); scores_expanded.append(sc)
                scores_expanded = np.asarray(scores_expanded, dtype=np.float32)
                per_class_info[klass_name].append({
                    "i": i, "display": display,
                    "scores": scores_expanded,
                    "pred": pred, "adj": mean_score - threshold,
                    "raw": mean_score, "truth": true_label,
                    "source": s["source"],
                })
                fname = out_dir / f"{klass_name}_{i:02d}.png"
                render_text(display, scores_expanded,
                            pred, mean_score - threshold, true_label, fname,
                            scale_pct=args.scale_pct, width=args.width)
                print(f"  {fname.name}  pred={pred}  raw={mean_score:+.3f}  "
                      f"adj={mean_score - threshold:+.3f}  src={s['source']}")

        # Build (fake, real) pair PNGs. Sort each side by absolute confidence
        # (most-confident-first) so the pair samples are the clearer ones.
        pairs_dir = out_dir / "pairs"
        pairs_dir.mkdir(exist_ok=True)
        fakes_sorted = sorted(per_class_info["fake"],
                              key=lambda r: -r["adj"])      # most-fake first
        reals_sorted = sorted(per_class_info["real"],
                              key=lambda r: r["adj"])       # most-real first
        for k in range(min(len(fakes_sorted), len(reals_sorted))):
            f, r = fakes_sorted[k], reals_sorted[k]
            fname = pairs_dir / f"pair_{k:02d}.png"
            render_pair(
                f["display"], f["scores"], f["pred"], f["adj"], f["truth"],
                r["display"], r["scores"], r["pred"], r["adj"], r["truth"],
                fname, scale_pct=args.scale_pct, width=args.width,
            )
            print(f"  pairs/{fname.name}  fake({f['source']}) + real({r['source']})")
    finally:
        grabber.close()


if __name__ == "__main__":
    main()
