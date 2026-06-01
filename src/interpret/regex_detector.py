"""Token/phrase-level baseline detector built from the patterns surfaced by
the steering-vector interpretation.

Features (counts per sample, plus per-1k-chars normalized version):
  - structural: sentence-final " and", " ." (space-period), inline "\\n"
  - HTML markup: total tag count, <td>, <sub>, <p>, <body>, <html>, <script>,
                 <ref>, <div>, <li>, <table>, <span>
  - LaTeX / math: \\textit{, \\begin{, $...$, \\frac
  - invisible unicode: zwnj/zwsp/word-joiner/soft-hyphen
  - ellipsis: ..., …
  - numbered-list endings: ". 10", ". 11", ... ". 99"
  - cliché AI phrases: "essential to", "in conclusion", "world around them", etc.
  - casual hedges: basically, probably, pretty, really, actually, maybe, stuff,
                   kinda, gonna, dunno, ngl, lol
  - quoted dialogue: count of "X said," / "said X."

Trains LogisticRegression with l1, sweeps C, reports val/test metrics, and
prints the top features by coefficient.
"""

import argparse
import os
import re
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--dev-jsonl", required=True)
    p.add_argument("--test-jsonl", required=True)
    p.add_argument("--cs", nargs="+", type=float,
                   default=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
    p.add_argument("--penalty", default="l1", choices=["l1", "l2"])
    p.add_argument("--top-k", type=int, default=15)
    p.add_argument("--out-tsv", default=None,
                   help="Save per-sample features here (val + test only).")
    p.add_argument("--features", nargs="+", default=None,
                   help="If set, restrict to these feature names (raw + _per1k variants "
                        "are kept). Useful for ablation, e.g. 'space_period inline_newline'.")
    return p.parse_args()


# Each feature: (name, regex compiled with re.compile, IGNORECASE flag)
def build_feature_extractors():
    feats = OrderedDict()

    # Structural
    # Old `sent_final_and` matched only end-of-line "and" — turns out that's a
    # *human* signal (line breaks). Replace with patterns that actually capture
    # the LLM compound-list style we saw in the n-gram scan:
    #   - oxford_list: "X, Y, and Z" Oxford-comma lists (LLMs love these)
    #   - compound_VP_and: common AI-prose verbs followed shortly by "and"
    feats["oxford_list"]       = re.compile(r"\b\w+,\s+\w+,?\s+and\s+\w+", re.IGNORECASE)
    feats["compound_VP_and"]   = re.compile(
        r"\b(?:provide|develop|encourage|support|help|create|address|enhance|"
        r"promote|build|foster|navigate|highlight|include|emphasize|enable|"
        r"empower|inspire|cultivate|leverage|achieve|incorporate|gain|explore|"
        r"strengthen|maximize|deliver)\b[\w\s,]{1,40}\sand\s\w+",
        re.IGNORECASE,
    )
    feats["space_period"]      = re.compile(r"\s\.")
    feats["inline_newline"]    = re.compile(r"[a-zA-Z]\n")

    # HTML tags - total + per-tag
    feats["html_tag_total"]    = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*[^>]{0,80}>")
    for tag in ["td", "sub", "p", "body", "html", "script", "ref", "div", "li",
                "table", "span", "br", "tr", "title", "head"]:
        feats[f"html_{tag}"] = re.compile(rf"</?{tag}\b[^>]{{0,80}}>", re.IGNORECASE)

    # LaTeX / math
    feats["latex_textit"]      = re.compile(r"\\textit\{")
    feats["latex_begin"]       = re.compile(r"\\begin\{")
    feats["latex_frac"]        = re.compile(r"\\frac")
    feats["math_dollar"]       = re.compile(r"\$[^$\n]{1,40}\$")

    # Invisible unicode (single chars)
    feats["unicode_zwnj"]      = re.compile(r"‌")
    feats["unicode_zwsp"]      = re.compile(r"​")
    feats["unicode_word_joiner"] = re.compile(r"⁠")
    feats["unicode_soft_hyphen"] = re.compile(r"\xad")
    feats["unicode_em_space"]    = re.compile(r" ")

    # Ellipsis
    feats["ellipsis_three_dots"] = re.compile(r"\.\.\.")
    feats["ellipsis_unicode"]    = re.compile(r"…")

    # Numbered-list endings ". 10", ". 11", ... ". 99"
    feats["numbered_step"]     = re.compile(r"\.\s\d{1,2}\b")

    # Cliché AI phrases (case-insensitive)
    cliche_phrases = [
        r"\bin conclusion\b", r"\bit is essential\b", r"\bit's essential\b",
        r"\bthe world around (them|us)\b", r"\bin today's rapidly changing world\b",
        r"\baddress(ing)? these concerns\b", r"\bessential to develop\b",
        r"\bencourage and support\b", r"\b(must be|need to be|should be) addressed\b",
        r"\bhighlights their abilities\b", r"\bsignificant concerns\b",
        r"\bbenefit them throughout\b", r"\bthose in need\b",
        r"\bmake (better|knowledgeable) decisions\b",
        r"\bbring people together\b", r"\brobust and versatile\b",
        r"\bplays? a (crucial|vital|key|significant) role\b",
        r"\bmaximum benefit of society\b", r"\bcomplex cognitive abilities\b",
        r"\bset (them|us) apart from\b", r"\bunique experiences\b",
        r"\bhelp\s+(them|us|individuals|students) (develop|build|navigate)\b",
        r"\b(furthermore|moreover|additionally),", r"\bworkshops and conferences\b",
    ]
    feats["ai_cliche"] = re.compile("|".join(cliche_phrases), re.IGNORECASE)

    # Casual hedges
    casual_words = [
        r"\bbasically\b", r"\bprobably\b", r"\bpretty\b", r"\breally\b",
        r"\bactually\b", r"\bmaybe\b", r"\bstuff\b", r"\bkinda\b",
        r"\bgonna\b", r"\bdunno\b", r"\bngl\b", r"\blol\b", r"\bidk\b",
        r"\byea(h)?\b", r"\bnope\b", r"\blmao\b", r"\bdamn\b", r"\bsorta\b",
    ]
    feats["casual_hedge"] = re.compile("|".join(casual_words), re.IGNORECASE)

    # Quoted dialogue tags
    feats["dialogue_said"] = re.compile(r"[\"']\s*,\s*(?:said|says|asked|replied|whispered|shouted)", re.IGNORECASE)

    # Wikipedia-style citations / references
    feats["wiki_ref_bracket"] = re.compile(r"\[\d+\]")
    feats["wiki_citation"]    = re.compile(r"\b(et\s+al|cit(ed)?\.|p\.\s*\d+|vol\.\s*\d+)\b", re.IGNORECASE)

    return feats


def featurize(text: str, extractors: dict) -> np.ndarray:
    counts = np.zeros(len(extractors), dtype=np.float32)
    for i, regex in enumerate(extractors.values()):
        counts[i] = len(regex.findall(text))
    return counts


def featurize_split(jsonl_path: str, extractors: dict):
    df = pd.read_json(jsonl_path, lines=True)
    texts = df["text"].tolist()
    labels = df["label"].to_numpy()
    print(f"  {jsonl_path}: {len(texts)} samples")

    n_feats = len(extractors)
    X = np.zeros((len(texts), 2 * n_feats + 1), dtype=np.float32)
    for i, t in enumerate(texts):
        counts = featurize(t, extractors)
        char_len = max(1, len(t))
        # raw counts + per-1k-chars normalized + log(char_len)
        X[i, :n_feats]         = counts
        X[i, n_feats:2*n_feats] = 1000.0 * counts / char_len
        X[i, -1]               = np.log1p(char_len)
        if (i + 1) % 50000 == 0:
            print(f"    featurized {i + 1}")
    return X, labels


def feature_names(extractors: dict) -> list:
    base = list(extractors.keys())
    return base + [f"{n}_per1k" for n in base] + ["log_charlen"]


def evaluate(model, scaler, X, y, threshold=0.5):
    proba = model.predict_proba(scaler.transform(X))[:, 1]
    pred = proba > threshold
    return {
        "accuracy": accuracy_score(y, pred),
        "f1_macro": f1_score(y, pred, average="macro"),
        "roc_auc": roc_auc_score(y, proba),
    }


def main():
    args = parse_args()
    extractors = build_feature_extractors()
    names = feature_names(extractors)
    print(f"feature count: {len(names)}")

    # Optional feature ablation
    if args.features:
        keep_set = set(args.features)
        keep_idx = [i for i, n in enumerate(names)
                    if n in keep_set or n.replace("_per1k", "") in keep_set or n == "log_charlen" and "log_charlen" in keep_set]
        if not keep_idx:
            raise SystemExit(f"no feature names matched {args.features}; available: {names}")
        names = [names[i] for i in keep_idx]
        print(f"ablation: keeping {len(names)} features: {names}")

    print("featurizing train...")
    X_train, y_train = featurize_split(args.train_jsonl, extractors)
    print("featurizing val...")
    X_val, y_val = featurize_split(args.dev_jsonl, extractors)
    print("featurizing test...")
    X_test, y_test = featurize_split(args.test_jsonl, extractors)
    if args.features:
        X_train = X_train[:, keep_idx]
        X_val   = X_val[:,   keep_idx]
        X_test  = X_test[:,  keep_idx]
    print(f"shapes: train {X_train.shape}, val {X_val.shape}, test {X_test.shape}")

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)

    solver = "saga" if args.penalty == "l1" else "lbfgs"
    print(f"\nSweeping C over {args.cs} with penalty={args.penalty}")

    best_val_f1 = -1
    best_model = None
    best_C = None
    for C in args.cs:
        model = LogisticRegression(
            solver=solver, penalty=args.penalty, C=C,
            max_iter=2000, random_state=42,
        )
        model.fit(X_train_s, y_train)
        val_m  = evaluate(model, scaler, X_val,  y_val)
        test_m = evaluate(model, scaler, X_test, y_test)
        print(f"C={C:>7g}  "
              f"val acc={val_m['accuracy']:.4f} f1={val_m['f1_macro']:.4f} auc={val_m['roc_auc']:.4f}   "
              f"test acc={test_m['accuracy']:.4f} f1={test_m['f1_macro']:.4f} auc={test_m['roc_auc']:.4f}")
        if val_m["f1_macro"] > best_val_f1:
            best_val_f1 = val_m["f1_macro"]
            best_model = model
            best_C = C

    print(f"\nBest C = {best_C} (val F1 {best_val_f1:.4f})")
    coef = best_model.coef_[0]
    nonzero = (np.abs(coef) > 1e-8).sum()
    print(f"non-zero features: {nonzero} / {len(coef)}")

    print(f"\nTop {args.top_k} features by coef (positive = fake-leaning):")
    for i in np.argsort(-coef)[:args.top_k]:
        if coef[i] <= 0:
            break
        print(f"  +{coef[i]:>7.3f}   {names[i]}")
    print(f"\nTop {args.top_k} features by coef (negative = human-leaning):")
    for i in np.argsort(coef)[:args.top_k]:
        if coef[i] >= 0:
            break
        print(f"  {coef[i]:>7.3f}   {names[i]}")

    if args.out_tsv:
        out = pd.DataFrame(np.vstack([X_val, X_test]), columns=names)
        out["label"] = np.concatenate([y_val, y_test])
        out["split"] = ["val"] * len(y_val) + ["test"] * len(y_test)
        out.to_csv(args.out_tsv, sep="\t", index=False)
        print(f"wrote {args.out_tsv}")


if __name__ == "__main__":
    main()
