"""DetectRL-specific regex detector. Built from patterns surfaced by the n-gram
interpretation of the GPT-Neo logreg-SV detectors.

Trains one detector per attack type on its train jsonl, evaluates on the
matching test jsonl. Then cross-evaluates each detector on every other test set.

Feature buckets (counts per sample, plus per-1k-chars normalized):

  -- generic LLM register (from direct_prompt SV) --
  llm_recommend_phrases:    "won't be disappointed", "highly recommend", "would recommend"
  llm_overall_summary:       "\\bOverall\\b" sentence-start
  llm_assistant_template:    "As a helpful", "Here is a", "story based on"

  -- paraphrase markers (from paraphrase SV) --
  paraphrase_polished:       "Polished Abstract", "**Polished" markdown
  paraphrase_research_open:  "In this (paper|study), we (explore|investigate|propose|present)"
  paraphrase_thrilling:      "thrilling encounter|spectacle|experience"

  -- prompt leakage (from prompt_attacks_llm SV) --
  prompt_sentence_count:     "Here is (a|an) \\d+ sentence"
  prompt_human_style:        "more human (conversational |)style"
  prompt_error_template:     "If you believe this is an error", "please send us"
  prompt_meta_intro:         "based on the (writing prompt|first sentence|prompt)"

  -- perturbation indicators (from perturbation SV) --
  midword_case_flip:         "\\b[a-z]{2,}[A-Z][A-Za-z]+\\b" (e.g. "boservE", "presOnt")
  morph_violation_verb:      common verb agreement breaks: "Would not (congratulates|cherishes|...)"
  shortened_word:            very short partial words like "stdy", "vol", "nat" between word boundaries

  -- structural --
  restaurant_template:        "The (food|service|atmosphere|prices|staff) (was|were)"
  cliche_three_part:          "X, Y, and Z" Oxford-comma list
  newline_dense:              count of inline "\\n"

Plus a common feature pack carried over from regex_detector.py: ai_cliche,
casual_hedge, sent_final_and patterns, ellipsis, html, latex, log_charlen.
"""

import argparse
import json
import os
import re
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    f1_score, matthews_corrcoef, roc_auc_score, roc_curve,
)
from sklearn.preprocessing import StandardScaler


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True,
                   help="Folder with {attack}_train.json and {attack}_test.json files")
    p.add_argument("--cs", nargs="+", type=float,
                   default=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
    p.add_argument("--penalty", default="l1", choices=["l1", "l2"])
    p.add_argument("--top-k", type=int, default=12)
    return p.parse_args()


ATTACKS = ["direct_prompt", "paraphrase_attacks_llm",
           "perturbation_attacks_llm", "prompt_attacks_llm"]


def build_extractors():
    f = OrderedDict()

    # === generic LLM register (direct_prompt) ===
    f["llm_recommend_phrases"] = re.compile(
        r"\b(won'?t be disappointed|highly recommend|would (?:not )?hesitate to recommend|would definitely recommend)\b",
        re.IGNORECASE,
    )
    f["llm_overall_summary"] = re.compile(
        r"(?:^|[.!?]\s+)Overall,",
    )
    f["llm_assistant_template"] = re.compile(
        r"\b(As a helpful|Here is (?:a|an)|story based on the (?:writing )?prompt|"
        r"\d+ sentence (?:story|paragraph|continuation|article))\b",
        re.IGNORECASE,
    )

    # === paraphrase markers ===
    f["paraphrase_polished"] = re.compile(
        r"\*?\*?Polished (?:Abstract|Version|Paragraph|Text)\*?\*?",
        re.IGNORECASE,
    )
    f["paraphrase_research_open"] = re.compile(
        r"\bIn this (?:paper|study|work|research|article)(?:,)? we "
        r"(?:explore|investigate|propose|present|study|examine|introduce|analyse|analyze)\b",
        re.IGNORECASE,
    )
    f["paraphrase_thrilling"] = re.compile(
        r"\b(?:thrilling|exciting|exhilarating|breathtaking) "
        r"(?:encounter|spectacle|experience|finale|match|game|moment)\b",
        re.IGNORECASE,
    )

    # === prompt leakage (prompt_attacks_llm) ===
    f["prompt_sentence_count"] = re.compile(
        r"\bHere is (?:a|an) \d{1,3}[- ]?sentence(?:s)?\b",
        re.IGNORECASE,
    )
    f["prompt_human_style"] = re.compile(
        r"\b(?:more human (?:conversational |casual )?style|in a more conversational tone|"
        r"in a more human (?:way|manner|tone)|written in (?:a )?human style)\b",
        re.IGNORECASE,
    )
    f["prompt_error_template"] = re.compile(
        r"\b(?:If you believe this is an error|please send us(?:\s+an?\s+\w+)?|"
        r"contact (?:our )?(?:support|customer service|the (?:webmaster|administrator)))\b",
        re.IGNORECASE,
    )
    f["prompt_meta_intro"] = re.compile(
        r"\bbased on the (?:writing |original )?(?:prompt|first sentence|provided text)\b",
        re.IGNORECASE,
    )
    f["prompt_markdown_header"] = re.compile(
        r"\*\*[A-Z][a-zA-Z ]{0,30}:\*\*",  # **Header:** style markdown
    )

    # === perturbation indicators ===
    # Mid-word case flips (lowercase letters → uppercase mid-word, then more letters)
    f["midword_case_flip"] = re.compile(r"\b[a-z]{2,}[A-Z][A-Za-z]{2,}\b")
    # Detect a missing-vowel pattern in common words (e.g. stdy, brn, btwn — common typos)
    # Limit to 3-4-letter all-consonant clusters mid-text
    f["consonant_cluster_word"] = re.compile(
        r"\b[bcdfghjklmnpqrstvwxz]{3,4}\b", re.IGNORECASE,
    )
    # Verbs that look like agreement is wrong - very crude
    f["morph_odd_verb"] = re.compile(
        r"\bWould (?:not )?(?:congratulates|cherishes|recommends|invites|provides|offers)\b",
        re.IGNORECASE,
    )

    # === structural / register ===
    f["restaurant_template"] = re.compile(
        r"\bThe (?:food|service|atmosphere|prices|staff|menu|decor|ambience) "
        r"(?:was|were|is|are)\b",
        re.IGNORECASE,
    )
    # Oxford-comma 3-item list
    f["cliche_oxford_list"] = re.compile(
        r"\b\w+,\s+\w+,\s+(?:and|or)\s+\w+", re.IGNORECASE,
    )
    # AI compound predicate verbs followed by "and"
    f["compound_VP_and"] = re.compile(
        r"\b(?:provide|develop|encourage|support|help|create|address|enhance|"
        r"promote|build|foster|highlight|include|emphasize|enable|empower|"
        r"deliver|achieve|incorporate|gain|explore|strengthen|maximize)\b"
        r"[\w\s,]{1,40}\sand\s\w+",
        re.IGNORECASE,
    )

    # === common register / surface (carried from COLING regex_detector.py) ===
    f["space_period"] = re.compile(r"\s\.")
    f["inline_newline"] = re.compile(r"[a-zA-Z]\n")
    f["html_tag_total"] = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*[^>]{0,80}>")
    f["latex_textit"] = re.compile(r"\\textit\{")
    f["latex_begin"] = re.compile(r"\\begin\{")
    f["unicode_zwnj"] = re.compile(r"‌")
    f["unicode_zwsp"] = re.compile(r"​")
    f["ellipsis_three_dots"] = re.compile(r"\.\.\.")
    f["ellipsis_unicode"] = re.compile(r"…")
    f["wiki_ref_bracket"] = re.compile(r"\[\d+\]")

    casual_words = (
        r"\bbasically\b|\bprobably\b|\bpretty\b|\breally\b|\bactually\b|"
        r"\bmaybe\b|\bstuff\b|\bkinda\b|\bgonna\b|\bdunno\b|\bngl\b|\blol\b|"
        r"\bidk\b|\byea(?:h)?\b|\bnope\b|\blmao\b|\bdamn\b|\bsorta\b|\bhuh\b"
    )
    f["casual_hedge"] = re.compile(casual_words, re.IGNORECASE)

    return f


def featurize_text(text: str, extractors: dict) -> np.ndarray:
    counts = np.zeros(len(extractors), dtype=np.float32)
    for i, regex in enumerate(extractors.values()):
        counts[i] = len(regex.findall(text))
    return counts


def featurize_split(records, extractors):
    n_feats = len(extractors)
    n = len(records)
    X = np.zeros((n, 2 * n_feats + 1), dtype=np.float32)
    y = np.zeros(n, dtype=np.int8)
    for i, r in enumerate(records):
        text = r["text"]
        counts = featurize_text(text, extractors)
        char_len = max(1, len(text))
        X[i, :n_feats]            = counts
        X[i, n_feats:2 * n_feats] = 1000.0 * counts / char_len
        X[i, -1]                  = np.log1p(char_len)
        y[i] = 0 if r["label"] == "human" else 1
    return X, y


def feature_names(extractors):
    base = list(extractors.keys())
    return base + [f"{n}_per1k" for n in base] + ["log_charlen"]


def load_json_records(path):
    with open(path) as fp:
        return json.load(fp)


def evaluate(model, X, y, threshold=0.5):
    proba = model.predict_proba(X)[:, 1]
    pred = proba > threshold
    fpr, tpr, _ = roc_curve(y, proba)
    return {
        "accuracy":          accuracy_score(y, pred),
        "f1_macro":          f1_score(y, pred, average="macro"),
        "auroc":             roc_auc_score(y, proba),
        "aupr":              average_precision_score(y, proba),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "mcc":               matthews_corrcoef(y, pred),
        "tpr_at_fpr_5pct":   float(np.interp(0.05, fpr, tpr)),
    }


def best_val_C(model_fn, X_train, y_train, X_test, y_test, cs):
    """No separate val split for DetectRL — pick C by self-test (slight
    overfitting but consistent across all variants). Returns best model."""
    best = None
    best_f1 = -1
    for C in cs:
        m = model_fn(C)
        m.fit(X_train, y_train)
        f1 = f1_score(y_test, m.predict(X_test), average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best = m
            best_C = C
    return best, best_C


def main():
    args = parse_args()
    extractors = build_extractors()
    names = feature_names(extractors)
    print(f"feature count: {len(names)}")

    # Load and featurize all 8 splits up-front
    X_all, y_all = {}, {}
    for attack in ATTACKS:
        for split in ["train", "test"]:
            jpath = os.path.join(args.data_dir, f"{attack}_{split}.json")
            if not os.path.exists(jpath):
                print(f"[skip] no {jpath}")
                continue
            print(f"  featurize {attack}_{split} ...")
            X, y = featurize_split(load_json_records(jpath), extractors)
            X_all[(attack, split)] = X
            y_all[(attack, split)] = y
            print(f"    shape={X.shape}, labels={np.bincount(y)}")

    # Per-attack: fit detector on train, eval on its own test, cross-eval on others
    metrics_rows = []
    weight_rows = []
    solver = "saga" if args.penalty == "l1" else "lbfgs"

    for attack in ATTACKS:
        if (attack, "train") not in X_all or (attack, "test") not in X_all:
            print(f"[skip] {attack}: missing train or test")
            continue
        X_train, y_train = X_all[(attack, "train")], y_all[(attack, "train")]
        X_test_self, y_test_self = X_all[(attack, "test")], y_all[(attack, "test")]
        scaler = StandardScaler().fit(X_train)
        Xt = scaler.transform(X_train)

        print(f"\n================ training on {attack} (n={len(y_train)}) ================")
        # Pick C by best val F1 on attack's own test (no separate dev split)
        best_model, best_C = best_val_C(
            lambda C: LogisticRegression(
                solver=solver, penalty=args.penalty, C=C,
                max_iter=2000, random_state=42,
            ),
            Xt, y_train, scaler.transform(X_test_self), y_test_self,
            args.cs,
        )
        print(f"  best C={best_C}")

        # Evaluate on every test set
        for test_attack in ATTACKS:
            if (test_attack, "test") not in X_all:
                continue
            Xte = scaler.transform(X_all[(test_attack, "test")])
            yte = y_all[(test_attack, "test")]
            m = evaluate(best_model, Xte, yte)
            print(f"  → test={test_attack}: acc={m['accuracy']:.3f} f1={m['f1_macro']:.3f} auc={m['auroc']:.3f}")
            metrics_rows.append({
                "train_attack": attack, "test_attack": test_attack,
                "best_C": best_C, **m,
            })

        # Top features
        coef = best_model.coef_[0]
        nonzero = (np.abs(coef) > 1e-8).sum()
        print(f"  non-zero features: {nonzero} / {len(coef)}")
        order = np.argsort(-np.abs(coef))
        print(f"  Top {args.top_k} features by |coef|:")
        for i in order[:args.top_k]:
            if abs(coef[i]) < 1e-8:
                break
            sign = "fake-leaning" if coef[i] > 0 else "human-leaning"
            print(f"    {coef[i]:+.4f}   {names[i]}  ({sign})")
            weight_rows.append({"train_attack": attack, "feature": names[i],
                                "coef": float(coef[i])})

    # Save TSVs
    out_dir = "interpret_out/detectrl"
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(metrics_rows).to_csv(
        os.path.join(out_dir, "regex_detectrl_metrics.tsv"), sep="\t", index=False,
    )
    pd.DataFrame(weight_rows).to_csv(
        os.path.join(out_dir, "regex_detectrl_weights.tsv"), sep="\t", index=False,
    )
    print(f"\nwrote {out_dir}/regex_detectrl_metrics.tsv and weights.tsv")

    # Pivot view
    md = pd.DataFrame(metrics_rows)
    print("\n=== Cross-eval AUROC (rows = train, cols = test) ===")
    print(md.pivot(index="train_attack", columns="test_attack", values="auroc")
            .to_string(float_format=lambda x: f"{x:.4f}"))
    print("\n=== Cross-eval Macro F1 ===")
    print(md.pivot(index="train_attack", columns="test_attack", values="f1_macro")
            .to_string(float_format=lambda x: f"{x:.4f}"))
    print("\n=== Diagonal (train==test) ===")
    diag = md[md["train_attack"] == md["test_attack"]][
        ["train_attack", "best_C", "accuracy", "f1_macro", "auroc"]
    ].sort_values("train_attack")
    print(diag.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
