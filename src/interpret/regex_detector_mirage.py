"""Mirage-specific regex detector.

Mirage records are paired (human `original` + LLM `rewritten`). For each of the
6 attack types ({DIG, SIG} × {generate, polish, rewrite}) we build a single
test dataset of (text, label) pairs (each record contributes one real and one
fake), then 80/20 split for train/eval.

Feature buckets (carried from DetectRL + Mirage-specific additions):

  -- formal connectives (LLM polishing tends to add these) --
  formal_connectives:    Furthermore, Moreover, Additionally, However, Nevertheless,
                          Consequently, Therefore, Subsequently, Notably, Specifically
  hedging_pro:           "It is worth noting", "It should be noted", "One should consider"
  closing_phrases:       "in summary", "in conclusion", "to summarize", "overall"

  -- politeness / register --
  professional_phrases:   "professional writing", "high-quality", "coherent", "engaging"
  greeting_polite:        "Dear", "Best regards", "Kind regards", "Sincerely"
  pronoun_we_us:          "\\bwe (?:can|will|should|believe)|\\bour (?:findings|approach)"

  -- LLM punctuation markers --
  em_dash:                "—" (em dash, LLMs use heavily)
  semicolon:              ";"
  excess_commas:          rough comma density per 100 words

  -- structural --
  oxford_list:            "X, Y, and Z"
  compound_VP_and:        common verbs followed by "and"
  numbered_list:          "1. ", "2. " sentence starts
  bullet_list:            "* " or "- " line starts
  markdown_bold:          "\\*\\*[^*]+\\*\\*"
  markdown_header:        sentence-start "## " or "# "

  -- DetectRL legacy (keep for cross-comparison) --
  llm_recommend_phrases, llm_assistant_template, paraphrase_polished,
  paraphrase_research_open, prompt_sentence_count, prompt_human_style,
  midword_case_flip, restaurant_template, casual_hedge, space_period,
  inline_newline, ai_cliche, ellipsis_three_dots, log_charlen
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


SOURCES = ["DIG", "SIG"]
TASKS = ["generate", "polish", "rewrite"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True,
                   help="Folder with raw_texts/{DIG,SIG}/{generate,polish,rewrite}.json")
    p.add_argument("--cs", nargs="+", type=float,
                   default=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
    p.add_argument("--penalty", default="l1", choices=["l1", "l2"])
    p.add_argument("--top-k", type=int, default=12)
    p.add_argument("--train-frac", type=float, default=0.8,
                   help="Fraction of records used for fitting; rest for eval.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-tsv-dir", default="interpret_out/mirage")
    return p.parse_args()


def build_extractors():
    f = OrderedDict()

    # --- formal connectives ---
    f["formal_connectives"] = re.compile(
        r"\b(?:Furthermore|Moreover|Additionally|However|Nevertheless|"
        r"Consequently|Therefore|Subsequently|Notably|Specifically|"
        r"Indeed|In addition|Likewise)\b",
    )
    f["hedging_pro"] = re.compile(
        r"\b(?:It is worth noting|It should be noted|One should consider|"
        r"It (?:is|might be) (?:important|essential|crucial) to|"
        r"It is interesting to note)\b",
        re.IGNORECASE,
    )
    f["closing_phrases"] = re.compile(
        r"\b(?:in summary|in conclusion|to summarize|to conclude|all in all|"
        r"overall,)\b",
        re.IGNORECASE,
    )

    # --- politeness / register ---
    f["professional_phrases"] = re.compile(
        r"\b(?:professional writing|high[- ]quality|coherent and engaging|"
        r"comprehensive (?:guide|overview|analysis)|cutting[- ]edge|"
        r"state[- ]of[- ]the[- ]art|seamless integration)\b",
        re.IGNORECASE,
    )
    f["greeting_polite"] = re.compile(
        r"\b(?:Dear (?:Sir|Madam|Customer|Team|Mr|Ms|Dr)|"
        r"Best (?:regards|wishes)|Kind regards|Sincerely yours?|Yours faithfully)\b",
        re.IGNORECASE,
    )
    f["pronoun_inclusive"] = re.compile(
        r"\b(?:we (?:can|will|should|believe|propose|present)|"
        r"our (?:findings|approach|results|analysis|study))\b",
        re.IGNORECASE,
    )

    # --- LLM punctuation markers ---
    f["em_dash"] = re.compile(r"—")
    f["semicolon"] = re.compile(r";")
    f["unicode_curly_quote"] = re.compile(r"[“”‘’]")  # smart quotes

    # --- structural ---
    f["oxford_list"] = re.compile(r"\b\w+,\s+\w+,\s+(?:and|or)\s+\w+", re.IGNORECASE)
    f["compound_VP_and"] = re.compile(
        r"\b(?:provide|develop|encourage|support|help|create|address|enhance|"
        r"promote|build|foster|highlight|include|emphasize|enable|empower|"
        r"deliver|achieve|incorporate|gain|explore|strengthen|maximize|"
        r"navigate|demonstrate|illustrate|explain|describe)\b"
        r"[\w\s,]{1,40}\sand\s\w+",
        re.IGNORECASE,
    )
    f["numbered_list"] = re.compile(r"(?:^|\n)\s*\d+\.\s+\w", re.MULTILINE)
    f["bullet_list"] = re.compile(r"(?:^|\n)\s*[-*•]\s+\w", re.MULTILINE)
    f["markdown_bold"] = re.compile(r"\*\*[^*\n]{1,40}\*\*")
    f["markdown_header"] = re.compile(r"(?:^|\n)#{1,3}\s+\w", re.MULTILINE)

    # --- DetectRL register signals ---
    f["llm_recommend_phrases"] = re.compile(
        r"\b(?:won'?t be disappointed|highly recommend|would (?:not )?hesitate to recommend)\b",
        re.IGNORECASE,
    )
    f["llm_assistant_template"] = re.compile(
        r"\b(?:As a helpful|Here is (?:a|an)|story based on the (?:writing )?prompt|"
        r"\d+ sentence (?:story|paragraph|continuation|article))\b",
        re.IGNORECASE,
    )
    f["paraphrase_polished"] = re.compile(
        r"\*?\*?Polished (?:Abstract|Version|Paragraph|Text)\*?\*?",
        re.IGNORECASE,
    )
    f["paraphrase_research_open"] = re.compile(
        r"\bIn this (?:paper|study|work|research|article)(?:,)? we "
        r"(?:explore|investigate|propose|present|study|examine|introduce|analyse|analyze)\b",
        re.IGNORECASE,
    )

    # --- common register / surface ---
    f["midword_case_flip"] = re.compile(r"\b[a-z]{2,}[A-Z][A-Za-z]{2,}\b")
    f["space_period"] = re.compile(r"\s\.")
    f["inline_newline"] = re.compile(r"[a-zA-Z]\n")
    f["ellipsis_three_dots"] = re.compile(r"\.\.\.")
    f["ellipsis_unicode"] = re.compile(r"…")

    casual_words = (
        r"\bbasically\b|\bprobably\b|\bpretty\b|\breally\b|\bactually\b|"
        r"\bmaybe\b|\bstuff\b|\bkinda\b|\bgonna\b|\bdunno\b|\bngl\b|\blol\b|"
        r"\bidk\b|\byea(?:h)?\b|\bnope\b|\blmao\b|\bdamn\b|\bsorta\b|\bhuh\b"
    )
    f["casual_hedge"] = re.compile(casual_words, re.IGNORECASE)
    f["ai_cliche"] = re.compile(
        r"\bin today's (?:rapidly )?changing world\b|"
        r"\bplays? a (?:crucial|vital|key|significant) role\b|"
        r"\bbring people together\b|"
        r"\b(?:must be|need to be|should be) addressed\b|"
        r"\bworld around (?:them|us)\b|"
        r"\bin a way that (?:highlights|emphasizes|reflects)\b",
        re.IGNORECASE,
    )

    return f


def featurize(text: str, extractors: dict) -> np.ndarray:
    counts = np.zeros(len(extractors), dtype=np.float32)
    for i, regex in enumerate(extractors.values()):
        counts[i] = len(regex.findall(text))
    return counts


def feature_names(extractors: dict) -> list:
    base = list(extractors.keys())
    return base + [f"{n}_per1k" for n in base] + ["log_charlen"]


def featurize_records(records, extractors):
    n_feats = len(extractors)
    rows = []
    for r in records:
        for cls, key in [(0, "original"), (1, "rewritten")]:
            text = r.get(key, "") or ""
            counts = featurize(text, extractors)
            char_len = max(1, len(text))
            row = np.zeros(2 * n_feats + 1, dtype=np.float32)
            row[:n_feats]            = counts
            row[n_feats:2 * n_feats] = 1000.0 * counts / char_len
            row[-1]                  = np.log1p(char_len)
            rows.append((row, cls))
    X = np.stack([r[0] for r in rows])
    y = np.array([r[1] for r in rows], dtype=np.int8)
    return X, y


def evaluate(model, scaler, X, y, threshold=0.5):
    proba = model.predict_proba(scaler.transform(X))[:, 1]
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


def main():
    args = parse_args()
    extractors = build_extractors()
    names = feature_names(extractors)
    print(f"feature count: {len(names)}")

    rng = np.random.default_rng(args.seed)
    metrics_rows = []
    weight_rows = []

    solver = "saga" if args.penalty == "l1" else "lbfgs"

    for source in SOURCES:
        for task in TASKS:
            jp = os.path.join(args.data_dir, "raw_texts", source, f"{task}.json")
            print(f"\n================ {source}/{task} ================")
            with open(jp) as fp:
                records = json.load(fp)
            print(f"  {len(records)} records ({2*len(records)} samples after pairing)")

            X_all, y_all = featurize_records(records, extractors)
            print(f"  features: {X_all.shape}")

            # 80/20 split by record (so paired samples stay together)
            n_records = len(records)
            idx = rng.permutation(n_records)
            cut = int(args.train_frac * n_records)
            tr_recs, te_recs = idx[:cut], idx[cut:]
            # Each record contributes 2 rows (real + fake); compute row indices
            tr_rows = np.concatenate([2 * tr_recs, 2 * tr_recs + 1])
            te_rows = np.concatenate([2 * te_recs, 2 * te_recs + 1])
            X_train, y_train = X_all[tr_rows], y_all[tr_rows]
            X_test,  y_test  = X_all[te_rows], y_all[te_rows]
            print(f"  train: {X_train.shape}, test: {X_test.shape}")

            scaler = StandardScaler().fit(X_train)
            Xt = scaler.transform(X_train)

            best_f1 = -1
            best_model = None
            best_C = None
            for C in args.cs:
                m = LogisticRegression(
                    solver=solver, penalty=args.penalty, C=C,
                    max_iter=2000, random_state=args.seed,
                )
                m.fit(Xt, y_train)
                f1 = f1_score(y_test, m.predict(scaler.transform(X_test)), average="macro")
                if f1 > best_f1:
                    best_f1 = f1
                    best_model = m
                    best_C = C
            metrics = evaluate(best_model, scaler, X_test, y_test)
            print(f"  best C={best_C}  test acc={metrics['accuracy']:.4f} "
                  f"f1={metrics['f1_macro']:.4f} auroc={metrics['auroc']:.4f}")
            metrics_rows.append({"source": source, "task": task,
                                 "best_C": best_C, **metrics})

            coef = best_model.coef_[0]
            order = np.argsort(-np.abs(coef))
            print(f"  Top {args.top_k} features by |coef|:")
            for i in order[:args.top_k]:
                if abs(coef[i]) < 1e-8: break
                sign = "fake-leaning" if coef[i] > 0 else "human-leaning"
                print(f"    {coef[i]:+.4f}   {names[i]}  ({sign})")
                weight_rows.append({"source": source, "task": task,
                                    "feature": names[i],
                                    "coef": float(coef[i])})

    os.makedirs(args.out_tsv_dir, exist_ok=True)
    pd.DataFrame(metrics_rows).to_csv(
        os.path.join(args.out_tsv_dir, "regex_mirage_metrics.tsv"), sep="\t", index=False,
    )
    pd.DataFrame(weight_rows).to_csv(
        os.path.join(args.out_tsv_dir, "regex_mirage_weights.tsv"), sep="\t", index=False,
    )
    print(f"\nwrote {args.out_tsv_dir}/regex_mirage_*.tsv")

    md = pd.DataFrame(metrics_rows)
    print("\n=== Mirage regex baseline summary ===")
    print(md[["source", "task", "best_C", "accuracy", "f1_macro", "auroc"]]
            .to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
