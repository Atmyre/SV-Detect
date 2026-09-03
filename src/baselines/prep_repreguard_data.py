"""Convert DetectRL and MIRAGE JSONs into RepreGuard's expected input format.

RepreGuard's process_data expects a list of items each with fields:
  - direct_prompt  : the LGT (LLM-generated) text
  - human_text     : the corresponding HWT (human-written) text

For MIRAGE this is a trivial rename (original->human_text, rewritten->direct_prompt).
For DetectRL we zip 'human' and 'llm' rows from the same file (no semantic pairing
needed for repe direction fitting — the paper doesn't require it either).

Also emits train JSONs that match SV-Detect's training setup:
  - mirage_train_all.json (500 polish + 150 gen + 150 rewrite = 800 pairs)
  - mirage_train_polish_only.json (500 pairs)
"""

import argparse
import json
import os
import random
from pathlib import Path


MIRAGE_TRAIN_FILES = {
    "polish":   "ai_detection_500_polish.raw_data.json",
    "generate": "xsum_generation_gpt-3.5-turbo.raw_data.json",
    "rewrite":  "xsum_rewrite_gpt-3.5-turbo.raw_data.json",
}


def mirage_train(mirage_train_dir: str, subset: str = "all") -> list[dict]:
    files = ([MIRAGE_TRAIN_FILES[subset]] if subset in MIRAGE_TRAIN_FILES
             else list(MIRAGE_TRAIN_FILES.values()))
    out = []
    for f in files:
        d = json.load(open(os.path.join(mirage_train_dir, f)))
        for h, l in zip(d["original"], d["rewritten"]):
            out.append({"direct_prompt": l, "human_text": h})
    return out


def mirage_test(mirage_root: str, scenario: str, task: str) -> list[dict]:
    """Read raw_texts/{DIG,SIG}/{task}.json which is list[{original,rewritten,...}]."""
    d = json.load(open(os.path.join(mirage_root, "raw_texts", scenario, f"{task}.json")))
    return [{"direct_prompt": r["rewritten"], "human_text": r["original"]} for r in d]


def detectrl(json_path: str, seed: int = 42) -> list[dict]:
    """Read a DetectRL split (flat text/label list), pair human & llm by shuffle+zip."""
    rows = json.load(open(json_path))
    human = [r["text"] for r in rows if r["label"] == "human"]
    llm   = [r["text"] for r in rows if r["label"] == "llm"]
    rng = random.Random(seed)
    rng.shuffle(human); rng.shuffle(llm)
    n = min(len(human), len(llm))
    return [{"direct_prompt": llm[i], "human_text": human[i]} for i in range(n)]


def _write(out_path: str, data: list[dict]):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f)
    print(f"  wrote {out_path}  n={len(data)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mirage-train-dir", required=True)
    p.add_argument("--mirage-root",      required=True)
    p.add_argument("--detectrl-benchmark-dir", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    # MIRAGE train (both variants)
    _write(f"{args.out_dir}/mirage_train_all.json",         mirage_train(args.mirage_train_dir, "all"))
    _write(f"{args.out_dir}/mirage_train_polish_only.json", mirage_train(args.mirage_train_dir, "polish"))

    # MIRAGE test (6 files)
    for scen in ("DIG", "SIG"):
        for task in ("generate", "polish", "rewrite"):
            _write(f"{args.out_dir}/mirage_test_{scen}_{task}.json",
                   mirage_test(args.mirage_root, scen, task))

    # DetectRL Multi-Domain (4 subsets × train+test)
    md = f"{args.detectrl_benchmark_dir}/Multi_Domain"
    for sub in ("arxiv", "writing_prompt", "xsum", "yelp_review"):
        for split in ("train", "test"):
            _write(f"{args.out_dir}/detectrl_md_{sub}_{split}.json",
                   detectrl(f"{md}/multi_domains_{sub}_{split}.json"))

    # DetectRL Multi-LLM (4 subsets × train+test)
    ml = f"{args.detectrl_benchmark_dir}/Multi_LLM"
    for sub in ("ChatGPT", "Claude-instant", "Google-PaLM", "Llama-2-70b"):
        for split in ("train", "test"):
            _write(f"{args.out_dir}/detectrl_mllm_{sub}_{split}.json",
                   detectrl(f"{ml}/multi_llms_{sub}_{split}.json"))

    # DetectRL Multi-Attack train/test (main 4 attacks)
    subsets = [
        ("direct_prompt",           "Direct_Prompt/direct_prompt"),
        ("prompt_attacks_llm",      "Prompt_Attacks/prompt_attacks_llm"),
        ("paraphrase_attacks_llm",  "Paraphrase_Attacks/paraphrase_attacks_llm"),
        ("perturbation_attacks_llm","Perturbation_Attacks/perturbation_attacks_llm"),
    ]
    for label, stem in subsets:
        for split in ("train", "test"):
            _write(f"{args.out_dir}/detectrl_matt_{label}_{split}.json",
                   detectrl(f"{args.detectrl_benchmark_dir}/{stem}_{split}.json"))


if __name__ == "__main__":
    main()
