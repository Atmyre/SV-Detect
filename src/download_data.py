"""Download the English COLING-2025 MGT Task 1 jsonl files from the public
Drive folder shared by the task organizers.

Files (with labels):
  en_train.jsonl                 - training set (610k rows)
  en_dev.jsonl                   - dev set     (262k rows)
  test_set_en_with_label.jsonl   - held-out test set (74k rows)

Drive folder: https://drive.google.com/drive/folders/1Mz8vTnqi7truGrc05v6kWaod6mEK7Enj
"""

import argparse
import os
import subprocess
import sys

# (output_filename, Google Drive file ID, expected size in bytes for sanity)
FILES = [
    ("en_train.jsonl",               "1o8LE5p5xRdEFGrZOKiY4In2xW2BiWJbG", 1_003_385_472),
    ("en_dev.jsonl",                 "1hYIHqU3IMnJjPMTvl99K8pQUIOe7a957",   430_337_669),
    ("test_set_en_with_label.jsonl", "1zd6Q0kGIk5CcDMIKKGktnSSZ7Zh0iiPt",   238_442_945),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="./coling_data")
    p.add_argument("--force", action="store_true",
                   help="Re-download even if file exists with the right size")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    try:
        import gdown  # noqa: F401
    except ImportError:
        print("gdown not installed; running `pip install gdown`")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])

    for fname, fid, expected_size in FILES:
        out_path = os.path.join(args.out_dir, fname)
        if (not args.force) and os.path.exists(out_path) \
                and os.path.getsize(out_path) == expected_size:
            print(f"[skip] {fname} already present ({expected_size} bytes)")
            continue
        url = f"https://drive.google.com/uc?id={fid}"
        print(f"[download] {fname} <- {url}")
        subprocess.check_call(
            [sys.executable, "-m", "gdown", url, "-O", out_path]
        )
        actual = os.path.getsize(out_path)
        if actual != expected_size:
            print(f"[warn] size mismatch for {fname}: expected {expected_size}, got {actual}")

    print("all files present:")
    for fname, _, _ in FILES:
        p = os.path.join(args.out_dir, fname)
        print(f"  {p}  {os.path.getsize(p)} bytes")


if __name__ == "__main__":
    main()
