# Data directory

This directory is intentionally empty in the repository. Datasets and
intermediate artefacts produced by the pipeline are gitignored.

## What goes here at runtime

After running the pipeline you will see roughly:

```
data/
├── DetectRL/                  # raw DetectRL JSONL (from arxiv 2410.23746)
├── MIRAGE/
│   ├── raw_texts/{DIG,SIG}/*.json
│   └── ...
├── coling_data/               # COLING-2025 MGT English jsonl files
├── activations/
│   ├── gpt-neo-2.7B/
│   │   ├── DetectRL/  *_activations_*.npy
│   │   ├── MIRAGE/
│   │   └── COLING_2025_MGT_en/
│   └── Llama-2-7b-hf/
└── svs/
    ├── gpt-neo-2.7B/
    │   ├── DetectRL/    steering_vectors_<method>.npy
    │   │                <split>_dot_products_<method>.npy
    │   ├── MIRAGE/
    │   └── COLING_2025_MGT_en/
    └── Llama-2-7b-hf/COLING_2025_MGT_en/
```

## Downloading

`src/download_data.py` fetches the public HuggingFace mirrors:

```bash
python -m src.download_data --benchmark detectrl --out-dir data/DetectRL
python -m src.download_data --benchmark mirage   --out-dir data/MIRAGE
python -m src.download_data --benchmark coling   --out-dir data/coling_data
```

For COLING the Codabench test-set labels are not on HuggingFace; download
`test_set_en_with_label.jsonl` separately from the shared-task
organisers and place it under `data/coling_data/`.

## Disk footprint

Activations are large — the full `gpt-neo-2.7B/COLING_2025_MGT_en/`
extraction is ~100 GB (610,767 samples × 32 layers × 2560 dim ×
float32). Plan storage accordingly. The downstream dot-products
(`(N, L)` per sample) are much smaller (~80 MB per split).
