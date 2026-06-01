# SV-Detect

Code for *SV-Detect: Steering-Vector-Based Detection of Machine-Generated
Text*. SV-Detect represents each text as the layer-wise alignment of its
hidden-state activations with learned "real-vs-fake" steering vectors in
a frozen reference language model, then trains a lightweight logistic
regression on those alignment scores.

## Layout

```
sv-detect/
├── src/                       # Pipeline + analysis code
│   ├── extract/               # 1) activations -> SVs -> dot products
│   │   ├── extract_activations.py
│   │   ├── compute_steering_vectors.py
│   │   └── train_logreg.py
│   ├── classifier/            # 2) per-benchmark evaluation drivers
│   │   ├── analyze_detectrl.py
│   │   └── analyze_mirage.py
│   ├── interpret/             # 3) Section D interpretability code
│   │   ├── interpret_steering_vectors.py
│   │   ├── interpret_steering_ngrams.py
│   │   ├── interpret_steering_vectors_text.py
│   │   ├── regex_detector.py
│   │   ├── regex_detector_detectrl.py
│   │   ├── regex_detector_mirage.py
│   │   ├── compare_sae_dense.py
│   │   └── inspect_logreg_weights.py
│   ├── ablation/              # 4) classifier / construction ablations
│   │   └── compare_classifiers.py
│   ├── visualize_tokens.py    # Figure 1 per-token visualisation
│   └── download_data.py       # Helpers to fetch DetectRL / MIRAGE / COLING
├── data/                      # (Reserved for cached datasets; gitignored)
├── environment.yml            # Conda environment used in our experiments
├── requirements.txt           # Pip equivalent for non-conda users
└── README.md
```

## Quick start

### 1. Install

```bash
conda env create -f environment.yml
conda activate sv-detect
# or:
pip install -r requirements.txt
```

Required: Python ≥ 3.11, PyTorch 2.10+ with CUDA 12.8, transformers
4.57+, scikit-learn 1.8+.

### 2. Download benchmarks

```bash
python -m src.download_data --benchmark detectrl
python -m src.download_data --benchmark mirage
python -m src.download_data --benchmark coling   # optional
```

DetectRL and MIRAGE are pulled from their official HuggingFace mirrors.
You will need a HuggingFace token (`export HF_TOKEN=...`) to download
gated models such as `Llama-2-7b-hf`.

### 3. Extract activations

```bash
python -m src.extract.extract_activations \
    --llm EleutherAI/gpt-neo-2.7B \
    --data-root data/DetectRL \
    --out-dir data/activations/gpt-neo-2.7B/DetectRL \
    --splits real_train fake_train real_val fake_val test
```

Output: per-sample mean-pooled residuals as
`(N, num_layers, hidden_size)` `.npy` chunks. One forward per sample;
splits run in parallel as a SLURM array (see `scripts/slurm/run_extract.slurm`).

### 4. Compute steering vectors + dot-product features

```bash
python -m src.extract.compute_steering_vectors \
    --activations-dir data/activations/gpt-neo-2.7B/DetectRL \
    --methods mean pca logreg \
    --filters all woweak \
    --out-dir data/svs/gpt-neo-2.7B/DetectRL
```

This produces both the steering vectors `steering_vectors_<method>.npy`
(shape `(L, H)`) and per-sample dot-product features
`<split>_dot_products_<method>.npy` (shape `(N, L)`) for the downstream
classifier.

### 5. Train and evaluate the detector

```bash
# DetectRL (in-domain + Multi-Domain / Multi-LLM / Multi-Attack splits)
python -m src.classifier.analyze_detectrl \
    --svs-dir data/svs/gpt-neo-2.7B/DetectRL \
    --method logreg

# MIRAGE
python -m src.classifier.analyze_mirage \
    --svs-dir data/svs/gpt-neo-2.7B/MIRAGE \
    --tasks generate polish rewrite

# COLING-2025 MGT
python -m src.extract.train_logreg \
    --dots-dir data/svs/gpt-neo-2.7B/COLING_2025_MGT_en \
    --methods mean pca logreg \
    --filters all woweak woTGPT35weak
```

### 6. Interpretability (Section D)

```bash
# D.1 + D.2: per-layer attribution + classical logit-lens
python -m src.interpret.interpret_steering_vectors \
    --llm EleutherAI/gpt-neo-2.7B \
    --sv-path data/svs/gpt-neo-2.7B/MIRAGE/steering_vectors_logreg_generate.npy \
    --top-k 12 --out-tsv interpret_out/logit_lens_mirage_generate.tsv

# D.3: stylistic-feature baseline
python -m src.interpret.regex_detector_detectrl --data-root data/DetectRL
python -m src.interpret.regex_detector_mirage --data-root data/MIRAGE

# D.5 (Figure 1): per-token visualisation
python -m src.visualize_tokens \
    --sv-path data/svs/gpt-neo-2.7B/MIRAGE/steering_vectors_logreg_generate.npy \
    --raw-json data/MIRAGE/raw_texts/DIG/generate.json \
    --out-dir figures/teaser_pngs
```
