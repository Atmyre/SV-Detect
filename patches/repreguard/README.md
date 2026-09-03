# RepreGuard patch

`rep_reading_pipeline.patch` — patch against
[NLP2CT/RepreGuard](https://github.com/NLP2CT/RepreGuard) commit
`53677be` (the current tip of `main` as of 2026-01-27).

Applied to `repe/rep_reading_pipeline.py`. Three fixes are needed to run
RepreGuard against our benchmarks with a modern `transformers` release
and with backbones whose position embeddings are shorter than the
longest text in the benchmark (RAID). Without them, reproduction of
paper Appendix G (Section 4.2 in-text: SV-Detect vs. RepreGuard at a
matched backbone) fails.

## What the patch fixes

1. **`self.framework` → `"pt"`** *(one line, `preprocess`)*. The
   `Pipeline.framework` attribute was removed in `transformers` 5.x, so
   the upstream code raises `AttributeError` on that release. We pin
   the tensor return type to PyTorch directly.

2. **Truncation on long inputs** *(same line, `preprocess`)*. RAID
   texts extend to ~70 k characters, well past GPT-Neo-2.7B's 2048
   position-embedding limit. We add `truncation=True, max_length=2048`
   to the tokenizer call. Without this, forward passes crash on
   overflowing sequences.

3. **`.float()` before `.numpy()`** *(one line, `_forward`)*. bfloat16
   tensors cannot be converted with `.numpy()`; when the backbone is
   loaded in bf16 the pipeline crashes. We cast to fp32 before the
   NumPy conversion.

4. **`del hidden_states_batch` scope + indentation** *(one line,
   `_forward`)*. In upstream, the `del` statement is inside the
   `for batch in hidden_states_batch:` loop and uses a
   mixed-tabs-and-spaces indent that Python's parser accepts on some
   releases but not others. It also causes an `UnboundLocalError` on
   the second iteration of the outer loop, since the variable is
   deleted while still needed. Moving it out one level (and normalising
   the indent to spaces) fixes both issues at once.

## Applying the patch

From a fresh RepreGuard checkout:

```bash
git clone https://github.com/NLP2CT/RepreGuard.git
cd RepreGuard
git checkout 53677be   # pin to the commit we patched against
git apply /path/to/SV-Detect/patches/repreguard/rep_reading_pipeline.patch
```
