# Task accuracy and WikiText-2 perplexity

Accuracy does not depend on the device, so these runs used a desktop CPU build of `llama-perplexity` (llama.cpp `42916d8`, x86, 8 threads). The models are the same five GGUF files used on the phone, with SHA-256 hashes listed in the top-level README. Every quant ran on exactly the same items.

| Quant | WikiText-2 PPL | HellaSwag (400) | WinoGrande (1267) | MMLU (1548) |
|---|---|---|---|---|
| Q3_K_M | 11.64 (+12.5% ± 0.3) | 64.0 (−1.5 ± 1.1) | 61.4 (−3.1 ± 1.1) | 33.9 (**−2.2 ± 0.7**) |
| Q4_0 | 10.87 (+5.1% ± 0.3) | 64.5 (−1.0 ± 1.2) | 62.4 (−2.1 ± 1.0) | 34.9 (−1.1 ± 0.6) |
| Q4_K_M | 10.80 (+4.5% ± 0.2) | 65.3 (−0.3 ± 1.1) | 61.8 (−2.7 ± 1.0) | 35.9 (−0.2 ± 0.6) |
| Q5_K_M | 10.46 (+1.1% ± 0.1) | 66.8 (+1.3 ± 0.9) | 64.0 (−0.5 ± 0.7) | 36.2 (+0.1 ± 0.5) |
| Q8_0 | 10.34 | 65.5 | 64.5 | 36.0 |

- **How to read the table:** task scores are accuracy in %. Each value in brackets is the paired difference from Q8_0 on the same items, ± one standard error, in percentage points; for WikiText-2 it is the % change in perplexity.
- **Why paired:** the standard errors that `llama-perplexity` prints for each score are ±1.2–1.4 pp on MMLU and WinoGrande. That spread mostly reflects which items were sampled, and every quant saw the same items, so a paired comparison is much tighter.
- **How the per-item results were recovered:** `llama-perplexity` prints a running value after each item, so `run_accuracy.py` gets each item's result by differencing consecutive running values. As a check, the paired means match the differences between the final scores.

What it shows:

- **Q5_K_M matches Q8_0 everywhere.** Its only measurable cost is +1.1% perplexity.
- **The 4-bit quants lose little on tasks.** Q4_K_M is within noise on MMLU and HellaSwag. Both 4-bit quants lose about 2–3 pp on WinoGrande (about 2–3 SE).
- **Q3_K_M is the only quant with a clear MMLU loss** (−2.2 pp, about 3 SE). Its perplexity cost (+12.5%) is about 2.5× that of the 4-bit quants.
- **Caveat on significance:** 16 comparisons are made here. Read single 2-SE results, such as Q4_0 on WinoGrande, as suggestive rather than proven.

Data sources:

- [WikiText-2 raw](https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip): test split, first 100 chunks of 512 tokens.
- [HellaSwag validation](https://raw.githubusercontent.com/klosax/hellaswag_text_data/main/hellaswag_val_full.txt): first 400 tasks.
- [WinoGrande debiased eval](https://huggingface.co/datasets/ikawrakow/winogrande-eval-for-llama.cpp).
- [MMLU validation](https://huggingface.co/datasets/ikawrakow/validation-datasets-for-llama.cpp): the full 1548 tasks, using `--multiple-choice`. llama.cpp labels this "TruthfulQA score" in its log, but the data is MMLU.

To rerun:

```sh
python run_accuracy.py --exe PATH/llama-perplexity --models DIR --data DIR --threads 8
python run_accuracy.py --parse-only   # rebuild results.json from logs/ (CI does this)
```

Each full log is kept in `logs/`. On a 14-thread laptop CPU, one quant takes about 80 minutes.
