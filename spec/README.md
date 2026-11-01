# Speculative decoding on the A53

Target Qwen2.5-1.5B-Instruct Q4_K_M, draft Qwen2.5-0.5B-Instruct Q4_K_M, both `-t 4`, 128 new tokens, greedy.
Six prompts (2 code, 2 chat, 2 summary) × 3 repeats × 3 modes, 20 s cooldown between runs, phone on USB power.

- **baseline:** target alone.
- **fixed:** draft length 4.
- **adaptive:** draft length set each round by an EWMA of the recent acceptance rate.

All 18 prompt/repeat groups produced identical tokens in all three modes.

| Mode | Median decode (tok/s) | Acceptance | Mean draft length | Paired speed vs baseline |
|---|---|---|---|---|
| baseline | 3.55 | – | – | 1.00× |
| fixed (4) | 1.81 | 65.7% | 3.94 | 0.51× |
| adaptive | 1.77 | 71.8% | 2.99 | 0.50× |

Adaptive is 1.10× faster than fixed in the paired median: 1.25× on chat, where acceptance is low, and about 0.95× on code. Both speculative modes are still about half as fast as baseline.

## Why it cannot win here

Speculative decoding assumes that checking k+1 tokens in one target pass costs about the same as decoding one token, which holds when decode is memory-bound. On this core it is not (see `../a53/`). Timing on the same phone with `llama-bench`:

| Target batch (tokens) | 1 | 2 | 3 | 4 | 5 | 8 |
|---|---|---|---|---|---|---|
| Seconds | 0.286 | 0.484 | 0.659 | 0.900 | 1.038 | 1.558 |

A 5-token check costs 3.6× a single decode step. The draft model costs 0.097 s/token, which is 0.34× the target.

With **every** draft token accepted, the best speedup over baseline is 0.96–1.02× for draft lengths 1–7. Accepting only 66–72% of drafts, plus per-round overhead, gives the measured 0.5×. Tuning the controller cannot fix this. A faster verification kernel (int8 dot product, ARMv8.2+) or a much cheaper draft would be needed.

## Files and versions

- `runs.jsonl`: one line per run, with per-round `[drafted, accepted]` and the output token hash.
- `metadata.json`: command settings and SHA-256 hashes of the binary and models.
- `verify_cost_raw.csv`: raw `llama-bench` output (commit `23deec0`).
- The speculative runs used the `on-device-llm-lab` build of llama.cpp at commit `42916d8`, with the adaptive controller added.

`python summarize_spec.py` rebuilds `spec_summary.json`, and CI checks it.
