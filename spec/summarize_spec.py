#!/usr/bin/env python3
"""Summarize the paired speculative-decoding runs on the A53 phone.

Reads runs.jsonl (baseline / fixed draft length 4 / adaptive, same prompt and repeat
for each mode) and verify_cost_raw.csv (llama-bench batch timings), writes spec_summary.json.
usage: python summarize_spec.py
"""
import csv
import json
import os
from collections import defaultdict
from statistics import median

HERE = os.path.dirname(os.path.abspath(__file__))
MODES = ["baseline", "fixed", "adaptive"]

runs = [json.loads(line) for line in open(f"{HERE}/runs.jsonl", encoding="utf-8")]
by_key = defaultdict(dict)
for r in runs:
    by_key[(r["prompt_id"], r["repeat"])][r["mode"]] = r

# outputs must match across modes, otherwise the speed comparison is meaningless
identical = all(len({m["tokens_sha256"] for m in g.values()}) == 1 for g in by_key.values())


def stats(rows):
    tok_s = [r["decode_tok_s"] for r in rows]
    out = dict(runs=len(rows), tok_s_median=round(median(tok_s), 3))
    drafted = [r for r in rows if r["proposed"]]
    if drafted:
        proposed = sum(r["proposed"] for r in drafted)
        out["acceptance"] = round(sum(r["accepted"] for r in drafted) / proposed, 3)
        rounds = [d for r in drafted for d, _ in r["rounds"]]
        out["mean_draft_len"] = round(sum(rounds) / len(rounds), 2)
    return out


summary = {"identical_outputs": identical, "all": {}, "by_task": {}}
for mode in MODES:
    summary["all"][mode] = stats([r for r in runs if r["mode"] == mode])
for task in sorted({r["task"] for r in runs}):
    summary["by_task"][task] = {m: stats([r for r in runs if r["mode"] == m and r["task"] == task]) for m in MODES}

ratios = defaultdict(list)
for g in by_key.values():
    for m in ("fixed", "adaptive"):
        ratios[f"{m}_vs_baseline"].append(g[m]["decode_tok_s"] / g["baseline"]["decode_tok_s"])
    ratios["adaptive_vs_fixed"].append(g["adaptive"]["decode_tok_s"] / g["fixed"]["decode_tok_s"])
summary["paired_speedup_median"] = {k: round(median(v), 3) for k, v in ratios.items()}

# Ceiling from llama-bench on the same phone: one round drafts k tokens, then the target
# checks k+1 tokens in one batch. With every draft accepted it yields k+1 tokens.
bench = list(csv.DictReader(open(f"{HERE}/verify_cost_raw.csv", encoding="utf-8")))
target = {int(r["n_prompt"]): int(r["n_prompt"]) * 1e9 / float(r["avg_ns"]) for r in bench
          if "1.5b" in r["model_filename"] and r["n_gen"] == "0"}  # tok/s for a batch of n
t_one = 1 / target[1]
t_draft = min(float(r["avg_ns"]) / 1e9 / int(r["n_gen"]) for r in bench
              if "0.5b" in r["model_filename"] and r["n_gen"] != "0")
summary["verify_cost"] = {
    "target_s_per_batch": {n: round(n / ts, 3) for n, ts in sorted(target.items())},
    "draft_s_per_token": round(t_draft, 4),
    "ceiling_speedup_full_accept": {
        k: round((k + 1) * t_one / (k * t_draft + (k + 1) / target[k + 1]), 3)
        for k in range(1, 8) if k + 1 in target},
}

with open(f"{HERE}/spec_summary.json", "w", encoding="utf-8", newline="\n") as f:
    f.write(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
