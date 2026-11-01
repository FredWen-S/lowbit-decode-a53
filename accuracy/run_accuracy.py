#!/usr/bin/env python3
"""Task accuracy and WikiText-2 perplexity for the five quants, via llama-perplexity.

Accuracy does not depend on the device, so this runs on a desktop CPU build.
Every run keeps its full log in logs/; results.json is rebuilt from the logs.

usage:
  python run_accuracy.py --exe PATH/llama-perplexity --models DIR --data DIR [--threads 8]
  python run_accuracy.py --parse-only        # rebuild results.json from logs/
"""
import argparse
import json
import math
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOGS = HERE / "logs"
QUANTS = ["q3_k_m", "q4_0", "q4_k_m", "q5_k_m", "q8_0"]
REF = "q8_0"

# name -> (extra args, data file); task counts are fixed so every quant sees the same items
TESTS = {
    "wikitext2": (["-c", "512", "--chunks", "100"], "wikitext-2-raw/wiki.test.raw"),
    "hellaswag": (["--hellaswag", "--hellaswag-tasks", "400", "-c", "512"], "hellaswag_val_full.txt"),
    "winogrande": (["--winogrande", "--winogrande-tasks", "1267", "-c", "512"], "winogrande-debiased-eval.csv"),
    "mmlu": (["--multiple-choice", "-c", "1024"], "mmlu-validation.bin"),
}


def parse(test, text):
    """Return (score, error) from a llama-perplexity log; score is PPL or accuracy in %."""
    if test == "wikitext2":
        m = re.search(r"Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)", text)
    elif test == "hellaswag":
        rows = re.findall(r"^(\d+)\t([\d.]+)%\t\[([\d.]+)%, ([\d.]+)%\]", text, re.M)
        if not rows:
            return None
        n, acc, lo, hi = rows[-1]
        return dict(score=float(acc), ci95=[float(lo), float(hi)], tasks=int(n))
    elif test == "winogrande":
        m = re.search(r"Final Winogrande score\((\d+) tasks\): ([\d.]+) \+/- ([\d.]+)", text)
        return m and dict(score=float(m[2]), stderr=float(m[3]), tasks=int(m[1]))
    else:
        m = re.search(r"Final result: ([\d.]+) \+/- ([\d.]+)", text)
    return m and dict(score=float(m[1]), stderr=float(m[2]))


def items(test, text):
    """Per-item values from a log: chunk NLL for wikitext2, 0/1 correctness for the tasks.

    llama-perplexity prints running values, so each item is recovered by differencing
    i * running_mean(i) - (i-1) * running_mean(i-1).
    """
    if test == "wikitext2":
        running = [math.log(float(p)) for p in re.findall(r"\[\d+\]([\d.]+)", text)]
    elif test == "winogrande":
        return [int(ans == pred) for ans, pred in re.findall(r"^\d+\t[\d.]+\t\s*\S+\s+\S+\s+(\d)\s+(\d)\s*$", text, re.M)]
    else:
        running = [float(a) / 100 for a in re.findall(r"^\d+\t([\d.]+)%?(?:\t|$)", text, re.M)]
    out, prev = [], 0.0
    for i, r in enumerate(running, 1):
        out.append(i * r - prev)
        prev = i * r
    return out if test == "wikitext2" else [round(v) for v in out]


def paired(test, a, b):
    """Mean of a - b over shared items and its standard error (wikitext2: % change in PPL)."""
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    mean = sum(d) / n
    se = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1) / n)
    if test == "wikitext2":
        return dict(delta_pct=round(100 * math.expm1(mean), 2), se_pct=round(100 * se * math.exp(mean), 2), n=n)
    return dict(delta_pp=round(100 * mean, 2), se_pp=round(100 * se, 2), n=n)


def run(a):
    LOGS.mkdir(exist_ok=True)
    data = Path(a.data)
    for quant in QUANTS:
        model = Path(a.models) / f"qwen2.5-1.5b-instruct-{quant}.gguf"
        for test, (extra, fname) in TESTS.items():
            log = LOGS / f"{test}_{quant}.log"
            if log.exists() and parse(test, log.read_text(encoding="utf-8", errors="replace")):
                continue  # already done
            flag = "-bf" if fname.endswith(".bin") else "-f"
            cmd = [a.exe, "-m", str(model), flag, str(data / fname), "-t", str(a.threads), "-b", "512"] + extra
            print(" ".join(cmd), flush=True)
            out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            log.write_text(" ".join(cmd) + "\n\n" + out.stdout + out.stderr, encoding="utf-8")
            if out.returncode != 0:
                raise SystemExit(f"{test} {quant} failed, see {log}")


def collect():
    results, per_item = {}, {}
    for quant in QUANTS:
        for test in TESTS:
            log = LOGS / f"{test}_{quant}.log"
            if log.exists():
                text = log.read_text(encoding="utf-8", errors="replace")
                r = parse(test, text)
                if r:
                    results.setdefault(quant, {})[test] = r
                    per_item[quant, test] = items(test, text)
    # every quant sees the same items, so differences against Q8_0 are paired
    for quant in QUANTS[:-1]:
        for test in TESTS:
            if (quant, test) in per_item and (REF, test) in per_item:
                results[quant][test]["vs_q8_0"] = paired(test, per_item[quant, test], per_item[REF, test])
    (HERE / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8", newline="\n")
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exe")
    p.add_argument("--models")
    p.add_argument("--data")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--parse-only", action="store_true")
    a = p.parse_args()
    if not a.parse_only:
        if not (a.exe and a.models and a.data):
            p.error("--exe, --models and --data are required unless --parse-only")
        run(a)
    for quant, r in collect().items():
        print(quant, {k: v["score"] for k, v in r.items()})


if __name__ == "__main__":
    main()
