#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
端侧 LLM 可复用测量框架
--------------------------------
对一个「配置」(模型 × 量化 × 线程 × 后端)做一次完整测量,统一记录:
  - 速度:llama-bench 的 pp / tg (tok/s)            [--bench]
  - 质量(客观):固定 12 题集,贪婪解码,自动判分  [--quality]
  - 质量(困惑度):llama-perplexity               [--ppl TEXT]
结果连同元数据(模型 sha256、构建标签、参数、时间戳)写成一份 JSON。

单变量原则:每次只改一个参数(量化 / 线程 / 平台),其余固定,便于横比。
支持两种后端:
  local —— 直接在本机(桌面)跑二进制
  adb   —— 通过 adb 在真机上跑(二进制与模型已在设备目录)

示例:
  # 桌面
  python3 bench_harness.py --backend local \
      --bin-dir .../build-mac/bin --model .../qwen2.5-1.5b-instruct-q4_k_m.gguf \
      --threads 4 --label desktop-q4km --prompts prompts.jsonl \
      --bench --ppl ppl_zh.txt --quality --out results/desktop-q4km.json
  # 真机
  python3 bench_harness.py --backend adb --serial XXXX \
      --remote-dir /data/local/tmp/llm --model qwen2.5-1.5b-instruct-q4_k_m.gguf \
      --threads 8 --label device-q4km --prompts prompts.jsonl \
      --bench --ppl ppl_zh.txt --quality --out results/device-q4km.json
仅用标准库。
"""
import argparse, json, subprocess, re, sys, time, hashlib, os, shlex

# ---------- 运行一条命令(本机或经 adb) ----------
def run_local(bin_dir, argv, timeout):
    env = dict(os.environ); env["LD_LIBRARY_PATH"] = bin_dir
    p = subprocess.run(argv, cwd=bin_dir, env=env, capture_output=True,
                       text=True, timeout=timeout)
    return p.stdout + "\n" + p.stderr

def run_adb(serial, remote_dir, remote_argv, timeout):
    inner = "cd %s && LD_LIBRARY_PATH=. %s" % (remote_dir, remote_argv)
    base = ["adb"] + (["-s", serial] if serial else []) + ["shell", inner]
    p = subprocess.run(base, capture_output=True, text=True, timeout=timeout)
    return p.stdout + "\n" + p.stderr

def exe(args, name, extra, timeout):
    """构造并运行一个 llama 工具,返回合并输出。"""
    if args.backend == "local":
        argv = [os.path.join(args.bin_dir, name)] + extra
        return run_local(args.bin_dir, argv, timeout)
    else:
        remote = "./%s %s" % (name, " ".join(shlex.quote(e) for e in extra))
        return run_adb(args.serial, args.remote_dir, remote, timeout)

# ---------- 输出清洗与判分 ----------
def extract_answer(raw):
    lines = raw.splitlines()
    start = 0
    for i, l in enumerate(lines):
        if l.lstrip().startswith(">"):
            start = i + 1; break
    body = []
    for l in lines[start:]:
        s = l.strip()
        if (s.startswith("[ Prompt") or s.startswith("Exiting")
                or s.startswith("llama_") or s.startswith("main:")):
            continue
        body.append(l)
    return "\n".join(body).strip()

def norm(s):
    return re.sub(r"\s+", "", str(s)).strip().lower()

def extract_json(text):
    for pat in (r"\{.*\}", r"\[.*\]"):
        m = re.search(pat, text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return None

def score(check, output):
    t = check["type"]
    if t == "exact":
        want = check["value"]
        return (norm(want) in norm(output)), None
    if t == "json":
        got = extract_json(output)
        return (got == check["value"]), None
    if t == "manual":
        return (None, check["rubric"])   # 需人工按 rubric 打分
    return (None, "unknown check type")

# ---------- 三个测量 ----------
def do_bench(args):
    out = exe(args, "llama-bench",
              ["-m", args.model, "-t", str(args.threads),
               "-p", "128", "-n", "64", "-r", "2"], timeout=1200)
    res = {}
    for line in out.splitlines():
        m = re.search(r"\|\s*(pp128|tg64)\s*\|\s*([0-9.]+)\s*±", line)
        if m:
            res[m.group(1)] = float(m.group(2))
    return res

def do_ppl(args):
    out = exe(args, "llama-perplexity",
              ["-m", args.model, "-f", args.ppl, "-c", "256",
               "-t", str(args.threads)], timeout=1800)
    m = re.search(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)", out)
    return float(m.group(1)) if m else None

def do_quality(args, prompts):
    rows, auto_total, auto_pass = [], 0, 0
    for item in prompts:
        out = exe(args, "llama-cli",
                  ["-m", args.model, "-t", str(args.threads),
                   "-c", "512", "-n", "96", "-st", "--no-display-prompt",
                   "--temp", "0", "-p", item["prompt"]], timeout=600)
        ans = extract_answer(out)
        ok, note = score(item["check"], ans)
        if ok is not None:
            auto_total += 1
            auto_pass += 1 if ok else 0
        rows.append({"id": item["id"], "pass": ok, "answer": ans[:400],
                     "note": note})
    return {"auto_total": auto_total, "auto_pass": auto_pass,
            "auto_accuracy": round(auto_pass / auto_total, 3) if auto_total else None,
            "items": rows}

# ---------- 元数据 ----------
def model_sha256(args):
    if args.backend == "local" and os.path.exists(args.model):
        h = hashlib.sha256()
        with open(args.model, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        return h.hexdigest()
    if args.backend == "adb":
        base = ["adb"] + (["-s", args.serial] if args.serial else [])
        p = subprocess.run(base + ["shell",
            "sha256sum %s/%s" % (args.remote_dir, args.model)],
            capture_output=True, text=True)
        m = re.match(r"([0-9a-f]{64})", p.stdout.strip())
        return m.group(1) if m else None
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["local", "adb"], required=True)
    ap.add_argument("--bin-dir", default="")          # local: 二进制与 .so 所在目录
    ap.add_argument("--remote-dir", default="/data/local/tmp/llm")  # adb: 设备目录
    ap.add_argument("--serial", default="")
    ap.add_argument("--model", required=True)         # local: 路径;adb: 目录内文件名
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--prompts", default="prompts.jsonl")
    ap.add_argument("--ppl", default="")              # 留出文本(目标可访问)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--quality", action="store_true")
    ap.add_argument("--build", default="")            # 构建标签/提交
    ap.add_argument("--note", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prompts = []
    if args.quality:
        with open(args.prompts, encoding="utf-8") as f:
            prompts = [json.loads(l) for l in f if l.strip()]

    rec = {
        "label": args.label,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {"backend": args.backend, "model": args.model,
                   "threads": args.threads, "build": args.build,
                   "note": args.note},
        "model_sha256": model_sha256(args),
    }
    if args.bench:
        print("[bench] ...", file=sys.stderr); rec["bench"] = do_bench(args)
    if args.ppl:
        print("[ppl] ...", file=sys.stderr); rec["ppl"] = do_ppl(args)
    if args.quality:
        print("[quality] 12 题 ...", file=sys.stderr)
        rec["quality"] = do_quality(args, prompts)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)

    # 控制台摘要
    print("== %s ==" % args.label)
    if "bench" in rec: print("  speed:", rec["bench"])
    if "ppl" in rec:   print("  ppl  :", rec["ppl"])
    if "quality" in rec:
        q = rec["quality"]
        print("  12题自动判分: %d/%d  (准确率 %s)" %
              (q["auto_pass"], q["auto_total"], q["auto_accuracy"]))
    print("  -> %s" % args.out)

if __name__ == "__main__":
    main()
