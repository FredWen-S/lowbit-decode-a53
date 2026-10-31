#!/usr/bin/env python3
"""生成 t4_crosscheck.ipynb:Colab 上「运行时 → 全部运行」即可。
模型从 Hugging Face 下载(与手机上的文件 sha256 逐字节一致,已核对),不需要挂载 Drive。"""
import json
SHA = """58cb5c05ecef48e82961f1a2be6544145ea26136f69dddda4bbbd092f0e4b993  qwen2.5-1.5b-instruct-q3_k_m.gguf
dcd819ff094852c38faba6873d8ff0c9d51eadb2844539e52042ae5d647bbfdb  qwen2.5-1.5b-instruct-q4_0.gguf
6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e  qwen2.5-1.5b-instruct-q4_k_m.gguf
b46661073c18e5b56a41fa320975f866a00def1ff08feef4718e013258896f8c  qwen2.5-1.5b-instruct-q5_k_m.gguf
d7efb072e7724d25048a4fda0a3e10b04bdef5d06b1403a1c93bd9f1240a63c8  qwen2.5-1.5b-instruct-q8_0.gguf"""

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})

md("""# T4 cross-check: is the A53 slowdown really unpacking?
Pre-registered predictions (written 2026-10-30, before running): **A** bandwidth-bound (BW max/min < 1.15 and tg falls monotonically with file size) · **B** overhead-bound (tg max/min < 1.10) · **C** still unpacking-bound (large BW spread and Q3_K_M slower than Q4_K_M).

Runtime → Change runtime type → **T4 GPU**, then Runtime → **Run all**. About 30–40 min, mostly the build. No Drive mount needed; all results are printed in the last cell.""")

code("""%%bash
# 1. GPU check — must be a Tesla T4
nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,clocks.max.mem,power.limit --format=csv | tee /content/gpu_env.txt
grep -q "Tesla T4" /content/gpu_env.txt || { echo "NOT A T4 — change runtime type"; exit 1; }""")

code(f"""%%bash
# 2. Same five GGUFs as the phone: download from Hugging Face, verify sha256 (byte-identical to the phone's copies)
mkdir -p /content/models && cd /content/models
for q in q3_k_m q4_0 q4_k_m q5_k_m q8_0; do
  f=qwen2.5-1.5b-instruct-$q.gguf
  [ -f $f ] || wget -q https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/$f
done
cat > sha.txt <<'EOF'
{SHA}
EOF
sha256sum -c sha.txt""")

code("""%%bash
# 3. Build llama-bench at the same commit as the phone (23deec0), CUDA, sm_75 only
cd /content
[ -d llama.cpp ] || git clone -q https://github.com/ggml-org/llama.cpp
cd llama.cpp && git checkout -q 23deec0 && git log -1 --oneline
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DLLAMA_CURL=OFF > /content/cmake.log 2>&1 || { tail -30 /content/cmake.log; exit 1; }
cmake --build build -j$(nproc) --target llama-bench > /content/build.log 2>&1 || { tail -40 /content/build.log; exit 1; }
ls -l build/bin/llama-bench""")

code("""%%bash
# 4. Sanity: must report CUDA / Tesla T4, and tg far above single digits
cd /content/llama.cpp && ./build/bin/llama-bench -m /content/models/qwen2.5-1.5b-instruct-q3_k_m.gguf -ngl 99 -p 0 -n 32 -r 1 2>&1 | grep -E "CUDA|T4|tg32"
""")

code("""# 5. Experiment 1: decode only, all layers on GPU, forward then reverse order, -n 128 -r 5.
# A quant whose cv > 2% is re-run (up to 3 tries). GPU clocks/power logged every 0.5 s.
import subprocess, csv, io, os, time
B = "/content/llama.cpp/build/bin/llama-bench"
M = "/content/models/qwen2.5-1.5b-instruct-{}.gguf"
Q = ["q3_k_m", "q4_0", "q4_k_m", "q5_k_m", "q8_0"]
os.makedirs("/content/t4", exist_ok=True)
smi = subprocess.Popen("nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu "
                       "--format=csv,noheader,nounits -lms 500 > /content/t4/gpu.csv", shell=True)
marks = []
def bench(q):
    out = subprocess.run([B, "-m", M.format(q), "-ngl", "99", "-p", "0", "-n", "128", "-r", "5", "-o", "csv"],
                         capture_output=True, text=True, check=True).stdout
    r = list(csv.DictReader(io.StringIO(out)))[-1]
    return out, float(r["avg_ts"]), float(r["stddev_ts"])
for p, L in [("fwd", Q), ("rev", Q[::-1])]:
    for q in L:
        for attempt in range(3):
            t0 = time.time()
            out, ts, sd = bench(q)
            marks.append((p, q, attempt + 1, t0, time.time()))
            open(f"/content/t4/{p}_{q}.csv", "w").write(out)
            print(f"{p} {q:7} try {attempt+1}: {ts:7.1f} tok/s  cv {sd/ts*100:4.1f}%")
            if sd / ts < 0.02:
                break
        time.sleep(10)
smi.terminate()
open("/content/t4/marks.txt", "w").write("\\n".join(" ".join(map(str, m)) for m in marks))""")

code("""%%bash
# 6. Experiment 2 (optional): nsys kernel breakdown for Q3_K_M, Q4_K_M, Q8_0. Skipped if nsys is unavailable.
NSYS=$(which nsys || ls /usr/local/cuda/bin/nsys 2>/dev/null || ls /opt/nvidia/nsight-systems/*/bin/nsys 2>/dev/null | head -1)
if [ -z "$NSYS" ]; then
  PKG=$(apt-cache search nsight-systems 2>/dev/null | awk '{print $1}' | grep -E '^nsight-systems' | sort | tail -1)
  [ -n "$PKG" ] && apt-get install -y -qq $PKG > /dev/null 2>&1
  NSYS=$(which nsys || ls /opt/nvidia/nsight-systems/*/bin/nsys 2>/dev/null | head -1)
fi
if [ -z "$NSYS" ]; then echo "nsys not available — skipping experiment 2"; exit 0; fi
echo "using $NSYS"; $NSYS --version
cd /content/llama.cpp
for q in q3_k_m q4_k_m q8_0; do
  $NSYS profile -t cuda --cuda-graph-trace=node -f true -o /content/t4/nsys_$q \\
    ./build/bin/llama-bench -m /content/models/qwen2.5-1.5b-instruct-$q.gguf -ngl 99 -p 0 -n 128 -r 1 > /content/t4/nsys_$q.log 2>&1
  $NSYS stats --report cuda_gpu_kern_sum --format csv -o /content/t4/kern_$q /content/t4/nsys_$q.nsys-rep > /dev/null 2>&1
  ls /content/t4/kern_$q* 2>/dev/null
done""")

code("""# 7. Analysis + verdict (criteria fixed before running)
import os, glob, json, pandas as pd
A53 = {"q3_k_m": 5.38, "q4_0": 6.31, "q4_k_m": 5.67, "q5_k_m": 5.26, "q8_0": 4.56}   # USB, counter runs
rows = []
for q in Q:
    r = [pd.read_csv(f"/content/t4/{p}_{q}.csv").iloc[-1] for p in ("fwd", "rev")]
    tg = sum(x.avg_ts for x in r) / 2
    size = os.path.getsize(M.format(q))
    rows.append(dict(quant=q, MiB=round(size / 2**20), a53_tg=A53[q], t4_tg=round(tg, 1),
                     GBps=round(size * tg / 1e9, 1), pct_of_320=round(size * tg / 3.2e9, 1),
                     cv_pct=round(max(x.stddev_ts / x.avg_ts for x in r) * 100, 1),
                     fwd_rev_gap_pct=round(abs(r[0].avg_ts - r[1].avg_ts) / tg * 100, 1)))
d = pd.DataFrame(rows).sort_values("MiB")
# Experiment 2 columns, if present
for q in d.quant:
    f = glob.glob(f"/content/t4/kern_{q}*cuda_gpu_kern_sum*.csv")
    if not f: continue
    k = pd.read_csv(f[0]); tot = k["Total Time (ns)"].sum()
    mmv = k[k["Name"].str.contains("mul_mat_vec|mmvq", case=False)]["Total Time (ns)"].sum()
    tg = float(d.loc[d.quant == q, "t4_tg"].iloc[0]); wall = 129 / tg * 1e9
    d.loc[d.quant == q, "gpu_busy_pct"] = round(tot / wall * 100, 1)
    d.loc[d.quant == q, "mmv_share_pct"] = round(mmv / tot * 100, 1)
    d.loc[d.quant == q, "kernel_GBps"] = round(os.path.getsize(M.format(q)) * 129 / mmv, 1) if mmv else None
print(open("/content/gpu_env.txt").read())
print(d.to_string(index=False))
bw = d.GBps.max() / d.GBps.min(); spread = d.t4_tg.max() / d.t4_tg.min()
mono = d.t4_tg.is_monotonic_decreasing
q3_slower = float(d.loc[d.quant == "q3_k_m", "t4_tg"].iloc[0]) < float(d.loc[d.quant == "q4_k_m", "t4_tg"].iloc[0])
bad = d[(d.cv_pct > 2) | (d.fwd_rev_gap_pct > 3)].quant.tolist()
if bw < 1.15 and mono: verdict = "A (bandwidth-bound)"
elif spread < 1.10: verdict = "B (overhead-bound)"
elif q3_slower: verdict = "C (still unpacking-bound)"
else: verdict = "undetermined"
g = pd.read_csv("/content/t4/gpu.csv", header=None, names=["ts", "sm", "mem", "W", "C"])
half = len(g) // 2
print(f"\\nBW max/min = {bw:.2f}   tg max/min = {spread:.2f}   tg monotonic decreasing: {mono}   Q3_K_M slower than Q4_K_M: {q3_slower}")
print(f"SM clock MHz 1st half {g.sm[:half].mean():.0f} / 2nd half {g.sm[half:].mean():.0f};  power max {g.W.max():.1f} W;  temp max {g.C.max():.0f} C")
print("NEEDS RERUN (cv>2% or fwd/rev gap>3%):", bad or "none")
print("VERDICT:", verdict)
print("\\nJSON:", json.dumps(dict(table=d.to_dict("records"), bw_ratio=bw, tg_ratio=spread, monotonic=bool(mono),
                                 q3_slower=bool(q3_slower), rerun=bad, verdict=verdict,
                                 sm_clock=[g.sm[:half].mean(), g.sm[half:].mean()], power_max=g.W.max())))""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
      "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open("t4_crosscheck.ipynb", "w"), indent=1, ensure_ascii=False)
print("ok", len(cells), "cells")
