#!/usr/bin/env python3
"""实验一:由 n=32 / n=160 两次计数做差,得到纯解码的每 token 指标。
X_per_token = (X_n160 - X_n32) / (128 * 3)
A 组: cpu-cycles, instructions, raw-l1-dcache-refill, raw-l2-dcache-refill
B 组: cpu-cycles, instructions, raw-bus-access

干净判据:llama-bench 3 次重复的 tg 变异系数 < 2%。
被干扰的运行里有线程被抢占,其余线程在 ggml_barrier 上忙等,
会凭空多出大量高 IPC(~1.0)的自旋指令;访存计数基本不受影响。
cycles/instructions 只用干净运行;每个 (n160, n32) 干净组合各做一次差,取中位数和极差。
"""
import csv, os, sys, json, statistics, itertools

STAT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    sys.argv[1] if len(sys.argv) > 1 else "stat2")
QUANTS = ["q3_k_m", "q4_0", "q4_k_m", "q5_k_m", "q8_0"]
SIZE_MIB = {"q3_k_m": 876, "q4_0": 1011, "q4_k_m": 1066, "q5_k_m": 1226, "q8_0": 1807}
TOK = 128 * 3
CV_MAX = 0.02

def read_stat(path):
    out = {}
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[0].strip().isdigit():
                out[row[1].strip().split(":")[0]] = int(row[0])
                # 覆盖率检查
                cov = [c for c in row if c.strip().startswith("(")]
                if cov and cov[0].strip() != "(100%)":
                    print("WARN multiplexed:", path, row[1], cov[0], file=sys.stderr)
    return out

def read_bench(path):
    with open(path) as f:
        r = list(csv.DictReader(f))[-1]
    return float(r["avg_ts"]), float(r["stddev_ts"])

def runs(q, n):
    out = []
    for rep in (1, 2):
        for g in "AB":
            base = f"{STAT}/{q}_n{n}_r{rep}_{g}"
            ts, sd = read_bench(base + "_bench.csv")
            out.append(dict(read_stat(base + ".csv"), group=g, rep=rep, ts=ts,
                            clean=sd / ts < CV_MAX))
    return out

def diff(hi_runs, lo_runs, k):
    vals = [(h[k] - l[k]) / TOK for h, l in itertools.product(hi_runs, lo_runs)]
    return statistics.median(vals), min(vals), max(vals)

res = {}
for q in QUANTS:
    hi, lo = runs(q, 160), runs(q, 32)
    hc, lc = [r for r in hi if r["clean"]], [r for r in lo if r["clean"]]
    r = {"n_clean": f"{len(hc)}/4+{len(lc)}/4"}
    for k, name in [("instructions", "instr"), ("cpu-cycles", "cycles")]:
        r[name], r[name + "_lo"], r[name + "_hi"] = diff(hc, lc, k)
    # 访存计数:A/B 各自只有一组,用干净运行(若没有就退回全部)
    for k, name, g in [("raw-l1-dcache-refill", "l1d", "A"),
                       ("raw-l2-dcache-refill", "l2d", "A"),
                       ("raw-bus-access", "bus", "B")]:
        h = [x for x in hc if x["group"] == g] or [x for x in hi if x["group"] == g]
        l = [x for x in lc if x["group"] == g] or [x for x in lo if x["group"] == g]
        r[name], r[name + "_lo"], r[name + "_hi"] = diff(h, l, k)
    r["ipc"] = r["instr"] / r["cycles"]
    r["tg160"] = statistics.median(x["ts"] for x in hc)
    r["bus_GB"] = r["bus"] * 16 / 1e9      # A53 总线 128-bit,每次访问 16 B
    r["l2d_GB"] = r["l2d"] * 64 / 1e9      # 缓存行 64 B
    r["dram_GBps"] = r["bus_GB"] * r["tg160"]
    res[q] = r

ref = res["q8_0"]
print(f"{'quant':8} {'MiB':>5} {'clean':>7} {'Minstr/tok':>17} {'Mcyc/tok':>17} {'IPC':>5} "
      f"{'L1D M/tok':>9} {'L2D M/tok':>9} {'bus M/tok':>9} {'bus GB':>6} {'ins/Q8':>6} "
      f"{'cyc/Q8':>6} {'bus/Q8':>6} {'size/Q8':>7} {'tg160':>6} {'GB/s':>5}")
for q in QUANTS:
    r = res[q]
    rng = lambda k: f"{r[k]/1e6:5.0f} [{r[k+'_lo']/1e6:4.0f}-{r[k+'_hi']/1e6:4.0f}]"
    print(f"{q:8} {SIZE_MIB[q]:>5} {r['n_clean']:>7} {rng('instr'):>17} {rng('cycles'):>17} "
          f"{r['ipc']:>5.2f} {r['l1d']/1e6:>9.2f} {r['l2d']/1e6:>9.2f} {r['bus']/1e6:>9.1f} "
          f"{r['bus_GB']:>6.2f} {r['instr']/ref['instr']:>6.2f} {r['cycles']/ref['cycles']:>6.2f} "
          f"{r['bus']/ref['bus']:>6.2f} {SIZE_MIB[q]/SIZE_MIB['q8_0']:>7.2f} "
          f"{r['tg160']:>6.2f} {r['dram_GBps']:>5.1f}")
json.dump(res, open(os.path.join(os.path.dirname(STAT),
          f"counters_result_{os.path.basename(STAT)}.json"), "w"), indent=2)
