#!/usr/bin/env python3
"""simpleperf record 按符号拆分:每档 cycles/instructions 落在
点积内核(ggml_vec_dot_q*:解包+乘加)、线程同步(ggml_graph_compute_thread,ggml_barrier 已内联其中,
另含少量调度循环,所以是同步开销的上限)和其余部分的比例。
record 的事件总数还包含模型加载和预热,所以每 token 绝对值 = 采样比例 × 表 1 的每 token 总数
(counters_result_stat2.json),不用事件总数直接除 token 数。"""
import csv, json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "rec")
QUANTS = ["q3_k_m", "q4_0", "q4_k_m", "q5_k_m", "q8_0"]
TOT = json.load(open(os.path.join(HERE, "counters_result_stat2.json")))   # 表 1:每 token instr / cycles
res = {}
for q in QUANTS:
    ev, out = None, {}
    for line in open(f"{R}/{q}_report.txt"):
        if line.startswith("Event:"):
            ev = line.split()[1].split(":")[0]; out[ev] = {"total": 0, "sym": {}}
        elif line.startswith("Event count:"):
            out[ev]["total"] = int(line.split()[-1])
        m = re.match(r"\s*([\d.]+)%\s+(\S.*)", line)
        if m and ev:
            out[ev]["sym"][m.group(2).strip()] = float(m.group(1))
    b = list(csv.DictReader(open(f"{R}/{q}_bench.csv")))[-1]
    r = {"tg": float(b["avg_ts"]), "tg_cv": float(b["stddev_ts"]) / float(b["avg_ts"])}
    for ev, d in out.items():
        dot = {s: p for s, p in d["sym"].items() if s.startswith("ggml_vec_dot_q")}
        sync = sum(p for s, p in d["sym"].items() if s in ("ggml_graph_compute_thread", "ggml_barrier"))
        tot = TOT[q]["instr" if ev == "instructions" else "cycles"]
        r[ev] = {"dot_pct": sum(dot.values()), "sync_pct": sync,
                 "other_pct": 100 - sum(dot.values()) - sync, "dot_kernels": dot,
                 "dot_per_tok_M": tot * sum(dot.values()) / 100 / 1e6,
                 "sync_per_tok_M": tot * sync / 100 / 1e6}
    i, c = r["instructions"], r["cpu-cycles"]
    r["dot_ipc"] = i["dot_per_tok_M"] / c["dot_per_tok_M"]
    r["sync_ipc"] = i["sync_per_tok_M"] / c["sync_per_tok_M"]
    res[q] = r
print(f"{'quant':8} {'tg':>5} {'cv%':>4} | {'cyc dot%':>8} {'sync%':>6} | {'ins dot%':>8} {'sync%':>6} | "
      f"{'dot Mins/tok':>12} {'sync Mins/tok':>13} {'dot IPC':>7} {'sync IPC':>8} | kernels (instr %)")
for q in QUANTS:
    r = res[q]; c, i = r["cpu-cycles"], r["instructions"]
    ks = ", ".join(f"{k.replace('ggml_vec_dot_','')} {v:.1f}" for k, v in sorted(i["dot_kernels"].items(), key=lambda x: -x[1]) if v > 0.5)
    print(f"{q:8} {r['tg']:>5.2f} {r['tg_cv']*100:>4.1f} | {c['dot_pct']:>8.1f} {c['sync_pct']:>6.1f} | "
          f"{i['dot_pct']:>8.1f} {i['sync_pct']:>6.1f} | {i['dot_per_tok_M']:>12.0f} {i['sync_per_tok_M']:>13.0f} "
          f"{r['dot_ipc']:>7.2f} {r['sync_ipc']:>8.2f} | {ks}")
d = {q: res[q]["instructions"] for q in ("q3_k_m", "q8_0")}
print(f"Q3_K_M - Q8_0: total {(TOT['q3_k_m']['instr'] - TOT['q8_0']['instr']) / 1e6:.0f} M, "
      f"dot {d['q3_k_m']['dot_per_tok_M'] - d['q8_0']['dot_per_tok_M']:.0f} M, "
      f"sync {d['q3_k_m']['sync_per_tok_M'] - d['q8_0']['sync_per_tok_M']:.0f} M")
json.dump(res, open(os.path.join(os.path.dirname(R), "symbols_result.json"), "w"), indent=2)
