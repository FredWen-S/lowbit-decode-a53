#!/usr/bin/env python3
"""实验二:每档 E = (P_run - P_idle) / tg。
power.csv 行格式:
  S,uptime,current_now(uA),voltage_now(uV),capacity,temp(0.1C),status
  MARK,uptime,label
P_idle = idle 段最后 40 s 的平均功率;P_run = run 段去掉前 15 s(加载+预热)的平均功率。
tg 取 llama-bench 的 avg_ts;同一档跑两遍(正序/倒序),bench_$q.csv 里按顺序有两行。
"""
import csv, os, sys, json, statistics

D = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 sys.argv[1] if len(sys.argv) > 1 else "energy")
QUANTS = ["q3_k_m", "q4_0", "q4_k_m", "q5_k_m", "q8_0"]
SIZE_MIB = {"q3_k_m": 876, "q4_0": 1011, "q4_k_m": 1066, "q5_k_m": 1226, "q8_0": 1807}
IDLE_TAIL, RUN_SKIP = 40.0, 15.0

samples, marks = [], []
for row in csv.reader(open(f"{D}/power.csv")):
    if row[0] == "S" and len(row) >= 7:
        t, i, v = float(row[1]), int(row[2]), int(row[3])
        samples.append((t, abs(i) * v * 1e-12, int(row[5]) / 10, row[6].strip()))
    elif row[0] == "MARK":
        marks.append((float(row[1]), row[2].strip()))

bad = [s for s in samples if s[0] >= next(t for t, l in marks if l == "unplugged")
       and s[3] != "Discharging"]
if bad:
    print(f"WARN {len(bad)} samples not Discharging after unplug", file=sys.stderr)
# 采样间隔异常(系统挂起)检查
gaps = [b[0] - a[0] for a, b in zip(samples, samples[1:]) if b[0] - a[0] > 3]
if gaps:
    print(f"WARN {len(gaps)} sampling gaps >3s, max {max(gaps):.1f}s", file=sys.stderr)

def mean_p(t0, t1):
    ps = [p for t, p, _, _ in samples if t0 <= t < t1]
    return statistics.mean(ps), len(ps)

def temp_at(t0, t1):
    ts = [c for t, _, c, _ in samples if t0 <= t < t1]
    return max(ts) if ts else None

segs = {}   # q -> list of dict per pass
for k, (t, label) in enumerate(marks):
    if label.startswith("idle_"):
        q = label[5:]
        t_run = next(tt for tt, l in marks[k:] if l == f"run_{q}")
        t_end = next(tt for tt, l in marks[k:] if l == f"end_{q}")
        p_idle, n_i = mean_p(t_run - IDLE_TAIL, t_run)
        p_run, n_r = mean_p(t_run + RUN_SKIP, t_end)
        segs.setdefault(q, []).append(dict(p_idle=p_idle, p_run=p_run, n_idle=n_i, n_run=n_r,
                                           dur=t_end - t_run, tmax=temp_at(t_run, t_end)))

res = {}
for q in QUANTS:
    # 两遍 >> 追加,第二遍的表头会混成一行数据,跳过
    rows = [r for r in csv.DictReader(open(f"{D}/bench_{q}.csv"))
            if r.get("avg_ts") and r["avg_ts"] != "avg_ts"]
    for p, r in zip(segs[q], rows):
        p["tg"] = float(r["avg_ts"])
        p["dP"] = p["p_run"] - p["p_idle"]
        p["J_tok"] = p["dP"] / p["tg"]
        p["J_tok_total"] = p["p_run"] / p["tg"]
    agg = {k: statistics.median(p[k] for p in segs[q])
           for k in ["p_idle", "p_run", "dP", "tg", "J_tok", "J_tok_total"]}
    agg["J_tok_passdiff"] = abs(segs[q][0]["J_tok"] - segs[q][1]["J_tok"])
    agg["passes"] = segs[q]
    res[q] = agg

print(f"{'quant':8} {'MiB':>5} {'P_idle W':>8} {'P_run W':>8} {'dP W':>6} {'tg':>5} "
      f"{'J/tok':>6} {'(diff)':>7} {'J/tok whole':>11} {'Tmax C':>6}")
for q in QUANTS:
    r = res[q]
    print(f"{q:8} {SIZE_MIB[q]:>5} {r['p_idle']:>8.2f} {r['p_run']:>8.2f} {r['dP']:>6.2f} "
          f"{r['tg']:>5.2f} {r['J_tok']:>6.3f} {r['J_tok_passdiff']:>7.3f} "
          f"{r['J_tok_total']:>11.3f} {max(p['tmax'] for p in r['passes']):>6.1f}")
json.dump(res, open(os.path.join(os.path.dirname(D), "energy_result.json"), "w"), indent=2)
