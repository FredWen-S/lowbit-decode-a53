#!/usr/bin/env python3
"""两张图:
fig1  横轴文件大小,纵轴 cycles/token、bus access/token(及 instructions/token),各自归一化到 Q8_0
fig2  各档位焦耳/token(实验二跑完后才有)
用法: .venv/bin/python plot_results.py
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
QUANTS = ["q3_k_m", "q4_0", "q4_k_m", "q5_k_m", "q8_0"]
LABEL = {"q3_k_m": "Q3_K_M", "q4_0": "Q4_0", "q4_k_m": "Q4_K_M", "q5_k_m": "Q5_K_M", "q8_0": "Q8_0"}
# 张量大小(llama-bench model_size / 2^20),与表 1、表 4 同一口径
SIZE_MIB = {"q3_k_m": 876, "q4_0": 1011, "q4_k_m": 1060, "q5_k_m": 1220, "q8_0": 1801}
# 参考调色板(dataviz palette.md)前三个分类槽 + 文字/网格色
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "font.family": "Helvetica", "font.size": 10, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
})

def fig1():
    c = json.load(open(f"{HERE}/counters_result_stat2.json"))
    x = [SIZE_MIB[q] for q in QUANTS]
    ref = c["q8_0"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=200)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.plot(x, [s / SIZE_MIB["q8_0"] for s in x], ls=(0, (4, 3)), color=INK2, lw=1,
            label="weight size (proportional)")
    series = [("bus", "bus access / token", BLUE),
              ("cycles", "cycles / token", ORANGE),
              ("instr", "instructions / token", AQUA)]
    for k, name, col in series:
        y = [c[q][k] / ref[k] for q in QUANTS]
        ax.plot(x, y, color=col, lw=2, marker="o", ms=7, mec=SURF, mew=2, label=name)
        ax.annotate(name, (x[0], y[0]), xytext=(-8, 0), textcoords="offset points",
                    ha="right", va="center", color=INK, fontsize=9)
    # Q4_0 与 Q4_K_M 只差 49 MiB,Q4_K_M 标签放到第二行
    for q in QUANTS:
        ax.annotate(LABEL[q], (SIZE_MIB[q], 0.42), xytext=(0, -11 if q == "q4_k_m" else 0),
                    textcoords="offset points", ha="center", va="top", fontsize=8, color=INK2)
    ax.set_xlim(560, 1880); ax.set_ylim(0.3, 1.55)
    ax.set_xlabel("model weight size (MiB)"); ax.set_ylabel("per-token count, normalised to Q8_0")
    ax.set_title("Bytes fall with bit-width; instructions and cycles do not",
                 loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="lower right", bbox_to_anchor=(1.0, 0.08))
    fig.tight_layout(); fig.savefig(f"{HERE}/fig1_counters.png"); plt.close(fig)

def fig2():
    p = f"{HERE}/energy_result.json"
    if not os.path.exists(p):
        print("energy_result.json 不存在,跳过 fig2"); return
    e = json.load(open(p))
    x = range(len(QUANTS))
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    y = [e[q]["J_tok"] for q in QUANTS]
    passes = [[pp["J_tok"] for pp in e[q]["passes"]] for q in QUANTS]
    ax.bar(x, y, width=0.5, color=BLUE)
    for i, ps in enumerate(passes):     # 正序/倒序两遍的值
        ax.plot([i] * len(ps), ps, ls="none", marker="o", ms=5, color=INK, mfc=SURF, mew=1.2)
        ax.annotate(f"{y[i]:.2f}", (i, max(ps)), xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=9, color=INK)
    ax.set_xticks(list(x), [f"{LABEL[q]}\n{SIZE_MIB[q]} MiB" for q in QUANTS])
    ax.set_ylabel("incremental energy (J / token)")
    ax.set_ylim(0, max(max(ps) for ps in passes) * 1.2)
    ax.set_title("Decode energy per token, whole phone on battery (bar = median, dots = two passes)",
                 loc="left", color=INK, fontsize=10)
    fig.tight_layout(); fig.savefig(f"{HERE}/fig2_energy.png"); plt.close(fig)

fig1(); fig2()
