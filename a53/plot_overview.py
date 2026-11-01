#!/usr/bin/env python3
"""README overview: decode speed, energy and task accuracy against weight size.

Reads counters_result_stat2.json, energy_result.json and ../accuracy/results.json.
usage: python plot_overview.py   (needs matplotlib) -> fig0_overview.png
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
QUANTS = ["q3_k_m", "q4_0", "q4_k_m", "q5_k_m", "q8_0"]
LABEL = {"q3_k_m": "Q3_K_M", "q4_0": "Q4_0", "q4_k_m": "Q4_K_M", "q5_k_m": "Q5_K_M", "q8_0": "Q8_0"}
SIZE_MIB = {"q3_k_m": 876, "q4_0": 1011, "q4_k_m": 1060, "q5_k_m": 1220, "q8_0": 1801}
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "font.family": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 9, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
})

c = json.load(open(f"{HERE}/counters_result_stat2.json"))
e = json.load(open(f"{HERE}/energy_result.json"))
acc = json.load(open(f"{HERE}/../accuracy/results.json"))
x = [SIZE_MIB[q] for q in QUANTS]


# label placement: (dx, dy) in points, ha, va; anything not listed goes above the point
ABOVE = (0, 7, "center", "bottom")
BELOW = (0, -7, "center", "top")
LEFT = (-7, 0, "right", "center")


def panel(ax, ys, color, title, ylabel, fmt, errs=None, place=None, ref=None):
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.plot(x, ys, color=color, lw=2, marker="o", ms=6, mec=SURF, mew=1.5)
    if errs:
        ax.errorbar(x, ys, yerr=errs, fmt="none", ecolor=color, elinewidth=1, capsize=2)
    errs = errs or [0] * len(ys)
    for q, xi, yi, err in zip(QUANTS, x, ys, errs):
        dx, dy, ha, va = (place or {}).get(q, ABOVE)
        # anchor labels at the end of the error bar so they do not sit on it
        anchor = yi + err if va == "bottom" else yi - err if va == "top" else yi
        value = "(ref)" if q == ref else f"{yi:{fmt}}"
        text = f"{value}\n{LABEL[q]}" if va == "top" else f"{LABEL[q]}\n{value}"
        ax.annotate(text, (xi, anchor), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va=va, fontsize=7, color=INK)
    lo, hi = min(y - d for y, d in zip(ys, errs)), max(y + d for y, d in zip(ys, errs))
    pad = (hi - lo) * 0.3 or 1
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(580, 1950)
    ax.set_xlabel("weight size (MiB)"); ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", color=INK, fontsize=9.5)


fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), dpi=200)
# Q4_0 and Q4_K_M are only 49 MiB apart, so their labels go to opposite sides
panel(axes[0], [c[q]["tg160"] for q in QUANTS], BLUE,
      "Decode speed peaks at 4 bits", "A53 decode (tok/s)", ".2f",
      place={"q3_k_m": LEFT, "q4_0": LEFT, "q4_k_m": (6, 3, "left", "bottom")})
panel(axes[1], [e[q]["J_tok"] for q in QUANTS], ORANGE,
      "Energy follows time, not bytes", "incremental energy (J/token)", ".2f",
      place={"q4_0": (-4, -7, "right", "top"), "q4_k_m": (4, -7, "left", "top")})
# paired difference against Q8_0 on the same 1548 MMLU items, 95% interval
vs = [acc[q]["mmlu"].get("vs_q8_0", {"delta_pp": 0.0, "se_pp": 0.0}) for q in QUANTS]
panel(axes[2], [v["delta_pp"] for v in vs], AQUA, "MMLU holds at 4 bits, drops at Q3_K_M",
      "MMLU vs Q8_0 (pp, paired, 95% CI)", "+.1f", errs=[1.96 * v["se_pp"] for v in vs],
      place={"q3_k_m": LEFT, "q4_0": BELOW, "q5_k_m": (5, 3, "left", "bottom")}, ref="q8_0")
axes[2].axhline(0, color=INK2, lw=0.8, ls=(0, (3, 3)))
fig.tight_layout(w_pad=2.5)
fig.savefig(f"{HERE}/fig0_overview.png")
print("wrote fig0_overview.png")
