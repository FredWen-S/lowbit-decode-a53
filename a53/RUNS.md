# How each data folder was produced

Device: Sonim XP8800 (Snapdragon 630, 8× Cortex-A53, Android 10). Everything runs from
`/data/local/tmp/llm` over `adb shell`, with the llama.cpp `23deec0` arm64 build and the five
Qwen2.5-1.5B-Instruct GGUF files next to it. Before any counter run:

```sh
adb shell setprop security.perf_harden 0      # resets on reboot
adb shell cat /proc/sys/kernel/perf_event_paranoid
```

Thread count, `-n`, `--poll` and load mode below are read from the `llama-bench` CSV metadata
stored with each run. The CSVs do not record `-r`; it comes from the driver script, the stored
`simpleperf` command line, or (for `stat2/`) the report's method section.

| Folder | Driver | llama-bench flags | Used for |
|---|---|---|---|
| `stat/` | `counters.sh` | `-p 0 -n {32,160} -r 3 -t 8` (default `--poll 50`, mmap on) | First pass. Only the "single default-flag runs" in report §4, correction 1 |
| `stat2/` | not kept (see below) | `-p 0 -n {32,160} -r 3 -t 8 --poll 0 -lm none` | Table 1 (clean counter runs) |
| `rec/` | `simpleperf record`, command line stored in each `*_report.txt` | `-p 0 -n 64 -r 2 -t 8 --poll 0 -lm none` | Table 2 (symbol attribution) |
| `energy/` | `energy.sh` (USB unplugged) | `-p 0 -n 256 -r 3 -t 8 --poll 0 -lm none` | Table 3 (J/token) |

**`stat2/` driver.** The script for the clean pass was not kept. From the stored files it was
`counters.sh` with `--poll 0 -lm none` added and a separate bench CSV per event group
(`*_A_bench.csv`, `*_B_bench.csv`); the report adds that the page cache was pre-warmed. Pass `r1` ran
q4_k_m → q3_k_m → q8_0 → q5_k_m → q4_0 and pass `r2` ran the reverse order, which the CSV
timestamps confirm.

Event groups (≤ 4 events each, so no multiplexing):

- A: `cpu-cycles,instructions,raw-l1-dcache-refill,raw-l2-dcache-refill`
- B: `cpu-cycles,instructions,raw-bus-access`

## Regenerating the results

Pull the folders back with `adb pull`, then from this directory:

```sh
python analyze_counters.py stat2   # -> counters_result_stat2.json (Table 1)
python analyze_energy.py           # -> energy_result.json (Table 3)
python analyze_symbols.py          # -> symbols_result.json (Table 2)
python plot_results.py             # -> fig1_counters.png, fig2_energy.png (needs matplotlib)
python make_pdf.py                 # summary.md -> summary.pdf (needs markdown + headless Chrome)
```

The three analysis scripts use only the standard library. Rerunning them on the committed data
reproduces the committed JSON files exactly.

`counters_result.json` is the first pass (`stat/`) summarised by an earlier version of
`analyze_counters.py`; the current script no longer reads that file layout.
