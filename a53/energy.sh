#!/system/bin/sh
# 实验二:拔掉 USB 后测整机功率,E = (P_run - P_idle) / tg
# 配置与实验一一致:--poll 0、-lm none、预热进页缓存
cd /data/local/tmp/llm
export LD_LIBRARY_PATH=.
B=/sys/class/power_supply/battery
rm -rf energy; mkdir -p energy
mark() { echo "MARK,$(cut -d' ' -f1 /proc/uptime),$1" >> energy/power.csv; }
# 屏幕超时 5 分钟;注入一个 Shift 键刷新用户活动计时,不改任何设置
poke() { input keyevent 59; }
( while true; do
    echo "S,$(cut -d' ' -f1 /proc/uptime),$(cat $B/current_now),$(cat $B/voltage_now),$(cat $B/capacity),$(cat $B/temp),$(cat $B/status)" >> energy/power.csv
    sleep 0.5
  done ) &
SAMPLER=$!
# 等到真正拔线(状态变成 Discharging),再静置 30 s
mark wait_unplug
while [ "$(cat $B/status)" != "Discharging" ]; do poke; sleep 5; done
mark unplugged; sleep 30
for q in q4_k_m q3_k_m q8_0 q5_k_m q4_0 q4_0 q5_k_m q8_0 q3_k_m q4_k_m; do
  M=qwen2.5-1.5b-instruct-$q.gguf
  cat $M > /dev/null
  poke; mark idle_$q; sleep 60
  poke; mark run_$q
  ./llama-bench -m $M -p 0 -n 256 -r 3 -t 8 --poll 0 -lm none -o csv >> energy/bench_$q.csv 2>/dev/null
  mark end_$q
done
mark finished
kill $SAMPLER
echo DONE > energy/done.txt
