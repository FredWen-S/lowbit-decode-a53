#!/system/bin/sh
DIR=/data/local/tmp/llm; cd $DIR
OUT=$DIR/thermal.csv
echo "t_s,cpu0_mhz,cpu4_mhz,soc_temp_c,batt_ma,batt_temp_c,phase" > $OUT
sample(){
  f0=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq); f0=$((f0/1000))
  f4=$(cat /sys/devices/system/cpu/cpu4/cpufreq/scaling_cur_freq); f4=$((f4/1000))
  tmax=0
  for z in /sys/class/thermal/thermal_zone*; do
    ty=$(cat $z/type 2>/dev/null)
    case $ty in tsens_tz_sensor*) v=$(cat $z/temp 2>/dev/null); [ "$v" -gt "$tmax" ] 2>/dev/null && tmax=$v;; esac
  done
  tc=$((tmax/10))
  ma=$(cat /sys/class/power_supply/battery/current_now 2>/dev/null); ma=$((ma/1000))
  bt=$(cat /sys/class/power_supply/battery/temp 2>/dev/null); bt=$((bt/10))
  echo "$1,$f0,$f4,$tc,$ma,$bt,$2" >> $OUT
}
# 1) idle 基线 6 秒
S=0; for i in 1 2 3; do sample $S idle; S=$((S+2)); sleep 2; done
# 2) 持续推理负载
LD_LIBRARY_PATH=$DIR ./llama-cli -m qwen2.5-1.5b-instruct-q4_k_m.gguf \
   -t 8 -c 2048 -n 1400 --ignore-eos -st --no-display-prompt \
   -p "请详细讲述一次完整的多日徒步旅行经历,包括准备、路线、天气和见闻。" >/dev/null 2>&1 &
PID=$!
while kill -0 $PID 2>/dev/null; do sample $S load; S=$((S+2)); sleep 2; done
# 3) 冷却 20 秒
for i in 1 2 3 4 5 6 7 8 9 10; do sample $S cooldown; S=$((S+2)); sleep 2; done
echo "THERMAL_DONE total=${S}s"
