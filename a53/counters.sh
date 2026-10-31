#!/system/bin/sh
cd /data/local/tmp/llm
export LD_LIBRARY_PATH=.
EVA=cpu-cycles,instructions,raw-l1-dcache-refill,raw-l2-dcache-refill
EVB=cpu-cycles,instructions,raw-bus-access
rm -rf stat; mkdir -p stat
for rep in 1 2; do
  for q in q4_k_m q3_k_m q8_0 q5_k_m q4_0; do
    for n in 32 160; do
      /system/bin/simpleperf stat -e $EVA --csv -o stat/${q}_n${n}_r${rep}_A.csv \
        ./llama-bench -m qwen2.5-1.5b-instruct-${q}.gguf -p 0 -n $n -r 3 -t 8 -o csv \
        > stat/${q}_n${n}_r${rep}_bench.csv 2>/dev/null
      sleep 20
      /system/bin/simpleperf stat -e $EVB --csv -o stat/${q}_n${n}_r${rep}_B.csv \
        ./llama-bench -m qwen2.5-1.5b-instruct-${q}.gguf -p 0 -n $n -r 3 -t 8 -o csv \
        > /dev/null 2>&1
      sleep 20
    done
  done
done
echo DONE > stat/done.txt
