#!/system/bin/sh
DIR=/data/local/tmp/llm; cd $DIR
for m in q3_k_m q4_0 q4_k_m q5_k_m q8_0; do
  R=$(LD_LIBRARY_PATH=$DIR ./llama-perplexity -m $DIR/qwen2.5-1.5b-instruct-$m.gguf \
       -f $DIR/ppl_zh.txt -t 8 -c 256 2>&1 | grep -i 'Final estimate' | tail -1)
  echo "$m : $R"
done
echo "ALL_PPL_DONE"
