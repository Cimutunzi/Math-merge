nohup bash /data/qq/math-evaluation-harness/scripts/stage_3.sh > /data/qq/math-evaluation-harness/log_saves/Qwen2_1.5b_ins/gsm8k/stage_3.log 2>&1 &
wait $!
nohup bash /data/qq/math-evaluation-harness/scripts/stage_2.sh > /data/qq/math-evaluation-harness/log_saves/Qwen2_1.5b_ins/gsm8k/stage_2.log 2>&1 &
wait $!
nohup bash /data/qq/math-evaluation-harness/scripts/stage_1.sh > /data/qq/math-evaluation-harness/log_saves/Qwen2_1.5b_ins/gsm8k/stage_1.log 2>&1 &