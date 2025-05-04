nohup bash /data/qq/math-evaluation-harness/scripts/Qwen2.5-Math-1.5B/run_eval_ins.sh > /data/qq/math-evaluation-harness/log_saves/Qwen2.5-Math-1.5B/math-merge/qwen_math_1.5b_level_3_step_3-4.log 2>&1 &
wait $!
nohup bash /data/qq/math-evaluation-harness/scripts/Qwen2.5-Math-1.5B/run_eval_ins_1.sh > /data/qq/math-evaluation-harness/log_saves/Qwen2.5-Math-1.5B/math-merge/qwen_math_1.5b_level_3_step_1-4.log 2>&1 &
wait $!
nohup bash /data/qq/math-evaluation-harness/scripts/Qwen2.5-Math-1.5B/run_eval_ins_2.sh > /data/qq/math-evaluation-harness/log_saves/Qwen2.5-Math-1.5B/math-merge/qwen_math_1.5b_level_3_step_1-2.log 2>&1 &