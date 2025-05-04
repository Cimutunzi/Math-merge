
nohup bash /data/qq/math-evaluation-harness/scripts/qwen/math/tem_0.5/stage_3.sh > /data/qq/math-evaluation-harness/log_saves/Qwen2_1.5b_ins/math/stage_3_0.5.log 2>&1 &
wait $!
nohup bash /data/qq/math-evaluation-harness/scripts/qwen/math/tem_0.5/stage_2.sh > /data/qq/math-evaluation-harness/log_saves/Qwen2_1.5b_ins/math/stage_2_0.5.log 2>&1 &
wait $!
nohup bash /data/qq/math-evaluation-harness/scripts/qwen/math/tem_0.5/stage_1.sh > /data/qq/math-evaluation-harness/log_saves/Qwen2_1.5b_ins/math/stage_1_0.5.log 2>&1 &
wait $!
nohup bash /data/qq/math-evaluation-harness/scripts/qwen/math/stage_base.sh > /data/qq/math-evaluation-harness/log_saves/Qwen2_1.5b_ins/math/stage_base.log 2>&1 &
# nohup bash /data/qq/math-evaluation-harness/scripts/qwen/math/tem_0.5/stage_3_800.sh > /data/qq/math-evaluation-harness/log_saves/Qwen2_1.5b_ins/math/stage_3_0.5_800.log 2>&1 &
# wait $!
# nohup bash /data/qq/math-evaluation-harness/scripts/qwen/math/tem_0.5/stage_3_1200.sh > /data/qq/math-evaluation-harness/log_saves/Qwen2_1.5b_ins/math/stage_3_0.5_1200.log 2>&1 &
# wait $!
# nohup bash /data/qq/math-evaluation-harness/scripts/qwen/math/stage_base_1400.sh > /data/qq/math-evaluation-harness/log_saves/Qwen2_1.5b_ins/math/stage_base_1400.log 2>&1 &
# wait $!
# nohup bash /data/qq/math-evaluation-harness/scripts/qwen/math/stage_base_2100.sh > /data/qq/math-evaluation-harness/log_saves/Qwen2_1.5b_ins/math/stage_base_2100.log 2>&1 &
# wait $!
# nohup bash /data/qq/math-evaluation-harness/scripts/run_eval_ins.sh > /data/qq/math-evaluation-harness/log_saves/Qwen2_1.5b_ins/math/train_level_0.7.log 2>&1 &
# wait $!
