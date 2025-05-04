
nohup bash /data/qq/math-evaluation-harness/scripts/llama/ScaleQuest-Math/tem_0.5/stage_3_2000.sh > /data/qq/math-evaluation-harness/log_saves/llama-8b-ins/ScaleQuest-Math/stage_3_0.5_2000.log 2>&1 &
nohup bash /data/qq/math-evaluation-harness/scripts/llama/ScaleQuest-Math/tem_0.5/stage_3.sh > /data/qq/math-evaluation-harness/log_saves/llama-8b-ins/ScaleQuest-Math/stage_3_0.5.log 2>&1 &
nohup bash /data/qq/math-evaluation-harness/scripts/llama/ScaleQuest-Math/stage_base.sh > /data/qq/math-evaluation-harness/log_saves/llama-8b-ins/ScaleQuest-Math/base.log 2>&1 &
nohup bash /data/qq/math-evaluation-harness/scripts/llama/ScaleQuest-Math/stage_base_3000.sh > /data/qq/math-evaluation-harness/log_saves/llama-8b-ins/ScaleQuest-Math/base_3000.log 2>&1 &
# wait $!
# nohup bash /data/qq/math-evaluation-harness/scripts/llama/math/tem_0.7/stage_2.sh > /data/qq/math-evaluation-harness/log_saves/llama-8b-ins/math/stage_2_0.7.log 2>&1 &
# # wait $!
# nohup bash /data/qq/math-evaluation-harness/scripts/llama/math/tem_0.7/stage_1.sh > /data/qq/math-evaluation-harness/log_saves/llama-8b-ins/math/stage_1_0.7.log 2>&1 &
# # wait $!
# nohup bash /data/qq/math-evaluation-harness/scripts/llama/math/stage_base.sh > /data/qq/math-evaluation-harness/log_saves/llama-8b-ins/math/stage_base.log 2>&1 &
# nohup bash /data/qq/math-evaluation-harness/scripts/llama/math/tem_0.7/stage_3_800.sh > /data/qq/math-evaluation-harness/log_saves/llama-8b-ins/math/stage_3_0.7_800.log 2>&1 &
# # wait $!
# nohup bash /data/qq/math-evaluation-harness/scripts/llama/math/tem_0.7/stage_3_1200.sh > /data/qq/math-evaluation-harness/log_saves/llama-8b-ins/math/stage_3_0.7_1200.log 2>&1 &
# # wait $!
# nohup bash /data/qq/math-evaluation-harness/scripts/llama/math/stage_base_1400.sh > /data/qq/math-evaluation-harness/log_saves/llama-8b-ins/math/stage_base_1400.log 2>&1 &
# # wait $!
# nohup bash /data/qq/math-evaluation-harness/scripts/llama/math/stage_base_2000.sh > /data/qq/math-evaluation-harness/log_saves/llama-8b-ins/math/stage_base_2000.log 2>&1 &
# wait $!
# nohup bash /data/qq/math-evaluation-harness/scripts/run_eval_ins.sh > /data/qq/math-evaluation-harness/log_saves/llama-8b-ins/math/train_level_0.7.log 2>&1 &
# wait $!
