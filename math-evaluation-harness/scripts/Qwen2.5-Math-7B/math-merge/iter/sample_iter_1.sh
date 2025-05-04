#!/bin/bash
set -ex
# echo "start sleeping!"
# sleep 3600
PROMPT_TYPE='qwen_box'
MODEL_NAME_OR_PATH='/data/qq/models/Qwen/Qwen2.5-Math-7B'
ADAPTER_PATH='/data/qq/LLaMA-Factory/saves/Qwen2.5-Math-7B/lora/math-merge/iter/stage_1_tem_0.7/5.0e-05'
# ======= Base Models =======
# PROMPT_TYPE="cot" # direct / cot / pal / tool-integrated
# MODEL_NAME_OR_PATH=${HF_MODEL_DIR}/mistral/Mistral-7B-v0.1
# MODEL_NAME_OR_PATH=${HF_MODEL_DIR}/llemma/llemma_7b
# MODEL_NAME_OR_PATH=${HF_MODEL_DIR}/internlm/internlm2-math-base-7b
# MODEL_NAME_OR_PATH=${HF_MODEL_DIR}/deepseek/deepseek-math-7b-base


# ======= SFT Models =======
# PROMPT_TYPE="deepseek-math" # self-instruct / tora / wizard_zs / deepseek-math / kpmath
# MODEL_NAME_OR_PATH=${HF_MODEL_DIR}/deepseek/deepseek-math-7b-rl
# MODEL_NAME_OR_PATH=${HF_MODEL_DIR}/deepseek/deepseek-math-7b-instruct


OUTPUT_DIR=${MODEL_NAME_OR_PATH}/math_eval/iter
DATA_NAMES="math-merge"
# DATA_NAMES="gsm8k,minerva_math,svamp,asdiv,mawps,tabmwp,mathqa,mmlu_stem,sat_math"
SPLIT="iter_1"
NUM_TEST_SAMPLE=-1


# single-gpu

# for (( n_sampling=1; n_sampling<=1; n_sampling*=2 ))
# do
#     CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false \
#     python3 -u math_eval.py \
#         --model_name_or_path ${MODEL_NAME_OR_PATH} \
#         --output_dir ${OUTPUT_DIR} \
#         --data_names ${DATA_NAMES} \
#         --split ${SPLIT} \
#         --prompt_type ${PROMPT_TYPE} \
#         --num_test_sample ${NUM_TEST_SAMPLE} \
#         --seed 0 \
#         --temperature 0 \
#         --n_sampling ${n_sampling} \
#         --top_p 1 \
#         --start 0 \
#         --end -1 \
#         --use_vllm \
#         --save_outputs \
#         # --overwrite \
# done

# multi-gpu


n_sampling=4
python3 scripts/run_eval_multi_gpus.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --adpter_path ${ADAPTER_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --data_names ${DATA_NAMES} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --temperature 0.7 \
    --use_vllm \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --n_sampling ${n_sampling} \
    --save_outputs \
    --available_gpus 0,1,2,3\
    --gpus_per_model 4 \
    # --overwrite

