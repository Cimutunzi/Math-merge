set -ex
PROMPT_TYPE='deepseek-math'
MODEL_NAME_OR_PATH='/data/qq/models/deepseek-math-7b-instruct'
ADAPTER_PATH='/data/qq/LLaMA-Factory/saves/deepseek-math-7b-instruct/lora/math-merge/stage_2_tem_0.7/1.0e-06'
STAGE='2_tem_0.7'
# adaper_split='checkpoint-200'

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

TRAIN_DATA="math-merge"
# DATA_NAMES="math,MATH-500,gsm8k"
# DATA_NAMES="aime24,amc23,olympiadbench,minerva-math"
# DATA_NAMES="math,MATH-500,gsm8k,aime24,amc23,olympiadbench,minerva-math"
DATA_NAMES="MATH-500,gsm8k,aime24,amc23,olympiadbench,minerva-math"
OUTPUT_DIR=${MODEL_NAME_OR_PATH}/math_eval/${TRAIN_DATA}/Stage_${STAGE}/${adaper_split}/1.0e-06
# DATA_NAMES="math,minerva-math,svamp,asdiv,mawps,tabmwp,mathqa,mmlu_stem,sat_math"
SPLIT="test"
NUM_TEST_SAMPLE=-1
# single-gpu
for (( n_sampling=1; n_sampling<=1; n_sampling*=2 ))
do
    CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=false \
    python3 -u math_eval.py \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --adpter_path ${ADAPTER_PATH} \
        --output_dir ${OUTPUT_DIR} \
        --data_names ${DATA_NAMES} \
        --split ${SPLIT} \
        --prompt_type ${PROMPT_TYPE} \
        --num_test_sample ${NUM_TEST_SAMPLE} \
        --seed 0 \
        --temperature 0 \
        --n_sampling ${n_sampling} \
        --top_p 1 \
        --start 0 \
        --end -1 \
        --save_outputs \
        --use_vllm \
        # --overwrite \
done

# multi-gpu
# python3 scripts/run_eval_multi_gpus.py \
#     --model_name_or_path $MODEL_NAME_OR_PATH \
#     --output_dir $OUTPUT_DIR \
#     --data_names ${DATA_NAMES} \
#     --prompt_type "cot" \
#     --temperature 0 \
#     --use_vllm \
#     --save_outputs \
#     --available_gpus 0,1,2,3,4,5,6,7 \
#     --gpus_per_model 1 \
#     --overwrite
