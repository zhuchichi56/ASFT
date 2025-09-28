PROMPT_TYPE="alpaca"
N_SAMPLING=16
TEMPERATURE=1

# N_SAMPLING=1
# TEMPERATURE=0

# PROMPT_TYPE="llama-base-boxed"
# PROMPT_TYPE="deepseek-math"
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# 如果需要调整请去 math_eval.sh
# 请启动lm-eval-harness 环境进行测试
# N_SAMPLING=16
# TEMPERATURE=1

# Math
# MODEL_BASE_DIR="/volume/pt-train/users/wzhang/ghchen/zh/saves/sft/numina-math/"

# MODEL_LIST=(
#     "Qwen2.5-7B-Numina_Math_flatness_high_20pct"
#     "Qwen2.5-7B-Numina_Math_flatness_low_20pct"
#     "Qwen2.5-7B-Numina_Math_grad_high_20pct"
#     "Qwen2.5-7B-Numina_Math_grad_low_20pct"
#     "Qwen2.5-7B-Numina_Math_loss_high_20pct"
#     "Qwen2.5-7B-Numina_Math_loss_low_20pct"
#     "Qwen2.5-7B-Numina_Math_random_20pct"
#     "Qwen2.5-7B-Numina_Math_rank_fusion_20pct"
# )
# MODEL_BASE_DIR="/volume/pt-train/users/wzhang/ghchen/zh/saves/sft/"

# MODEL_LIST=(
#     "Llama-2-7b-math_flatness_high_20pct"
#     "Llama-2-7b-math_flatness_low_20pct"
#     "Llama-2-7b-math_grad_high_20pct"
#     "Llama-2-7b-math_grad_low_20pct"
#     "Llama-2-7b-math_loss_high_20pct"
#     "Llama-2-7b-math_loss_low_20pct"
#     "Llama-2-7b-math_random_20pct"
#     "Llama-2-7b-math_rank_fusion_20pct"
# )


# for MODEL_NAME in "${MODEL_LIST[@]}"; do
#     MODEL_NAME_OR_PATH="${MODEL_BASE_DIR}/${MODEL_NAME}"
#     OUTPUT_DIR="outputs/${MODEL_NAME}"
#     echo "Evaluating $MODEL_NAME_OR_PATH -> $OUTPUT_DIR"
#     bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $OUTPUT_DIR $N_SAMPLING $TEMPERATURE
# done

# MODEL_NAME_OR_PATH="/volume/pt-train/users/wzhang/ghchen/zh/DFT/verl/checkpoints/numina-cot-dft-llama2-7b/global_step_390"
# MODEL_NAME_OR_PATH="/volume/pt-train/users/wzhang/ghchen/zh/code/caft/emergent_misalignment/src/output/dft+kl+math/Llama-2-7b_kl_1000"
# MODEL_NAME_OR_PATH="/volume/pt-train/users/wzhang/ghchen/zh/models/Llama-2-7b"
# MODEL_NAME_OR_PATH="/volume/pt-train/users/wzhang/ghchen/zh/models/Qwen2.5-7B"
MODEL_NAME_OR_PATH="/volume/pt-train/users/wzhang/ghchen/zh/code/caft/emergent_misalignment/src/output/math+scale/qwen25-32b-sft-kl0500-10k"
OUTPUT_DIR="outputs/Qwen25_32B_SFTKL0500_3k_Sample16"
# MODEL_NAME_OR_PATH="/volume/pt-train/users/wzhang/ghchen/zh/models/Qwen2.5-Math-7B-Instruct"
# OUTPUT_DIR="outputs/Qwen25_7b_math_instruct"

echo "Evaluating $MODEL_NAME_OR_PATH -> $OUTPUT_DIR"
bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $OUTPUT_DIR $N_SAMPLING $TEMPERATURE






