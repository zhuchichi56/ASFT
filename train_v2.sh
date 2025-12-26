#!/bin/bash
# train_lora.sh - Script to run ASFT/LoRA training

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TRAIN_SCRIPT=${TRAIN_SCRIPT:-"$SCRIPT_DIR/train_lora.py"}

# ============ User Configurable Variables ============
# Select which GPUs to use (comma-separated)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"6"}

# Path to the pretrained model (make sure the model is downloaded)
MODEL_PATH=${MODEL_PATH:-"/volume/pt-train/users/wzhang/ghchen/zh/models/Llama-2-7b"}

# Path to the training data
DATA_PATH=${DATA_PATH:-"/volume/pt-train/users/wzhang/ghchen/zh/code/Jy/data/train_medmcqa_alpaca_10k.jsonl"}

# Output directory
OUTPUT_DIR=${OUTPUT_DIR:-"output/test_train_asft_lora"}

# Training parameters
MODE=${MODE:-"asft"} # Training mode: sft, dft, sft+kl, asft
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-512}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-3}
KL_WEIGHT=${KL_WEIGHT:-0.1}

# ============ LoRA Config ============
USE_LORA=${USE_LORA:-True}
LORA_R=${LORA_R:-8}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}

# ============ Run Training ============
echo "Starting training with the following settings:"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "MODEL_PATH=$MODEL_PATH"
echo "DATA_PATH=$DATA_PATH"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "MODE=$MODE"
echo "MODEL_MAX_LENGTH=$MODEL_MAX_LENGTH"
echo "GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE"
echo "LEARNING_RATE=$LEARNING_RATE"
echo "NUM_TRAIN_EPOCHS=$NUM_TRAIN_EPOCHS"
echo "KL_WEIGHT=$KL_WEIGHT"
echo "USE_LORA=$USE_LORA"
echo "LORA_R=$LORA_R"
echo "LORA_ALPHA=$LORA_ALPHA"
echo "LORA_DROPOUT=$LORA_DROPOUT"

python "$TRAIN_SCRIPT" \
    --mode "$MODE" \
    --model_max_length "$MODEL_MAX_LENGTH" \
    --global_batch_size "$GLOBAL_BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --kl_weight "$KL_WEIGHT" \
    --model_name_or_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --use_lora "$USE_LORA" \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout "$LORA_DROPOUT" 
