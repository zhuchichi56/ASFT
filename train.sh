#!/bin/bash
# train.sh - Script to run ASFT training

# ============ User Configurable Variables ============
# Select which GPUs to use (comma-separated)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1"}

# Path to the pretrained model (make sure the model is downloaded)
MODEL_PATH=${MODEL_PATH:-"/path/to/your/model"}

# Path to the training data
DATA_PATH=${DATA_PATH:-"/path/to/your/training_data.jsonl"}

# Output directory
OUTPUT_DIR=${OUTPUT_DIR:-"/path/to/your/output_dir"}

# Training parameters
MODE=${MODE:-"asft"} # Training mode: sft, sft+kl, asft, dft+kl
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-2048}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-256}
LEARNING_RATE=${LEARNING_RATE:-5e-5}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-3}
KL_WEIGHT=${KL_WEIGHT:-0.1}

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

python train.py \
    --mode $MODE \
    --model_max_length $MODEL_MAX_LENGTH \
    --global_batch_size $GLOBAL_BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --kl_weight $KL_WEIGHT \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR
