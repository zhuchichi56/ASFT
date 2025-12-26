#!/bin/bash
# merge_lora_batch.sh - Batch merge LoRA adapters into base model

set -euo pipefail
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"6"}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MERGE_SCRIPT=${MERGE_SCRIPT:-"$SCRIPT_DIR/merge_lora.py"}

# ============ User Config ============
BASE_MODEL_PATH=${BASE_MODEL_PATH:-"/volume/pt-train/users/wzhang/ghchen/zh/models/Llama-2-7b"}

# LoRA experiments root dir (each subdir contains one LoRA adapter)
LORA_ROOT_DIR=${LORA_ROOT_DIR:-"output/grid_search_lora_qv"}

# Where to save merged models
MERGED_ROOT_DIR=${MERGED_ROOT_DIR:-"output/grid_search_lora_qv_merged"}

DTYPE=${DTYPE:-"float16"}   # float16 | bfloat16

# ============ Sanity Check ============
if [ ! -d "$LORA_ROOT_DIR" ]; then
    echo "❌ LoRA root dir not found: $LORA_ROOT_DIR"
    exit 1
fi

mkdir -p "$MERGED_ROOT_DIR"

# ============ Batch Merge ============
for LORA_DIR in "$LORA_ROOT_DIR"/*; do
    if [ ! -d "$LORA_DIR" ]; then
        continue
    fi

    # 判断是不是一个合法的 LoRA 目录
    if [ ! -f "$LORA_DIR/adapter_config.json" ]; then
        echo "⚠️  Skip (not a LoRA adapter dir): $LORA_DIR"
        continue
    fi

    EXP_NAME=$(basename "$LORA_DIR")
    SAVE_DIR="$MERGED_ROOT_DIR/$EXP_NAME"

    echo "================================================="
    echo "🔧 Merging LoRA:"
    echo "    Base model : $BASE_MODEL_PATH"
    echo "    LoRA dir   : $LORA_DIR"
    echo "    Save to    : $SAVE_DIR"
    echo "================================================="

    python "$MERGE_SCRIPT" merge_lora \
        --base_model_path "$BASE_MODEL_PATH" \
        --lora_model_path "$LORA_DIR" \
        --save_path "$SAVE_DIR" \
        --dtype "$DTYPE"

done

echo "✅ All LoRA adapters merged successfully!"
