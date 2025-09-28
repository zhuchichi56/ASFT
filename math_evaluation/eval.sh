PROMPT_TYPE="alpaca"
N_SAMPLING=16
TEMPERATURE=1

# MODEL_NAME_OR_PATH="models/Qwen2.5-32B-SFT"
# OUTPUT_DIR="outputs/Qwen2.5-32B-SFT"

MODEL_NAME_OR_PATH=""/volume/pt-train/users/wzhang/ghchen/zh/models/Llama-2-7b""
OUTPUT_DIR="outputs/Test"



echo "Evaluating $MODEL_NAME_OR_PATH -> $OUTPUT_DIR"
bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $OUTPUT_DIR $N_SAMPLING $TEMPERATURE






