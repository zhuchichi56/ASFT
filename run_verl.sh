#!/usr/bin/env bash
# One-shot launcher for the verl track (recommended).
# Trains LLaMA-2-7B-base on the bio/med dataset with SFT/DFT/ASFT in sequence,
# then runs the medqa / mmlu_medical / medmcqa benchmark on each ckpt.
#
# Layout: 8 GPUs total -> 7 train + 1 eval. Override env vars to change.
#
# Examples:
#   # Default (sft + dft + asft, full data, 3 epochs)
#   bash run_verl.sh
#
#   # Pick one mode + smaller smoke run
#   LOSS_MODES=asft EPOCHS=1 TRAIN_MAX_SAMPLES=1000 bash run_verl.sh
#
#   # Custom model
#   MODEL_ID=meta-llama/Llama-2-7b-hf MODEL_PATH=/path/to/llama2 bash run_verl.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${REPO_ROOT}/verl:${PYTHONPATH:-}"

# Defaults — override via env
export MODEL_ID="${MODEL_ID:-NousResearch/Llama-2-7b-hf}"
export MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/models/Llama-2-7b-hf}"
export RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/checkpoints/asft_verl}"
export DATA_DIR="${DATA_DIR:-${REPO_ROOT}/verl/recipe/asft_bio/data}"
export LOSS_MODES="${LOSS_MODES:-sft dft asft}"
export EPOCHS="${EPOCHS:-3}"
export GLOBAL_BSZ="${GLOBAL_BSZ:-64}"
export MICRO_BSZ="${MICRO_BSZ:-2}"
export LR="${LR:-2e-5}"
export MAX_LEN="${MAX_LEN:-512}"
export ASFT_KL_COEF="${ASFT_KL_COEF:-0.1}"
export TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-10000}"
export VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-10000}"
export NUM_GPUS="${NUM_GPUS:-7}"
export CUDA_TRAIN="${CUDA_TRAIN:-0,1,2,3,4,5,6}"
export CUDA_EVAL="${CUDA_EVAL:-7}"
export EVAL_TP="${EVAL_TP:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"

# LLaMA-2-base ships without a chat template — inject a minimal alpaca one
# (idempotent) so verl's SFTDataset.apply_chat_template() works.
ensure_chat_template() {
  local p="${MODEL_PATH}/tokenizer_config.json"
  [[ -f "${p}" ]] || return 0
  python - <<PY
import json, sys
p = "${p}"
c = json.load(open(p))
if not c.get("chat_template"):
    c["chat_template"] = (
        "{% for m in messages %}"
        "{% if m['role']=='user' %}### Instruction:\n{{ m['content'] }}\n\n### Response:\n"
        "{% else %}{{ m['content'] }}{{ eos_token }}{% endif %}"
        "{% endfor %}"
    )
    json.dump(c, open(p, "w"), indent=2)
    print("[run_verl] injected default alpaca chat_template into", p)
PY
}

mkdir -p "${REPO_ROOT}/models" "$(dirname "${RUN_ROOT}")"

# Download the model if not present
if ! ls "${MODEL_PATH}"/*.safetensors >/dev/null 2>&1 && \
   ! ls "${MODEL_PATH}"/pytorch_model*.bin >/dev/null 2>&1; then
  echo "[run_verl] downloading ${MODEL_ID} -> ${MODEL_PATH}"
  mkdir -p "${MODEL_PATH}"
  huggingface-cli download "${MODEL_ID}" --local-dir "${MODEL_PATH}" --exclude "*.bin"
fi
ensure_chat_template

bash "${REPO_ROOT}/verl/recipe/asft_bio/run_sft_variants_med.sh"
