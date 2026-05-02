# Anchored Supervised Fine-Tuning (ASFT)

[![arXiv](https://img.shields.io/badge/arXiv-2509.23753-b31b1b.svg)](https://www.arxiv.org/abs/2509.23753)
*A principled and efficient post-training method for large language models*

## 👥 Authors

**He Zhu**¹*, **Junyou Su**¹*, **Peng Lai**², **Ren Ma**³, **Wenjia Zhang**¹, **Linyi Yang**², **Guanhua Chen**²†

¹Peking University
²Southern University of Science and Technology
³Shanghai Artificial Intelligence Laboratory

*Equal Contribution
†Corresponding Author

---

## 🚀 Introduction

Post-training large language models (LLMs) faces a trade-off:

* **Supervised Fine-Tuning (SFT)** is efficient but prone to memorization.
* **Reinforcement Learning (RL)** improves generalization but is costly and unstable.
* **Dynamic Fine-Tuning (DFT)** tightens the learning bound but suffers from **distributional drift** and instability.

👉 We propose **Anchored Supervised Fine-Tuning (ASFT)** — a lightweight extension of DFT that adds **KL anchoring**.
This ensures **tightness + stability**, combining the best of SFT and RL while keeping efficiency.

---

## 📰 News

**🔥 2026-05-02** (recommended): New [`verl`](https://github.com/zhuchichi56/ASFT/tree/verl) branch — SFT / DFT / ASFT on top of the [verl](https://github.com/volcengine/verl) FSDP training framework, with built-in vLLM-based **medical MCQ evaluation** (medqa / mmlu_medical / medmcqa). One-shot script: `bash run_verl.sh`. See [verl branch usage](#-verl-branch-recommended) below.

**📄 2026-02-12**: ASFT has been merged into LLaMA-Factory main ([commit #10174](https://github.com/hiyouga/LLaMA-Factory/commit/675ce8cc7f70a65de403ccfd05195ca3ea6f3bd4)).  
Latest release is `v0.9.4`, so ASFT support is currently available on main and will be included in the next tagged release.

**📄 2026-01-30**: Accepted to ICLR 2026.

**📄 2026-01-23**: Added support for DeepSpeed and LoRA.

**📄 2025-09-28**: Released ASFT code and paper - [Paper](asft.pdf) | [Code](https://github.com/zhuchichi56/ASFT)

---

## ⭐ verl branch (recommended)

The `verl` branch is the **recommended** way to reproduce ASFT going forward. It ships:

- A self-contained `verl/` sub-tree built on [verl](https://github.com/volcengine/verl) v0.6.1 (FSDP).
- A unified `fsdp_sft_trainer` supporting `loss_mode ∈ {sft, dft, asft}` — switch with a single config key.
- A `recipe/asft_bio/` pipeline that prepares the bio/med dataset, trains, and evaluates on the medical MCQ benchmarks (medqa / mmlu_medical / medmcqa) via vLLM.
- A top-level **one-shot script** that does the full train → eval loop on 8 GPUs (7 train + 1 eval).

### One-shot

```bash
git clone -b verl https://github.com/zhuchichi56/ASFT.git
cd ASFT
pip install -r verl/requirements.txt           # verl framework deps
pip install -e ./verl                          # install the verl package
bash run_verl.sh                               # default: sft+dft+asft, LLaMA-2-7B-base, 3 epochs
```

This will:
1. Download `Llama-2-7b-hf` (NousResearch mirror) into `./models/` if missing
2. Inject a minimal Alpaca chat template into the tokenizer (idempotent)
3. Prepare the bio/med train/val parquet under `verl/recipe/asft_bio/data/med/`
4. Train **SFT → DFT → ASFT** sequentially on GPUs 0–6
5. After each mode, evaluate the saved checkpoint on medqa / mmlu_medical / medmcqa using GPU 7 (vLLM)

Outputs go to `./checkpoints/asft_verl/<mode>/` (training log, ckpts, `medeval_<mode>.json`).

### Common overrides

```bash
# Run only ASFT, smaller smoke test
LOSS_MODES=asft EPOCHS=1 TRAIN_MAX_SAMPLES=1000 bash run_verl.sh

# Custom model
MODEL_ID=meta-llama/Llama-2-7b-hf MODEL_PATH=/path/to/llama2 bash run_verl.sh

# Tune ASFT KL anchor strength
ASFT_KL_COEF=0.05 LOSS_MODES=asft bash run_verl.sh

# Different GPU layout (e.g., 4 train + 1 eval on a 5-GPU box)
NUM_GPUS=4 CUDA_TRAIN=0,1,2,3 CUDA_EVAL=4 bash run_verl.sh
```

### Layout

```
ASFT/
├── run_verl.sh                              # one-shot entry point
└── verl/
    ├── verl/trainer/
    │   ├── fsdp_sft_trainer.py              # loss_mode dispatch (sft/dft/asft)
    │   └── config/sft_trainer.yaml          # asft_kl_coef, benchmark_eval_dir
    ├── verl/utils/med_mcq.py                # MCQ prompt + answer extraction
    ├── recipe/asft_bio/
    │   ├── prepare_all_data.py              # build train/val parquet from HF chichi56/ASFT
    │   └── run_sft_variants_med.sh          # main 8-GPU launcher
    ├── eval/medeval/
    │   ├── run_med_eval.py                  # vLLM medical MCQ evaluator
    │   └── test_data/{medqa,mmlu_medical,medmcqa}_test.jsonl
    └── scripts/build_medeval_val_datasets.py
```

> **Why `verl` over the legacy `train_v2.py` path?** The verl track gives you (a) FSDP-2 / sequence parallel out of the box for 7B+ scale, (b) a single launcher that sweeps SFT/DFT/ASFT for clean apples-to-apples comparison, (c) integrated medical MCQ benchmarking without touching extra eval scripts. Use the legacy path below only if you specifically need the LLaMA-Factory- or DeepSpeed-style entry points.

---

## ✨ Key Features

1. **Theoretical foundation**:

   * Formalized in the *Reward-Weighted Regression (RWR)* framework.
   * Proves DFT yields tighter RL lower bounds than SFT.
   * Identifies drift as the key weakness of DFT.

2. **Anchored stability**:

   * Adds a KL divergence regularization term to prevent drift.
   * Retains DFT’s advantages with controlled variance.

3. **Practical benefits**:

   * Minimal overhead compared to SFT.
   * Outperforms SFT, DFT, and iw-SFT across reasoning, medical, and code benchmarks.
   * Provides stronger initialization for RL methods like DAPO/GRPO. 

---

## 📊 Main Results

### Performance Comparison
<p align="center">
  <img src="fig/main.png" width="800">
</p>

*Performance comparison of fine-tuning methods on medical and math benchmarks under different dataset scales. ASFT consistently outperforms other methods.*

### Training Dynamics
<p align="center">
  <img src="fig/compare.png" width="800">
</p>

*Training dynamics comparison showing ASFT maintains stability through KL anchoring while DFT exhibits severe distributional drift.*

### Cross-Model Performance
<p align="center">
  <img src="fig/scale.png" width="800">
</p>

*Comparison across different model architectures (LLaMA-2, Qwen2.5) demonstrating ASFT's consistent effectiveness across various model sizes and families.*

---

## 🔧 Usage

### Quick Start

#### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/zhuchichi56/ASFT.git
cd ASFT
conda create -n asft python=3.10
conda activate asft
pip install -r requirements.txt
```

If you need flash-attn (prebuilt wheel):

```bash
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

> Note: install a matching PyTorch build first (e.g., CUDA 12 + PyTorch 2.4) before installing flash-attn.

#### 2. Basic Training

Train an ASFT model with default settings (v2 supports more model families and multi-GPU training):

```bash
python train_v2.py \
    --model_name_or_path models/your-model \
    --mode asft \
    --data_path data/your-data.jsonl \
    --kl_weight 0.03 \
    --num_train_epochs 3 \
    --learning_rate 2e-5
```

DeepSpeed is supported via `--deepspeed_config` (Zero-2/Zero-3). Config files are in `scripts/` (e.g., `scripts/ds_zero2_bf16.json`). In practice, DeepSpeed Zero tends to be less stable; native (non-DeepSpeed) runs are the most stable overall. For example:

```bash
deepspeed --num_gpus 8 train_v2.py \
    --deepspeed_config scripts/ds_zero2_bf16.json \
    --model_name_or_path models/your-model \
    --mode asft \
    --data_path data/your-data.jsonl \
    --kl_weight 0.03 \
    --num_train_epochs 3 \
    --learning_rate 2e-5
```

> Note: For mixed precision (bf16/fp16), we recommend `kl_weight=0.03`. Larger KL weights amplify precision noise and can destabilize training, leading to degraded accuracy. Setting `0.03` keeps the KL anchor effective without over-regularizing under lower precision.

#### 3. LoRA (Recommended)

We recommend LoRA with `rank=8`, `lora_alpha=16`, `lora_dropout=0.05`, and **learning rate `5e-4`** for medical tasks. In our grid, `lr=5e-4, r=8` performs best on average and is noticeably stronger than `lr=2e-5` under the same rank.

Example (LoRA):

```bash
python train_v2.py \
    --model_name_or_path models/your-model \
    --mode asft \
    --data_path data/your-data.jsonl \
    --use_lora True \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --learning_rate 5e-4
```

Partial grid (Med, LLaMA2-7B):

| lr | rank | medqa | mmlu | medmcqa | avg |
|----|------|-------|------|---------|-----|
| 2.00E-05 | 8  | 0.3064 | 0.3366 | 0.3376 | 0.3269 |
| 5.00E-05 | 8  | 0.3299 | 0.3607 | 0.3464 | 0.3457 |
| 1.00E-04 | 8  | 0.3511 | 0.3896 | 0.3588 | 0.3665 |
| 2.00E-04 | 4  | 0.3692 | 0.4188 | 0.3717 | 0.3866 |
| 5.00E-04 | 8  | 0.3951 | 0.4147 | 0.3737 | 0.3945 |

#### 3. Evaluation

Evaluate trained models on various benchmarks. See `eval/README.md` for detailed steps and required inputs.

```bash
# AlpacaEval-style evaluation
python /volume/pt-train/users/wzhang/ghchen/zh/valid_code/ASFT-dev/eval/alpaca_eval_test.py

# Math evaluation
bash eval/math_evaluation/eval.sh

# Medical evaluation
python eval/medeval/vllm_medical_test.py
```



---

## 📦 Data Access

Large-scale training data is not stored in this repository. Please download it from the Hugging Face dataset repository:
`chichi56/ASFT`

You can also download all dataset files with the provided script:

```bash
python download_data.py --output_dir data
```

## 📚 Citation

If you find this work useful, please cite:

```bibtex
@misc{zhu2025anchoredsupervisedfinetuning,
      title={Anchored Supervised Fine-Tuning}, 
      author={He Zhu and Junyou Su and Peng Lai and Ren Ma and Wenjia Zhang and Linyi Yang and Guanhua Chen},
      year={2025},
      eprint={2509.23753},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.23753}, 
}
```

---

## 🤝 Contributing

We welcome contributions! Please open issues or submit PRs for:

* Extending ASFT to new domains
* Improving training efficiency
* Adding evaluation benchmarks

---

## 🌟 Highlights

* **SFT efficiency + RL generalization**
* **Tighter theoretical guarantees**
* **Stable across tasks and scales**
* **Plug-and-play for LLaMA, Qwen, and more**

---
