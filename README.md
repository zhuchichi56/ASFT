# Anchored Supervised Fine-Tuning (ASFT)

[![arXiv](https://img.shields.io/badge/arXiv-2509.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2509.XXXXX)
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

**📄 2025-09-28**: Released ASFT code and paper - [Paper](ASFT.pdf) | [Code](https://github.com/zhuchichi56/ASFT)

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
git clone https://github.com/your-org/ASFT.git
cd ASFT
pip install -r requirements.txt
```

#### 2. Basic Training

Train an ASFT model with default settings:

```bash
python train.py \
    --model llama-2-7b \
    --method asft \
    --data data/medmcqa \
    --lambda 0.05 \
    --epochs 3 \
    --lr 2e-5
```

#### 3. Advanced Training Options

For custom training configurations:

```bash
# Training with custom hyperparameters
python train.py \
    --model llama-2-7b \
    --method asft \
    --data data/math_reasoning \
    --lambda 0.1 \
    --epochs 5 \
    --lr 1e-5 \
    --batch_size 32 \
    --gradient_accumulation_steps 4 \
    --warmup_ratio 0.1 \
    --save_steps 500 \
    --eval_steps 100
```

#### 4. Multi-Domain Training

Train on multiple datasets simultaneously:

```bash
python train.py \
    --model llama-2-7b \
    --method asft \
    --data data/medmcqa,data/math_reasoning,data/code_generation \
    --lambda 0.05 \
    --multi_domain \
    --domain_weights 0.3,0.4,0.3
```

#### 5. Evaluation

Evaluate trained models on various benchmarks:

```bash
# Single benchmark evaluation
python eval.py \
    --model checkpoints/asft_model \
    --bench medical

# Multi-benchmark evaluation
python eval.py \
    --model checkpoints/asft_model \
    --bench medical,math,code \
    --output_dir results/ \
    --batch_size 16
```

#### 6. Comparison with Baselines

Compare ASFT with other methods:

```bash
# Evaluate multiple methods
python compare.py \
    --models checkpoints/sft_model,checkpoints/dft_model,checkpoints/asft_model \
    --methods sft,dft,asft \
    --bench math_reasoning \
    --output results/comparison.json
```

---

## 📚 Citation

If you find this work useful, please cite:

```bibtex
@article{zhu2025asft,
  title={Anchored Supervised Fine-Tuning},
  author={Zhu, He and Su, Junyou and Lai, Peng and Ma, Ren and Zhang, Wenjia and Yang, Linyi and Chen, Guanhua},
  journal={arXiv preprint arXiv:2509.XXXXX},
  year={2025}
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
