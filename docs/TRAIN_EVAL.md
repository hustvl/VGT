# Training Guide

This guide covers training the VGT model, including both the autoencoder (VGT-AE) and autoregressive model (VGT-AR).

---

## Overview

The VGT training pipeline consists of two main components:

1. **VGT-AE (Autoencoder)**: Tokenizes images into discrete tokens
2. **VGT-AR (Autoregressive Model)**: Generates images from text prompts

You can either use pre-trained VGT-AE weights or train your own from scratch.

---

## Part 1: VGT-AE Setup

### Option A: Use Pre-trained VGT-AE (Recommended)

Download the pre-trained autoencoder weights:

```bash
hf download hustvl/vgt_ae --repo-type model --local-dir ckpts/vgt_ae
```

Configure in your model config file (e.g., `configs/models/vgt_internvl3_0_6B_448px.py`):

```python
vgt_ae = dict(
    type = "vgt_pretrain",
    mllm_path = "OpenGVLab/InternVL3-1B",
    dc_ae_path = "mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers",
    checkpoint_path = "ckpts/vgt_ae/vgt_ae_internvl3.pth",
    encoder_norm = True,
)
```

### Option B: Train VGT-AE from Scratch

If you want to train your own autoencoder, follow the detailed instructions in the [tokenizer documentation](tokenizer/readme.md).

After training, configure your customize VGT-AE in the model config:

```python
vgt_ae = dict(
    config_path = "tokenizer/configs/vgtae_intervl3/vlvae_intervl3_p28_448px_stage2.yaml",
    checkpoint_path = "tokenizer/checkpoints/VGTAE_intervl3_stage2/checkpoint-50000/unwrapped_model/pytorch_model.bin"
)
```

---

## Part 2: Train VGT-AR

The VGT-AR model is trained in two stages: pretraining and fine-tuning.

### Stage 1: Pretraining

**Prepare Dataset**

Configure your pretraining data following [DATASETS.md](docs/DATASETS.md):

```
configs/datasets/internvl3_1b_448/pretrain23m_images.py
```

**Run Training**

Single machine (8 GPUs):
```bash
cd /path/to/VGT
export PYTHONPATH=./:$PYTHONPATH
nproc_per_node=8 bash ./scripts/train_ddp_pretrain.sh
```

Multi-machine training (example with 2 machines):

- **Machine 1** (master node):
  ```bash
  cd /path/to/VGT
  export PYTHONPATH=./:$PYTHONPATH
  nproc_per_node=8 bash ./scripts/train_ddp_pretrain.sh 2 <master_ip> 0
  ```

- **Machine 2** (worker node):
  ```bash
  cd /path/to/VGT
  export PYTHONPATH=./:$PYTHONPATH
  nproc_per_node=8 bash ./scripts/train_ddp_pretrain.sh 2 <master_ip> 1
  ```

**Training Monitoring (Optional)**

Configure [SwanLab](https://github.com/swanhubx/swanlab) in `configs/pretrain/vgt_internvl3_0_6B_448px_pretrain.py` to visualize training metrics and generated samples in real-time.

### Stage 2: Fine-tuning

**Prepare Dataset**

Configure your fine-tuning data following [DATASETS.md](docs/DATASETS.md):

```
configs/datasets/internvl3_1b_448/finetune_images.py
```

**Run Fine-tuning**

Single machine (8 GPUs):
```bash
cd /path/to/VGT
export PYTHONPATH=./:$PYTHONPATH
nproc_per_node=8 bash ./scripts/train_ddp_sft.sh
```

---

## Part 3: Evaluation

### Setup Evaluation Environment

```bash
conda create -n vgt_eval python=3.10
conda activate vgt_eval
```

### Geneval Benchmark

**Install Geneval:**
```bash
cd /path/to/VGT/data
git clone https://github.com/djghosh13/geneval
# Follow their README for installation
```

**Run evaluation** (8 GPUs):
```bash
cd /path/to/VGT
export PYTHONPATH=./:$PYTHONPATH
bash scripts/evaluation/eval_geneval.sh
```

### DPG-Bench

**Install ELLA:**
```bash
cd /path/to/VGT/data
git clone https://github.com/TencentQQGYLab/ELLA
# Follow their README for installation
```

**Run evaluation** (8 GPUs):
```bash
cd /path/to/VGT
export PYTHONPATH=./:$PYTHONPATH
bash scripts/evaluation/eval_dpgbench.sh
```

---

## Training Configuration Details

### Pretraining Stage
- **Duration**: 50K steps (batch size 512) or 100K steps (batch size 256)
- **Dataset**: [BLIP-3o](https://github.com/JiuhaiChen/BLIP3o) (filtered and curated)

### Fine-tuning Stage
- **Duration**: 5K steps
- **Datasets**:
  - BLIP-3o-60K
  - [ShareGPT4o](https://sharegpt4o.github.io/)
  - [Echo-4o](https://github.com/yejy53/Nano-banana-150k)
