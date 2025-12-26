# Train and Evaluate VGT-AE

According to [tokenizer.md](tokenizer/readme.md).

---

## Train and Evaluate VGT-QuaryAR

### Training

#### Pretraining

**Step 1: Prepare Data**  
Prepare pretraining data following [DATASETS.md](docs/DATASETS.md) and configure the dataset file:

```
configs/datasets/internvl3_1b_448/pretrain23m_images.py
```

**Step 2: Run Training**  

Single machine (8 GPUs):
```bash
cd /path/to/VGT
export PYTHONPATH=./:$PYTHONPATH
nproc_per_node=8 bash ./scripts/train_ddp_pretrain.sh
```

Multi-machine training (2 machines example):

Machine 1:
```bash
cd /path/to/VGT
export PYTHONPATH=./:$PYTHONPATH
nproc_per_node=8 bash ./scripts/train_ddp_pretrain.sh 2 <master_ip> 0
```

Machine 2:
```bash
cd /path/to/VGT
export PYTHONPATH=./:$PYTHONPATH
nproc_per_node=8 bash ./scripts/train_ddp_pretrain.sh 2 <master_ip> 1
```

**Optional: Training Monitoring**  
Configure the monitoring API in `configs/pretrain/vgt_internvl3_0_6B_448px_pretrain.py` using [SwanLab](https://github.com/swanhubx/swanlab) to visualize training loss and intermediate sample images.

#### Fine-tuning

**Step 1: Prepare Data**  
Prepare fine-tuning data according to [DATASETS.md](docs/DATASETS.md) and configure the dataset file:

```
configs/datasets/internvl3_1b_448/finetune_images.py
```

**Step 2: Run Fine-tuning**  

Single machine (8 GPUs):
```bash
cd /path/to/VGT
export PYTHONPATH=./:$PYTHONPATH
nproc_per_node=8 bash ./scripts/train_ddp_sft.sh
```

---

### Evaluation
```bash
# Install dependencies
conda create -n vgt_eval python=3.10
conda activate vgt_eval

```

#### Geneval

Install Geneval in `/path/to/VGT/data`:
```bash
cd data
git clone https://github.com/djghosh13/geneval  # follow their readme for installation
```

Run evaluation (single machine, 8 GPUs):
```bash
cd /path/to/VGT
export PYTHONPATH=./:$PYTHONPATH
bash scripts/evaluation/eval_geneval.sh
```

#### DPG-Bench

Install ELLA in `/path/to/VGT/data`:
```bash
cd data
git clone https://github.com/TencentQQGYLab/ELLA  # follow their readme for installation
```

Run evaluation (single machine, 8 GPUs):
```bash
cd /path/to/VGT
export PYTHONPATH=./:$PYTHONPATH
bash scripts/evaluation/eval_dpgbench.sh
```

---

## 🔬 Training Details

### Data Sources

**Pretraining** (50K steps, batch size 512) or (100K steps, batch size 256):
- [BLIP-3o](https://github.com/JiuhaiChen/BLIP3o) (filtered and curated)

**SFT** (5K steps):
- BLIP-3o-60K
- [ShareGPT4o](https://sharegpt4o.github.io/)
- [Echo-4o](https://github.com/yejy53/Nano-banana-150k)

### Training Configuration
- **Pretraining**: 50,000 iterations, batch size 512 or (100K steps, batch size 256)
- **SFT**: 5,000 iterations, combining multiple high-quality datasets