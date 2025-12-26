#!/bin/bash
set -e
set -x

LOG_DIR="./eval/DPG-BENCH/logs"
mkdir -p "$LOG_DIR"
EVAL_ENV=./miniconda3/envs/vgt_eval/bin/python
IMAGE_SIZE=448

# Configure models to evaluate (arrays must have same length)
declare -a CONFIGS=(
    # "./configs/pretrain/vgt_internvl3_0_6B_448px_pretrain.py"
    "./configs/finetune/vgt_internvl3_0_6B_448px_sft.py"
    # ... muti conifg eval
)

declare -a CHECKPOINTS=(
    # "work_dirs/vgt_internvl3_0_6B_448px_pretrain/iter_70000.pth"
    "work_dirs/vgt_internvl3_0_6B_448px_sft/iter_10000.pth"
)


DATE=$(date +"%Y%m%d-%H%M%S")
LOG_FILE="${LOG_DIR}/dpgbench_eval_all_${DATE}.log"

ROOT_DIR="./"
EVAL_BASE="$ROOT_DIR/eval/DPG-BENCH"

ACCEL_CFG="$ROOT_DIR/scripts/evaluation/default_config.yaml"
DIST_EVAL_SCRIPT="$ROOT_DIR/scripts/evaluation/dpg_bench/dist_eval.sh"


for i in "${!CONFIGS[@]}"; do
    CONFIG_PATH="${CONFIGS[$i]}"
    CHECKPOINT_PATH="${CHECKPOINTS[$i]}"

    CKPT_NAME=$(basename "$CHECKPOINT_PATH" .pth)
    MODEL_NAME=$(basename "$(dirname "$CHECKPOINT_PATH")")

    OUTPUT_DIR="${EVAL_BASE}/${MODEL_NAME}/${CKPT_NAME}"
    mkdir -p "$OUTPUT_DIR"

    echo ">>> [Model $i] CONFIG=$CONFIG_PATH, CHECKPOINT=$CHECKPOINT_PATH" | tee -a "$LOG_FILE"

    # conda activate vgt
    echo ">>> Stage 1: Generating outputs" | tee -a "$LOG_FILE"
    accelerate launch --config_file "$ACCEL_CFG" \
        "$ROOT_DIR/scripts/evaluation/dpg_bench.py" \
        "$CONFIG_PATH" \
        --checkpoint "$CHECKPOINT_PATH" \
        --batch_size 8 \
        --output "$OUTPUT_DIR" \
        --height $IMAGE_SIZE --width $IMAGE_SIZE \
        --seed 42 \
        2>&1 | tee -a "$LOG_FILE"


    echo ">>> Stage 2: Running distributed evaluation" | tee -a "$LOG_FILE"
    bash "$DIST_EVAL_SCRIPT" \
        "$OUTPUT_DIR" \
        $IMAGE_SIZE \
        4 \
        2>&1 | tee -a "$LOG_FILE"

    echo "===== [Done] $MODEL_NAME / $CKPT_NAME =====" | tee -a "$LOG_FILE"
    sleep 30
done

conda deactivate
echo "✅ All models evaluated! Log file: $LOG_FILE"
