#!/bin/bash
set -e
set -x

# ==============================
# OneIG Benchmark - Batch Evaluation Script
# ==============================
# Usage: Simply run this script without any arguments
#   bash scripts/evaluation/eval_oneIG.sh
#
# Configuration: Edit the CONFIGS and CHECKPOINTS arrays below
# ==============================

# ==============================
# Configuration
# ==============================
LOG_DIR="./eval/OneIG/logs"
mkdir -p "$LOG_DIR"

IMAGE_SIZE=512
LANGUAGE="en"  # Language: en or zh

# Configure models to evaluate (arrays must have same length)
declare -a CONFIGS=(
    "./configs/models/VGT_b_internvl3_1b_sana_0_6b_512_hf.py"
    # Add more configs here
    # "./configs/models/model2.py"
)

declare -a CHECKPOINTS=(
    "work_dirs/VGT_model/iter_10000.pth"
    # Add corresponding checkpoints here
    # "work_dirs/model2/iter_20000.pth"
)

# ==============================
# Validate Configuration
# ==============================
if [ ${#CONFIGS[@]} -ne ${#CHECKPOINTS[@]} ]; then
    echo "Error: CONFIGS and CHECKPOINTS arrays must have the same length!"
    exit 1
fi

DATE=$(date +"%Y%m%d-%H%M%S")
LOG_FILE="${LOG_DIR}/oneig_eval_all_${DATE}.log"

ROOT_DIR="./"
ONEIG_DATA_DIR="$ROOT_DIR/data/OneIG-Benchmark"
ACCEL_CFG="$ROOT_DIR/scripts/evaluation/default_config.yaml"
OUTPUT_BASE="$ROOT_DIR/eval/OneIG/images"

echo "===============================" | tee -a "$LOG_FILE"
echo "OneIG Benchmark Batch Evaluation" | tee -a "$LOG_FILE"
echo "===============================" | tee -a "$LOG_FILE"
echo "Total models to evaluate: ${#CONFIGS[@]}" | tee -a "$LOG_FILE"
echo "Image size: ${IMAGE_SIZE}x${IMAGE_SIZE}" | tee -a "$LOG_FILE"
echo "Language: $LANGUAGE" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "===============================" | tee -a "$LOG_FILE"

# ==============================
# Batch Evaluation Loop
# ==============================
for i in "${!CONFIGS[@]}"; do
    CONFIG_PATH="${CONFIGS[$i]}"
    CHECKPOINT_PATH="${CHECKPOINTS[$i]}"

    # Extract model and checkpoint names
    PARENT_DIR=$(basename "$(dirname "$CHECKPOINT_PATH")")
    FILE_NAME=$(basename "$CHECKPOINT_PATH" .pth)
    CKPT_NAME="${PARENT_DIR}_${FILE_NAME}"

    echo "" | tee -a "$LOG_FILE"
    echo ">>> [Model $i] CONFIG=$CONFIG_PATH, CHECKPOINT=$CHECKPOINT_PATH" | tee -a "$LOG_FILE"
    echo ">>> Model Name: $CKPT_NAME" | tee -a "$LOG_FILE"

    # ------------------------------
    # Generate Images
    # ------------------------------
    echo ">>> Generating images for OneIG benchmark..." | tee -a "$LOG_FILE"

    accelerate launch --config_file "$ACCEL_CFG" \
        "$ROOT_DIR/scripts/evaluation/oneig_t2i_eval.py" \
        --data "$ONEIG_DATA_DIR/OneIG-Bench.csv" \
        --model_name "$CKPT_NAME" \
        --language "$LANGUAGE" \
        --config "$CONFIG_PATH" \
        --checkpoint "$CHECKPOINT_PATH" \
        --batch_size 12 \
        --output "$OUTPUT_BASE" \
        --height $IMAGE_SIZE \
        --width $IMAGE_SIZE \
        --seed 42 \
        2>&1 | tee -a "$LOG_FILE"

    echo "✅ OneIG generation completed" | tee -a "$LOG_FILE"
    echo "📁 Output: $OUTPUT_BASE/**/$CKPT_NAME" | tee -a "$LOG_FILE"
    echo "===== [Done] $CKPT_NAME =====" | tee -a "$LOG_FILE"
    sleep 5  # Optional: prevent resource conflicts
done

echo "" | tee -a "$LOG_FILE"
echo "===============================" | tee -a "$LOG_FILE"
echo "✅ All models evaluated!" | tee -a "$LOG_FILE"
echo "📄 Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Next steps:" | tee -a "$LOG_FILE"
echo "1. Use official OneIG evaluation tools to calculate metrics" | tee -a "$LOG_FILE"
echo "2. Visit: https://github.com/TIGER-AI-Lab/OneIG for evaluation code" | tee -a "$LOG_FILE"
echo "===============================" | tee -a "$LOG_FILE"