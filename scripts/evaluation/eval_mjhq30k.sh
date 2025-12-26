#!/bin/bash
set -e
set -x

# ==============================
# MJHQ-30K Benchmark - Batch Evaluation Script
# ==============================
# Usage: Simply run this script without any arguments
#   bash scripts/evaluation/eval_mjhq30k.sh
#
# Configuration: Edit the CONFIGS and CHECKPOINTS arrays below
# ==============================

# ==============================
# Configuration
# ==============================
LOG_DIR="./eval/MJHQ-30K/logs"
mkdir -p "$LOG_DIR"

IMAGE_SIZE=1024

# Configure models to evaluate (arrays must have same length)
declare -a CONFIGS=(
    "./configs/models/openuni_b_internvl3_1b_sana_0_6b_512_hf.py"
    # Add more configs here
    # "./configs/models/model2.py"
)

declare -a CHECKPOINTS=(
    "work_dirs/openuni_model/iter_10000.pth"
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
LOG_FILE="${LOG_DIR}/mjhq30k_eval_all_${DATE}.log"

ROOT_DIR="./"
MJHQ_DATA_DIR="$ROOT_DIR/data/MJHQ-30K"
ACCEL_CFG="$ROOT_DIR/scripts/evaluation/default_config.yaml"

echo "===============================" | tee -a "$LOG_FILE"
echo "MJHQ-30K Batch Evaluation" | tee -a "$LOG_FILE"
echo "===============================" | tee -a "$LOG_FILE"
echo "Total models to evaluate: ${#CONFIGS[@]}" | tee -a "$LOG_FILE"
echo "Image size: ${IMAGE_SIZE}x${IMAGE_SIZE}" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "===============================" | tee -a "$LOG_FILE"

# ==============================
# Batch Evaluation Loop
# ==============================
for i in "${!CONFIGS[@]}"; do
    CONFIG_PATH="${CONFIGS[$i]}"
    CHECKPOINT_PATH="${CHECKPOINTS[$i]}"

    # Extract model and checkpoint names
    CKPT_NAME=$(basename "$CHECKPOINT_PATH" .pth)
    MODEL_NAME=$(basename "$(dirname "$CHECKPOINT_PATH")")

    # Output directory
    OUTPUT_DIR="$ROOT_DIR/eval/MJHQ-30K/${MODEL_NAME}/${CKPT_NAME}"
    mkdir -p "$OUTPUT_DIR"

    echo "" | tee -a "$LOG_FILE"
    echo ">>> [Model $i] CONFIG=$CONFIG_PATH, CHECKPOINT=$CHECKPOINT_PATH" | tee -a "$LOG_FILE"
    echo ">>> Output: $OUTPUT_DIR" | tee -a "$LOG_FILE"

    # ------------------------------
    # Generate Images
    # ------------------------------
    echo ">>> Generating images for MJHQ-30K benchmark..." | tee -a "$LOG_FILE"

    accelerate launch --config_file "$ACCEL_CFG" \
        "$ROOT_DIR/scripts/evaluation/gen_mjhq30k.py" \
        "$CONFIG_PATH" \
        --checkpoint "$CHECKPOINT_PATH" \
        --meta_data "$MJHQ_DATA_DIR/meta_data.json" \
        --ref_dir "$MJHQ_DATA_DIR/mjhq30k_imgs" \
        --output "$OUTPUT_DIR" \
        --batch_size 1 \
        --height $IMAGE_SIZE \
        --width $IMAGE_SIZE \
        --cfg_scale 4.5 \
        --seed 42 \
        2>&1 | tee -a "$LOG_FILE"

    echo "✅ MJHQ-30K generation completed" | tee -a "$LOG_FILE"
    echo "📁 Output: $OUTPUT_DIR" | tee -a "$LOG_FILE"
    echo "===== [Done] $MODEL_NAME / $CKPT_NAME =====" | tee -a "$LOG_FILE"
    sleep 5  # Optional: prevent resource conflicts
done

echo "" | tee -a "$LOG_FILE"
echo "===============================" | tee -a "$LOG_FILE"
echo "✅ All models evaluated!" | tee -a "$LOG_FILE"
echo "📄 Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Next steps:" | tee -a "$LOG_FILE"
echo "1. Use official MJHQ-30K evaluation tools to calculate metrics" | tee -a "$LOG_FILE"
echo "2. Metrics include: FID, CLIP Score, Aesthetic Score, etc." | tee -a "$LOG_FILE"
echo "===============================" | tee -a "$LOG_FILE"
