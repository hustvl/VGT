#!/bin/bash
set -e
set -x

# ==============================
# GenEval Benchmark - Batch Evaluation Script
# ==============================
# Usage: Simply run this script without any arguments
#   bash scripts/evaluation/eval_geneval.sh
#
# Configuration: Edit the CONFIGS and CHECKPOINTS arrays below
# ==============================

# ==============================
# Configuration
# ==============================
LOG_DIR="./eval/GenEval/logs"
LOG_DIR=$(realpath "$LOG_DIR")
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

# ==============================
# Validate Configuration
# ==============================
if [ ${#CONFIGS[@]} -ne ${#CHECKPOINTS[@]} ]; then
    echo "Error: CONFIGS and CHECKPOINTS arrays must have the same length!"
    exit 1
fi

DATE=$(date +"%Y%m%d-%H%M%S")
LOG_FILE="${LOG_DIR}/geneval_eval_all_${DATE}.log"

ROOT_DIR="./"
ROOT_DIR=$(realpath "$ROOT_DIR")
GENEVAL_DATA_DIR="$ROOT_DIR/data/geneval"
ACCEL_CFG="$ROOT_DIR/scripts/evaluation/default_config.yaml"

echo "===============================" | tee -a "$LOG_FILE"
echo "GenEval Batch Evaluation" | tee -a "$LOG_FILE"
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

    # Output directories
    OUTPUT_DIR="$ROOT_DIR/eval/GenEval/${MODEL_NAME}/${CKPT_NAME}"
    RESULT_DIR="$ROOT_DIR/eval/GenEval/${MODEL_NAME}/results"
    RESULT_JSON="$RESULT_DIR/${CKPT_NAME}_results.jsonl"

    mkdir -p "$OUTPUT_DIR" "$RESULT_DIR"

    echo "" | tee -a "$LOG_FILE"
    echo ">>> [Model $i] CONFIG=$CONFIG_PATH, CHECKPOINT=$CHECKPOINT_PATH" | tee -a "$LOG_FILE"
    echo ">>> Output: $OUTPUT_DIR" | tee -a "$LOG_FILE"

    # ------------------------------
    # Stage 1: Generate Images
    # ------------------------------
    echo ">>> Stage 1: Generating images..." | tee -a "$LOG_FILE"
    
    accelerate launch --config_file "$ACCEL_CFG" \
        "$ROOT_DIR/scripts/evaluation/gen_eval.py" \
        "$CONFIG_PATH" \
        --checkpoint "$CHECKPOINT_PATH" \
        --batch_size 8 \
        --output "$OUTPUT_DIR" \
        --height $IMAGE_SIZE \
        --width $IMAGE_SIZE \
        --seed 42 \
        2>&1 | tee -a "$LOG_FILE"
    
    echo "✅ Image generation completed" | tee -a "$LOG_FILE"

    # ------------------------------
    # Stage 2: Evaluate Images
    # ------------------------------
    echo ">>> Stage 2: Evaluating generated images..." | tee -a "$LOG_FILE"

    cd "$GENEVAL_DATA_DIR"
        
    $EVAL_ENV evaluation/evaluate_images.py \
        "$OUTPUT_DIR" \
        --outfile "$RESULT_JSON" \
        --model-path "$GENEVAL_DATA_DIR" \
        2>&1 | tee -a "$LOG_FILE"
    
    # ------------------------------
    # Stage 3: Summarize Results
    # ------------------------------
    echo ">>> Stage 3: Summarizing evaluation results..." | tee -a "$LOG_FILE"
    $EVAL_ENV evaluation/summary_scores.py "$RESULT_JSON" \
        2>&1 | tee -a "$LOG_FILE"
    
    cd "$ROOT_DIR"
    
    echo "✅ Evaluation completed" | tee -a "$LOG_FILE"
    echo "📁 Images: $OUTPUT_DIR" | tee -a "$LOG_FILE"
    echo "📄 Results: $RESULT_JSON" | tee -a "$LOG_FILE"

    echo "===== [Done] $MODEL_NAME / $CKPT_NAME =====" | tee -a "$LOG_FILE"
    sleep 5  # Optional: prevent resource conflicts
done

echo "" | tee -a "$LOG_FILE"
echo "===============================" | tee -a "$LOG_FILE"
echo "✅ All models evaluated!" | tee -a "$LOG_FILE"
echo "📄 Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "===============================" | tee -a "$LOG_FILE"
