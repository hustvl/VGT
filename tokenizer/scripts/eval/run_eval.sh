#!/bin/bash
# Configuration file paths
CONFIG_PATH="./configs/vgtae_intervl3/vlvae_intervl3_p28_448px_stage1.yaml"
CHECKPOINT_PATH=

# export PYTHONPATH="${PYTHONPATH}:/path/to/vgt/tokenizer"
CONFIG_PATH="./configs/vgtae_intervl3/vlvae_intervl3_p28_448px_stage1.yaml"
CHECKPOINT_PATH="./checkpoints/VGTAE_intervl3_stage1/checkpoint-100000/unwrapped_model/pytorch_model.bin"

python -m accelerate.commands.launch \
    --num_processes=$(nvidia-smi --list-gpus | wc -l) \
    --num_machines=$NUM_MACHINES \
    --machine_rank=$MACHINE_RANK \
    --main_process_ip=$MAIN_PROCESS_IP \
    --main_process_port=$MAIN_PROCESS_PORT \
    --mixed_precision=bf16 \
    tokenizer/scripts/eval/eval_resconstruct.py \
    --config_path $CONFIG_PATH \
    --checkpoint_path $CHECKPOINT_PATH \
    --batch_size 32 \
    --model_image_size 256 \
    --eval_image_size 256 \
    --buffer_size 8 \
    --enable_rfid \
    --enable_inception_score

echo "Evaluation completed!"