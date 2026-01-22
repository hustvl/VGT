#!/bin/bash
# Configuration file paths
# export PYTHONPATH="${PYTHONPATH}:/path/to/vgt/tokenizer"
CONFIG_PATH="./configs/vgtae_intervl3/vlvae_intervl3_p28_448px_stage2.yaml"
CHECKPOINT_PATH="./ckpts/vgt_ae/vgt_ae_internvl3.pth"

NUM_MACHINES=1
MACHINE_RANK=0
MAIN_PROCESS_IP=127.0.0.1
MAIN_PROCESS_PORT=23912

python -m accelerate.commands.launch \
    --num_processes=$(nvidia-smi --list-gpus | wc -l) \
    --num_machines=$NUM_MACHINES \
    --machine_rank=$MACHINE_RANK \
    --main_process_ip=$MAIN_PROCESS_IP \
    --main_process_port=$MAIN_PROCESS_PORT \
    --mixed_precision=bf16 \
    eval_resconstruct.py\
    --config_path $CONFIG_PATH \
    --checkpoint_path $CHECKPOINT_PATH \
    --batch_size 16 \
    --buffer_size 8 \
    --enable_rfid \
    --enable_inception_score

echo "Evaluation completed!"