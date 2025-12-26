MASTER_ADDR="127.0.0.1"
MASTER_PORT=29508
NUM_NODES=1
NUM_GPUS_PER_NODE=8


export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export WORKSPACE="./"
export TORCH_DISTRIBUTED_DEBUG="INFO"

config_file=./configs/vgtae_intervl3/vlvae_intervl3_p28_448px_stage1.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /mmu-vcg-hdd/guojiahao/miniconda3/envs/openuni-12.3/bin/torchrun \
    --nproc_per_node=$NUM_GPUS_PER_NODE \
    --nnodes=$NUM_NODES \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    scripts/train_vgtae_imagenet.py \
    config=$config_file

