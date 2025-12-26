export nnodes=${1:-1}
export master_addr=${2:-"localhost"}
export node_rank=${3:-0}

export master_port=${master_port:-"10259"} 
export nproc_per_node=${nproc_per_node:-8}
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8


echo "🔍 distrubution training config check:"
echo "  - total nodes: $nnodes"
echo "  - current node rank: $node_rank"
echo "  - per node gpu num: $nproc_per_node"
echo "  - total gpu num: $((nnodes * nproc_per_node))"

cd /path/to/VGT
export PYTHONPATH=./:$PYTHONPATH

python_gen_config=configs/pretrain/vgt_internvl3_0_6B_448px_pretrain.py

python -m torch.distributed.launch \
    --nnodes=${nnodes} \
    --node_rank=${node_rank} \
    --master_addr=${master_addr} \
    --master_port=${master_port} \
    --nproc_per_node=${nproc_per_node} \
    ./scripts/train.py \
    $python_gen_config \
    --launcher pytorch
