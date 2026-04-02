python tokenizer/scripts/eval.py \
    --eval_data_dir /path/to/datasets/imagenet/CLS-LOC/val \
    --config_path /path/to/tokenizer/checkpoints/your_run/config.yaml \
    --checkpoint_path /path/to/tokenizer/checkpoints/your_run/checkpoint/unwrapped_model/pytorch_model.bin \
    --batch_size 2 \
    --model_image_size 256 \
    --eval_image_size 256 \
    --device cuda
