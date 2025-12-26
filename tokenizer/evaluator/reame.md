/mmu-vcg-hdd/dusinan/miniconda3/envs/blip3o-next/bin/python /mmu-vcg-hdd/dusinan/kolors-o-research/tokenizer/scripts/eval.py \
    --eval_data_dir /path/to/datasets/imagenet/CLS-LOC/val \
    --config_path /mmu-vcg-hdd/dusinan/kolors-o-research/tokenizer/checkpoints/rae_siglip2_so400m_p16_256px_cs_16384_d_1536_stage1_imagenet/config.yaml \
    --checkpoint_path /mmu-vcg-hdd/dusinan/kolors-o-research/tokenizer/checkpoints/rae_siglip2_so400m_p16_256px_cs_16384_d_1536_stage1_imagenet/checkpoint-100000/unwrapped_model/pytorch_model.bin \
    --batch_size 2 \
    --model_image_size 256 \
    --eval_image_size 256 \
    --device cuda