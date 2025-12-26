import torch
import os
# /path/to/miniconda3/envs/VGT-12.3/bin/python /path/to/VGT/src/models/vae/tokenizer/scripts/ckpt/covert_to_pth.py
# ==== 配置路径 ====
bin_path = "/path/to/VGT/src/models/vae/tokenizer/checkpoints/vlvae_qwen/vlvae_qwen2_5_p14_448px_stage2/checkpoint-35000/unwrapped_model/pytorch_model.bin"
pth_path = os.path.splitext(bin_path)[0] + ".pth"

# ==== 加载并保存 ====
print(f"Loading HuggingFace .bin file from:\n  {bin_path}")
state_dict = torch.load(bin_path, map_location="cpu")

# 某些 HuggingFace 权重是 {"model": state_dict}，检查一下
if isinstance(state_dict, dict) and "state_dict" in state_dict:
    state_dict = state_dict["state_dict"]
elif isinstance(state_dict, dict) and "model" in state_dict:
    state_dict = state_dict["model"]

# 保存为标准 .pth 格式
torch.save(state_dict, pth_path)

print(f"✅ Converted and saved as:\n  {pth_path}")
print(f"Total parameters: {len(state_dict)}")
