import torch
from src.models.vgt_internvl3 import VGT_InternVL3
from src.models.adaptive_metaquery import AdaptiveMetaQuery
from src.models.flow_head import FlowMatchingHead
from src.models.internvl3.modeling_internvl_chat import InternVLChatModel
from mmengine.config import read_base

with read_base():
    from .processors import \
        prompt_template, tokenizer, internvl3_model_name_or_path, image_size

model = dict(
    type=VGT_InternVL3,
    lmm=dict(
        type=InternVLChatModel.from_pretrained,
        pretrained_model_name_or_path=internvl3_model_name_or_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
    ),
    adaptive_metaquery=dict(
        type=AdaptiveMetaQuery,
        mode="truncation",
        crop_mode="center", 
        max_size = (64,64),
        encoding_type ="learnable",
    ),
    image_head=dict(
            type=FlowMatchingHead,
            dim=1536,
            layers=12,
    ),
    
    vgt_ae = dict(
        mllm_path = "OpenGVLab/InternVL3-1B",
        dc_ae_path = "mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers",
        encoder_norm=True,
    ),

    # Tokenizer and prompt
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    
    latent_patch_size = 1,

    # input
    max_length=1024,
)