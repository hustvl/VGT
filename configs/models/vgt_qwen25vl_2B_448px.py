import torch
from src.models.vgt_qwen2_5vl import VGT_Qwen25VL
from src.models.adaptive_metaquery import AdaptiveMetaQuery
from src.models.flow_head import FlowMatchingHead
from transformers import Qwen2_5_VLForConditionalGeneration
from mmengine.config import read_base

with read_base():
    from ..datasets.qwen25vl_2b_448.processors import \
        prompt_template, tokenizer, qwen2_5_vl_model_name_or_path, image_size

dcae_path = "mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers"

model = dict(
    type=VGT_Qwen25VL,
    lmm=dict(
        type=Qwen2_5_VLForConditionalGeneration.from_pretrained,
        pretrained_model_name_or_path=qwen2_5_vl_model_name_or_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation='flash_attention_2',
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
        type = "vgt_pretrain",
        mllm_path = qwen2_5_vl_model_name_or_path,
        dc_ae_path = dcae_path,
        encoder_norm=True,
    ),

    # Tokenizer and prompt
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    
    latent_patch_size = 1,

    # input
    max_length=1024,
)