import torch
from src.models.vgt_internvl3 import VGT_InternVL3
from src.models.adaptive_metaquery import AdaptiveMetaQuery
from src.models.flow_head import FlowMatchingHead
from src.models.internvl3.modeling_internvl_chat import InternVLChatModel
from mmengine.config import read_base

with read_base():
    from ..datasets.internvl3_2b_448.processors import \
        prompt_template, tokenizer, internvl3_model_name_or_path, image_size

dcae_path = "mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers"

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
            timeshift=1.4, # √32(latent)/16(base)
    ),
    
    vgt_ae = dict(
        type = "vgt_pretrain",
        mllm_path = "OpenGVLab/InternVL3-1B",
        dc_ae_path = "mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers",
        # checkpoint_path = "./pytorch_model.bin", # from https://huggingface.co/hustvl/vgt_internvl3_1_6B_pretrain extract
        encoder_norm=True,
    ),

    # custom train ae
    # vgt_ae = dict(
    #     config_path = "tokenizer/configs/vgtae_intervl3/vlvae_intervl3_p28_448px_stage2.yaml",
    #     checkpoint_path = "tokenizer/checkpoints/VGTAE_intervl3_stage2/checkpoint-50000/unwrapped_model/pytorch_model.bin"
    # ),

    # Tokenizer and prompt
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    
    latent_patch_size = 1,

    # input
    max_length=1024,
)