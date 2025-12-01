"""
OpenUni-MAR InternVL3修改版本 - 使用改进的AdaptiveMetaQuery
基于internvl3_sana_hf.py，集成MAR的patch-wise生成思想
修改版本：删除special token，添加宽高比token，2D截取模式
使用AdaptiveMetaQuery解决DDP训练问题
"""

import math
from turtle import pos
import torch
from timm.models.layers import trunc_normal_
import torch.nn as nn
import random
import torch.nn.functional as F
from torch.nn.modules.module import T
from xtuner.registry import BUILDER
from mmengine.model import BaseModel
from mmengine.logging import print_log
from torch.nn.utils.rnn import pad_sequence
from xtuner.model.utils import guess_load_checkpoint
from mmengine.runner.checkpoint import load_checkpoint
from peft import LoraConfig
from mmengine.dist import is_main_process, get_rank, get_world_size
from einops import rearrange
from src.models.openuni.dinov3_feature_extractor import create_dinov3_extractor
# 导入我们实现的组件
import numpy as np
from src.models.vae.unified_vae_wrapper import UnifiedVAEWrapper
from src.models.openuni.mar_diffusion_head import AdaptiveMARDiffusionHead
from src.models.openuni.generation_scheduler import get_generation_scheduler
from transformers.cache_utils import DynamicCache
from typing import Optional
from src.utils import pad_input_ids
# from src.utils import ensure_special_tokens
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def layer_norm_2d(input: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # input.shape = (bsz, c, h, w)
    _input = input.permute(0, 2, 3, 1)
    _input = F.layer_norm(_input, _input.size()[-1:], None, None, eps)
    _input = _input.permute(0, 3, 1, 2)
    return _input

def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            names = name.split('.')
            lora_module = names[0] if len(names) == 1 else names[-1]
            if lora_module == '0':
                lora_module = 'to_out.0'
            lora_module_names.add(lora_module)
    return list(lora_module_names)

def build_mlp(hidden_size, projector_dim, z_dim):
    """构建REPA投影头MLP"""
    return nn.Sequential(
        nn.Linear(hidden_size, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, z_dim),
    )

def layer_norm(input: torch.Tensor, normalized_shape: torch.Size, eps: float = 1e-6) -> torch.Tensor:
    return F.layer_norm(input, normalized_shape, None, None, eps)

def perceptual_loss_dino(dinov3_patch_features, vae_conditions, reduction='mean'):
    """
    计算基于 DINOv3 patch 特征的感知 loss

    Args:
        vae_conditions: [B, 256, D]
        dinov3_patch_features: [B, 256, D]
        reduction: 'mean' or 'sum'
    """
    # 特征归一化
    vae_norm = F.normalize(vae_conditions, dim=-1)
    dino_norm = F.normalize(dinov3_patch_features, dim=-1)

    # Patch 维度的 L2 距离
    loss = (vae_norm - dino_norm).pow(2).sum(-1)  # [B, 256]
    
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()

    return loss
    
def add_interleaved_noise(
    gt_projected: torch.Tensor,
    interleaved_noise_ratio: tuple[float, float],
    interleaved_noise_range: tuple[float, float]
) -> torch.Tensor:
    """
    对输入 token 向量进行随机位置加噪。

    Args:
        gt_projected (torch.Tensor): [B, L, D] token 表示
        interleaved_noise_ratio (tuple): (min_ratio, max_ratio)，决定加噪 token 比例
        interleaved_noise_range (tuple): (min_noise, max_noise)，决定加噪强度

    Returns:
        torch.Tensor: 加噪后的 token 张量
    """
    B, L, D = gt_projected.shape
    device = gt_projected.device
    gt_noised = gt_projected.clone()

    for b in range(B):
        # 随机采样比例并确定加噪数量
        r = random.uniform(*interleaved_noise_ratio)
        N = max(1, int(r * L))

        # 生成随机 index 掩码
        noise_indices = torch.randperm(L, device=device)[:N]

        # 从噪声范围采样噪声强度
        y = random.uniform(*interleaved_noise_range)

        # 生成随机噪声
        noise = torch.randn_like(gt_projected[b, noise_indices])

        # 按公式加噪
        gt_noised[b, noise_indices] = (1 - y) * gt_projected[b, noise_indices] + y * noise

    return gt_noised

class OpenUniMARInternVL3(BaseModel):
    """
    OpenUni-MAR InternVL3修改版本 - 基于InternVL3 + SANA，集成MAR patch-wise生成
    使用改进的AdaptiveMetaQuery解决DDP训练参数更新问题
    
    主要修改：
    1. 删除AdaptiveMetaQuery中的special token
    2. 在prepare_forward_input中添加宽高比token处理
    3. truncation模式改为2D区域截取
    4. 更新相关的长度计算和提取方法
    
    核心架构:
    1. InternVL3作为冻结的MLLM
    2. 改进的AdaptiveMetaQuery支持插值和截取两种模式
    3. MAR DiffusionHead实现patch-wise生成
    4. UnifiedVAEWrapper统一VAE接口
    """
    
    def __init__(self,
                 lmm,
                 tokenizer,
                 prompt_template,
                 image_head,
                 vae_wrapper,
                 adaptive_metaquery,
                 pretrained_pth=None,
                 use_activation_checkpointing=True,
                 train_llm=True,
                 train_vit=True,
                 vit_input_size=448,
                 max_length=2048,
                 latent_patch_size=1,
                 fm_head_batch_mul=4,
                 replace_ratio_range=(0.05, 0.3),
                 # REPA参数 - 简化版本
                 dinov3_model_name="vit_base_patch16_dinov3.lvd1689m",
                 repa_loss_weight=0.0,
                 loss_weight_llm_repa=0.0,
                 #lora
                 lora_modules=None,  # ["to_k", "to_q", "to_v"],
                 lora_rank=8,
                 lora_alpha=8,
                 # 新增参数
                 interleaved_noise_ratio=0.3, # gt的 30% 比例才加噪，其他是干净的
                 interleaved_shuffle=False,  # 是否打乱query和gt的顺序
                 interleaved_noise_range=None,  # 对gt部分加噪的强度范围，如(0.1, 0.2)
                 gt_condition_loss_weight=0.0,  # gt condition的辅助loss权重
                 **kwargs):
        super().__init__()
        
        if is_main_process():
            print_log(f"OpenUni-MAR InternVL3修改版模型初始化:")
            print_log(f"  VAE配置: {vae_wrapper}")
            print_log(f"latent_patch_size: {latent_patch_size}")
            print_log(f"  宽高比token: 启用")
        self.use_activation_checkpointing = use_activation_checkpointing
        self.vit_input_size = vit_input_size
        self.max_length = max_length
        self.train_llm = train_llm
        self.train_vit = train_vit
        self.latent_patch_size = latent_patch_size
        # REPA参数 - 简化版本
        self.dinov3_model_name = dinov3_model_name
        self.loss_weight_llm_repa = loss_weight_llm_repa
        self.repa_loss_weight = repa_loss_weight
        self.fm_head_batch_mul = fm_head_batch_mul

        self.replace_ratio_range = replace_ratio_range
        
        # 新增参数
        self.interleaved_shuffle = interleaved_shuffle
        self.interleaved_noise_range = interleaved_noise_range
        self.gt_condition_loss_weight = gt_condition_loss_weight
        self.interleaved_noise_ratio = interleaved_noise_ratio

        # 初始化MLLM (InternVL3)
        self.lmm = BUILDER.build(lmm)
        self.lmm.requires_grad_(False)

        if train_llm:
            self.lmm.language_model.requires_grad_(True)
        
        if train_vit:
            self.lmm.vision_model.requires_grad_(True)
            self.lmm.mlp1.requires_grad_(True)
        else:
            self.lmm.language_model.lm_head.requires_grad_(False)
            del self.lmm.vision_model
            del self.lmm.mlp1
        
        # 1. 统一VAE接口 - 使用SANA VAE
        self.vae_wrapper = BUILDER.build(vae_wrapper)
        
        # 1.5. DINOv3特征提取器（如果启用REPA或CLS token）
        if repa_loss_weight > 0.0 or loss_weight_llm_repa > 0.0:
            self.dinov3_extractor = create_dinov3_extractor(
                model_name=dinov3_model_name,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            # 获取特征维度用于后续配置
            self.dinov3_feature_dim = self.dinov3_extractor.feature_dim
            
            if is_main_process():
                print_log(f"DINOv3特征提取器配置:")
                print_log(f"  模型: {dinov3_model_name}")
                print_log(f"  特征维度: {self.dinov3_feature_dim}")
                print_log(f"repa_loss: {self.repa_loss_weight}")
        else:
            self.dinov3_extractor = None
            self.dinov3_feature_dim = None  # 默认值
        
        if loss_weight_llm_repa > 0:
            self.llm_repa_projector = build_mlp(self.llm.config.hidden_size, self.llm.config.hidden_size*2, self.dinov3_feature_dim)

        # 2. 改进的AdaptiveMetaQuery系统
        adaptive_metaquery.update(
            mllm_embed_dim=self.llm.config.hidden_size)
        self.adaptive_metaquery = BUILDER.build(adaptive_metaquery)
        
        token_dim = self.vae_wrapper.latent_dim*self.latent_patch_size**2
        # 4. 图像输入投影器
        self.image_in_projector = nn.Linear(token_dim, self.llm.config.hidden_size)
        self.image_in_projector.weight.data.normal_(mean=0.0, std=0.02)
        self.image_in_projector.bias.data.zero_()
        
        # 5. 图像输出投影器
        self.image_out_projector = nn.Linear(self.llm.config.hidden_size, self.llm.config.hidden_size)
        self.image_out_projector.weight.data.normal_(mean=0.0, std=0.02)
        self.image_out_projector.bias.data.zero_()
        
        # 6. Flow Matching Head
        if image_head is not None:
            repa_encoder_depth = max(2, int(image_head.layers * 1 / 2))
            image_head.update(
                input_dim=token_dim, 
                cond_dim=self.llm.config.hidden_size,
                enable_repa=self.repa_loss_weight>0.0,
                repa_encoder_depth=repa_encoder_depth,
                dinov3_feature_dim=self.dinov3_feature_dim,
            )
            self.image_head = BUILDER.build(image_head)
        else:
            raise ValueError("image_head config is required")


        # Tokenizer和prompt相关
        if tokenizer is not None:
            self.tokenizer = BUILDER.build(tokenizer)
        else:
            raise ValueError("tokenizer config is required")
            
        if prompt_template is not None:
            self.prompt_template = prompt_template
            self.image_token_id = self.tokenizer.convert_tokens_to_ids(prompt_template['IMG_CONTEXT_TOKEN'])
        else:
            raise ValueError("prompt_template config is required")
            
            
        self.register_buffer('vit_mean', torch.tensor(IMAGENET_MEAN), persistent=False)
        self.register_buffer('vit_std', torch.tensor(IMAGENET_STD), persistent=False)

        # 激活检查点
        if use_activation_checkpointing:
            self.llm.enable_input_require_grads()
            self.gradient_checkpointing_enable()

        if lora_modules is not None:
            if lora_modules == 'auto':
                lora_modules = find_all_linear_names(self.lmm.language_model)
            # import pdb; pdb.set_trace()
            transformer_lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                init_lora_weights="gaussian",
                target_modules=lora_modules,
            )
            self.lmm.language_model.add_adapter(transformer_lora_config)

        # 加载预训练权重
        if pretrained_pth is not None:
            loaded_checkpoint = load_checkpoint(
                model=self,
                filename=pretrained_pth,
                map_location='cpu',
                strict=True,
            )
            print_log(f"加载的 checkpoint 包含的键：{list(loaded_checkpoint.keys())}")
            if 'iter' in loaded_checkpoint:
                print_log(f"Checkpoint 对应的迭代次数：{loaded_checkpoint['iter']}")
            if 'epoch' in loaded_checkpoint:
                print_log(f"Checkpoint 对应的 epoch：{loaded_checkpoint['epoch']}")
        
        self.to(dtype=self.dtype)
        
        if is_main_process():
            print_log(f"OpenUni-MAR InternVL3修改版模型初始化完成!")
            self._print_architecture_summary()

    def _print_architecture_summary(self):
        """打印架构摘要"""
        print_log(f"\n=== OpenUni-MAR InternVL3修改版架构摘要 ===")
        print_log(f"  VAE维度: {self.vae_wrapper.latent_dim}")
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print_log(f"\n模型参数统计:")
        print_log(f"  总参数: {total_params:,}")
        print_log(f"  可训练参数: {trainable_params:,}")
        
    @property
    def llm(self):
        return self.lmm.language_model

    def gradient_checkpointing_enable(self):
        self.llm.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()
        
    @property
    def device(self):
        return self.llm.device

    @property
    def dtype(self):
        return self.llm.dtype

    def train(self: T, mode: bool = True) -> T:
        super().train(mode=mode)
        self.vae_wrapper.vae.train(mode=False)
        if not mode:
            self.gradient_checkpointing_disable()
        return self

    @torch.no_grad()
    def pixels_to_latents(self, x):
        # scaling_factor = self.vae_wrapper.vae.config.scaling_factor
        z = self.vae_wrapper.encode(x)
        assert len(z.shape) == 4, f"Invalid shape: {z.shape}"
        return z

    @torch.no_grad()
    def latents_to_pixels(self, z):
        # scaling_factor = self.vae_wrapper.vae.config.scaling_factor
        x_rec = self.vae_wrapper.decode(z)
        assert len(x_rec.shape) == 4, f"Invalid shape: {x_rec.shape}"
        return x_rec

    def prepare_forward_input(self,
                                queries,
                                inputs_embeds=None,
                                input_ids=None,
                                attention_mask=None):

        # 获取batch size
        batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        b, l, _ = queries.shape
        assert l > 0
        attention_mask = attention_mask.to(device=self.device, dtype=torch.bool)
        attention_mask = torch.cat([
            attention_mask, attention_mask.new_ones(b, l)
        ], dim=1)
        position_ids = torch.cumsum(attention_mask, dim=1) - 1
        position_ids[position_ids < 0] = 0

        # prepare context
        if inputs_embeds is None:
            input_ids = input_ids.to(self.device)
            inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([inputs_embeds, queries], dim=1)

        inputs_embeds = inputs_embeds.to(dtype=self.dtype)

        inputs = dict(inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids)

        return inputs


    def patchify(self, img: torch.Tensor, patch_size: int):
        """
        使用指定patch_size进行patchify
        img: (bsz, C, H, W)
        返回: (bsz, H*W/patch_size**2, patch_size**2*C)
        """
        bsz, c, h, w = img.shape
        p = patch_size
        h_, w_ = h // p, w // p

        img = img.reshape(bsz, c, h_, p, w_, p)
        img = torch.einsum("nchpwq->nhwcpq", img)
        x = img.reshape(bsz, h_ * w_, c * p**2)
        return x, (h_, w_)

    def unpatchify(self, x: torch.Tensor, patch_size: int, h: int = None, w: int = None):
        """
        使用指定patch_size进行unpatchify
        x: (bsz, H*W/patch_size**2, patch_size**2*C)
        返回: (bsz, C, H, W)
        """
        bsz = x.shape[0]
        p = patch_size
        c = self.vae_wrapper.latent_dim
        if h is None and w is None:
            h_ = w_ = int(x.shape[1] ** 0.5)
        else:
            h_, w_ = h, w
        assert h_ * w_ == x.shape[1], f"Invalid sequence length {x.shape[1]}."

        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum("nhwcpq->nchpwq", x)
        img = x.reshape(bsz, c, h_ * p, w_ * p)
        return img

    def patchify_dinov3_features(self, dinov3_features: torch.Tensor, target_h: int, target_w: int):
        """
        将DINOv3特征重塑为与VAE patches对应的格式
        Args:
            dinov3_features: [B,feature_dim,H,W] DINOv3的14x14 patches (224/16=14)
            target_h: VAE高度 (patch数量)
            target_w: VAE宽度 (patch数量)
        Returns:
            reshaped_features: [B, vae_h*vae_w, feature_dim] 与VAE patches对应的特征
        """
        B, feature_dim, dinov3_h, dinov3_w = dinov3_features.shape
        # 插值到VAE patch尺寸
        if target_h != dinov3_h or target_w != dinov3_w:
            dinov3_features = F.interpolate(
                dinov3_features, 
                size=(target_h, target_w), 
                mode='bilinear', 
                align_corners=False
            )  # [B, feature_dim, vae_h, vae_w]
        
        # 重塑为patch序列
        dinov3_patches = rearrange(dinov3_features, 'b c h w -> b (h w) c').contiguous()
        return dinov3_patches

    # @torch.no_grad()
    def get_semantic_features(self, pixel_values):
        """获取图像的语义特征（保持InternVL3的处理方式）"""
        # pixel_values: [-1, 1]
        pixel_values = (pixel_values + 1.0) / 2     # [0, 1]
        pixel_values = pixel_values - self.vit_mean.view(1, 3, 1, 1)
        pixel_values = pixel_values / self.vit_std.view(1, 3, 1, 1)

        pixel_values = F.interpolate(pixel_values, size=(self.vit_input_size, self.vit_input_size),
                                     mode='bilinear')
        vit_embeds = self.lmm.extract_feature(pixel_values)

        return vit_embeds

    @torch.no_grad()
    def prepare_text_conditions(self, prompt, cfg_prompt=None):
        """推理时的文本条件准备接口"""
        if cfg_prompt is None:
            cfg_prompt = self.prompt_template['CFG']
        else:
            cfg_prompt = self.prompt_template['GENERATION'].format(input=cfg_prompt.strip())
        prompt = self.prompt_template['GENERATION'].format(input=prompt.strip())
        
        # all_prompts = [
        #     self.prompt_template['INSTRUCTION'].format(input=prompt) + self.prompt_template['IMG_START_TOKEN'],
        #     self.prompt_template['INSTRUCTION'].format(input=cfg_prompt) + self.prompt_template['IMG_START_TOKEN'],
        # ]
        all_prompts = [
            self.prompt_template['INSTRUCTION'].format(input=prompt),
            self.prompt_template['INSTRUCTION'].format(input=cfg_prompt),
        ]

        input_ids = [self.tokenizer.encode(p, add_special_tokens=True, return_tensors='pt')[0]
                     for p in all_prompts]

        input_ids, attention_mask = pad_input_ids(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
            padding_mode="right"
        )

        return dict(input_ids=input_ids.to(self.device),
                    attention_mask=attention_mask.to(self.device))

    @torch.no_grad()
    def prepare_batch_text_conditions(self, prompts, cfg_prompts=None):
        """
        批量准备文本条件，支持多个prompt同时处理
        
        Args:
            prompts: list[str], 多个文本提示
            cfg_prompts: list[str] or None, 对应的CFG提示，如果为None则使用默认CFG
            
        Returns:
            dict: 包含批量处理后的 input_ids 和 attention_mask
        """
        if not isinstance(prompts, list):
            prompts = [prompts]
            
        if cfg_prompts is None:
            cfg_prompts = [self.prompt_template['CFG']] * len(prompts)
        elif not isinstance(cfg_prompts, list):
            cfg_prompts = [cfg_prompts] * len(prompts)
            
        all_prompts = []
        all_cfg_prompts = []
        
        for prompt, cfg_prompt in zip(prompts, cfg_prompts):
            # 处理prompt
            if cfg_prompt == self.prompt_template['CFG']:
                cfg_prompt = self.prompt_template['CFG']
            else:
                cfg_prompt = self.prompt_template['GENERATION'].format(input=cfg_prompt.strip())
            prompt = self.prompt_template['GENERATION'].format(input=prompt.strip())

            all_prompts.append(
                self.prompt_template['INSTRUCTION'].format(input=prompt) # + self.prompt_template['IMG_START_TOKEN']
            )
            all_cfg_prompts.append(
                self.prompt_template['INSTRUCTION'].format(input=cfg_prompt) # + self.prompt_template['IMG_START_TOKEN']
            )
        

        # 分别编码prompt和CFG prompt
        prompt_input_ids = [self.tokenizer.encode(p, add_special_tokens=True, return_tensors='pt')[0]
                           for p in all_prompts]
        cfg_prompt_input_ids = [self.tokenizer.encode(p, add_special_tokens=True, return_tensors='pt')[0]
                               for p in all_cfg_prompts]
        
        # 使用左padding方式，与训练数据保持一致
        prompt_input_ids, prompt_attention_mask = pad_input_ids(
            prompt_input_ids + cfg_prompt_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
            padding_mode="right"
        )

        return dict(
            input_ids=prompt_input_ids.to(self.device),
            attention_mask=prompt_attention_mask.to(self.device),
        )

    def _compute_repa_loss(self, dinov3_features, model_features):
        """
        计算REPA对齐损失
        Args:
            dinov3_features: [B*N, feature_dim] DINOv3特征
            model_features: [B*N, feature_dim] 模型中间特征
        Returns:
            repa_loss: REPA对齐损失
        """
        # L2归一化
        dinov3_norm = F.normalize(dinov3_features, dim=-1)  # [B*N, feature_dim]
        model_norm = F.normalize(model_features, dim=-1)    # [B*N, feature_dim]
        
        # 负余弦相似度（最大化相似度）
        cos_sim = (dinov3_norm * model_norm).sum(dim=-1)  # [B*N]
        repa_loss = -cos_sim.mean()
        
        return repa_loss
    
    def interleave_queries_latents(self, input_querys, latents):
        """
        input_querys: [B, L, D]
        latents: [B, L, D]
        返回: [B, 2*L, D]，交错排列
        """
        B, L, D = input_querys.shape

        # reshape 为 [B, L, 1, D] 与 [B, L, 1, D]
        q = input_querys.unsqueeze(2)          # [B, L, 1, D]
        l = latents.unsqueeze(2)       # [B, L, 1, D]

        # 拼接 dim=2 → [B, L, 2, D]
        interleaved = torch.cat([q, l], dim=2)

        # flatten dim=1+2 → [B, 2*L, D]
        interleaved = interleaved.view(B, 2*L, D)
        return interleaved

    def generate_embeding(self, B):
        generate_id = self.tokenizer.encode(self.prompt_template['IMG_START_TOKEN'], add_special_tokens=True, return_tensors='pt')[0].to(device=self.device)
        generate_embeding = self.llm.get_input_embeddings()(generate_id) #[1, dim]
        return generate_embeding.expand(B, -1, -1)

    def text2image_loss(self, data_dict):
        """文本到图像的训练损失 - 新的interleaved训练方式"""
        if 'image_latents' in data_dict:
            image_latents = data_dict['image_latents'].to(dtype=self.dtype, device=self.device)
        else:
            pixel_values = data_dict['pixel_values'].to(dtype=self.dtype, device=self.device)
            image_latents = self.pixels_to_latents(pixel_values)

        loss_mar = {}

        # 1. 直接flatten latents (normalization已在VAE wrapper中处理)
        image_latents_seq, target_shape = self.patchify(image_latents, patch_size=self.latent_patch_size)

        # 2.5. 提取DINOv3特征（如果启用REPA或CLS token）
        dinov3_patch_features = None
        dinov3_cls_features = None
        if (self.repa_loss_weight > 0.0) and 'pixel_values' in data_dict:
            pixel_values = data_dict['pixel_values'].to(dtype=self.dtype, device=self.device)
            with torch.no_grad():
                dinov3_output = self.dinov3_extractor(pixel_values)
                dinov3_patch_features = dinov3_output['features']  # [B, num_patches, feature_dim]
                dinov3_cls_features = dinov3_output['cls_token']         # [B, feature_dim]
            target_image_h, target_image_w = image_latents.shape[2], image_latents.shape[3]
            dinov3_patch_features = self.patchify_dinov3_features(dinov3_patch_features, target_h = target_shape[0], target_w = target_shape[1])
            dinov3_patch_features = dinov3_patch_features.reshape(-1, dinov3_patch_features.shape[-1])
        
        # 3. 准备MLLM输入
        input_ids = data_dict['input_ids'].to(self.device)
        attention_mask = data_dict['attention_mask'].to(self.device)
        b, l, _ = image_latents_seq.shape
        
        # 4. 生成改进的AdaptiveMetaQuery
        meta_queries, pos_embed = self.adaptive_metaquery(target_h=target_shape[0], target_w=target_shape[1])
        meta_queries = meta_queries+pos_embed
        meta_queries = meta_queries.expand(b, -1, -1).to(device=self.device, dtype=self.dtype)
        pos_embed = pos_embed.expand(b, -1, -1).to(device=self.device, dtype=self.dtype)

        # 5. 构造interleaved序列: [query1, inproj(gt1), query2, inproj(gt2), ...]
        gt_projected = self.image_in_projector(image_latents_seq)  # [B, L, hidden_dim]
        
        # 对gt部分加噪（如果启用）
        if self.interleaved_noise_range is not None:
            # gt_projected = self.add_noise(gt_projected) 
            gt_projected = add_interleaved_noise(gt_projected, self.interleaved_noise_ratio, self.interleaved_noise_range) 
            
        gt_projected = gt_projected + pos_embed

        if self.interleaved_shuffle:
            indices = torch.randperm(gt_projected.shape[1]) #tensor([3, 1, 2, 0, 4])
        else:
            indices = torch.arange(gt_projected.shape[1]) #tensor([0, 1, 2, 3, 4])

        # ================== indices 同步打乱 ==================
        meta_queries_shuffled = meta_queries[:, indices, :]
        gt_projected_shuffled = gt_projected[:, indices, :]

        interleaved_seq = self.interleave_queries_latents(
            input_querys=meta_queries_shuffled,
            latents=gt_projected_shuffled
        )  # [B, 2*L, hidden_dim]

        recover_indices = torch.argsort(indices)  # [L], meta_queries_shuffled[recover_indices] 恢复原始顺序
        
        # 7. MLLM前向传播
        mllm_inputs = self.prepare_forward_input(
            queries=torch.cat([self.generate_embeding(b), interleaved_seq] ,dim=1),
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        output = self.llm.model(**mllm_inputs, 
            return_dict=True,
            output_hidden_states=True
        )

        # 8. 提取hidden states并还原位置
        hidden_states = output.last_hidden_state[:, -interleaved_seq.shape[1]:, :]  # [B, 2*L, hidden_dim]

        # ================== interleaved_seq 分离 query 和 gt ==================
        query_seq_hidden = hidden_states[:, 0::2, :]
        gt_seq_hidden    = hidden_states[:, 1::2, :]
        query_seq_hidden = query_seq_hidden[:, recover_indices, :]
        gt_seq_hidden    = gt_seq_hidden[:, recover_indices, :]

        # 10. 分别过output projector
        query_conditions = self.image_out_projector(query_seq_hidden)  # [B, L, hidden_dim]
        gt_conditions = self.image_out_projector(gt_seq_hidden)        # [B, L, hidden_dim]
        
        # 11. 计算query condition的主要loss
        query_conditions_flat = query_conditions.view(-1, query_conditions.shape[-1])
        target_flat = image_latents_seq.contiguous().view(-1, image_latents_seq.shape[-1])
        mask_flat = torch.ones(b*l, device=self.device)  # 所有位置都计算loss
        
        if self.fm_head_batch_mul > 1:
            target_flat = target_flat.repeat(self.fm_head_batch_mul, 1)
            mask_flat = mask_flat.repeat(self.fm_head_batch_mul)
            query_conditions_flat = query_conditions_flat.repeat(self.fm_head_batch_mul, 1)
            if dinov3_patch_features is not None:
                dinov3_patch_features = dinov3_patch_features.repeat(self.fm_head_batch_mul, 1)
        
        loss_mar["loss_text2image"], repa_feature = self.image_head(
            target=target_flat.to(query_conditions_flat.dtype), 
            c=query_conditions_flat, 
            mask=mask_flat
        )
        
        # 12. 计算gt condition的辅助loss（如果启用）
        if self.gt_condition_loss_weight > 0.0:
            gt_conditions_flat = gt_conditions.view(-1, gt_conditions.shape[-1])
            if self.fm_head_batch_mul > 1:
                gt_conditions_flat = gt_conditions_flat.repeat(self.fm_head_batch_mul, 1)
            
            gt_loss, _ = self.image_head(
                target=target_flat.to(gt_conditions_flat.dtype),
                c=gt_conditions_flat,
                mask=mask_flat
            )
            loss_mar["loss_gt_auxiliary"] = self.gt_condition_loss_weight * gt_loss
        
        if self.repa_loss_weight > 0.0 and dinov3_patch_features is not None:
            loss_mar["loss_flow_repa"] = self.repa_loss_weight * self._compute_repa_loss(dinov3_patch_features, repa_feature)

        if self.loss_weight_llm_repa > 0.0 and dinov3_patch_features is not None:
            # 取1/2 处的中间 hidden state
            model_output = output.hidden_states[len(output.hidden_states)*1//2][:, -interleaved_seq.shape[1]:, :][:, 0::2, :]
            metaquery_repa = model_output[:, recover_indices, :]
            loss_mar["loss_llm_repa"] = self.loss_weight_llm_repa * self._compute_repa_loss(dinov3_patch_features, 
                                                                    self.llm_repa_projector(metaquery_repa))

        return loss_mar

    def image2image_loss(self, data_dict):
        """图像到图像的训练损失"""
        pass

    def compute_loss(self, data_dict, vae_loss=False, diff_loss=True):
        """计算损失（保持原有接口）"""
        losses = {}
        
        # 根据数据类型计算对应损失
        for data_type in ['text2image', 'image2image']:
            if data_type in data_dict:
                losses[f'loss_{data_type}'] = getattr(self, f'{data_type}_loss')(data_dict[data_type])
        
        if len(losses) == 0:
            if 'pixel_values_src' in data_dict:
                losses = self.image2image_loss(data_dict)
            else:
                losses = self.text2image_loss(data_dict)

        return losses

    def forward(self, data, data_samples=None, mode='loss'):
        """主前向传播接口（保持原有接口）"""
        model_dtype = next(self.llm.parameters()).dtype
        with torch.cuda.amp.autocast(dtype=model_dtype, enabled=True):
            if mode == 'loss':
                return self.compute_loss(data_dict=data)
            else:
                raise NotImplementedError

    # @torch.no_grad()
    # def generate(
    #     self,
    #     input_ids=None,
    #     inputs_embeds=None,
    #     attention_mask=None,
    #     cfg_scale=4.5,
    #     cfg2_scale=1.0,
    #     num_steps=30,
    #     height=256,
    #     width=256,
    #     progress_bar=True,
    #     total_step = 256,
    #     cfg_schedule="linear",
    #     scheduler_type='random',
    #     scheduler_seed=None,
    #     add_pos=True,
    #     **kwargs
    # ):
    #     """
    #     使用DynamicCache优化的MAR生成 - 新的interleaved模式
        
    #     核心思路：
    #     1. 初始化DynamicCache
    #     2. 第一组：计算完整 [text + first_query]，保存到cache
    #     3. 后续组：构造interleaved序列 [prev_query1, prev_sample1, ..., cur_query]
        
    #     KV更新策略：
    #     - text部分 [0:text_len]: 固定不变，直接从cache读取
    #     - interleaved部分: [prev_group_query1, prev_group_sample1, ..., cur_group_query]
    #     - position_ids策略: 为每个query和sample分配连续的位置ID
        
    #     支持的生成调度策略：
    #     - scheduler_type: 'sequential', 'random', 'spiral', 'exponential', 'cosine'
    #     - scheduler_seed: 随机种子，确保可复现性
    #     - total_step: 期望的总步数，用于部署

    #     add_pos: 开启gt 拼接位置
    #     """
        
    #     # ========== 1. 准备输入 ==========
    #     if inputs_embeds is None and input_ids is not None:
    #         inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        
    #     bsz_with_cfg = attention_mask.shape[0]
    #     if cfg_scale == 1.0 and cfg2_scale == 1.0:
    #         # No expansion
    #         bsz = bsz_with_cfg
    #     elif cfg_scale > 1.0 and cfg2_scale == 1.0:
    #         # Text CFG: 2x expansion
    #         bsz = bsz_with_cfg // 2
    #     elif cfg_scale > 1.0 and cfg2_scale > 1.0:
    #         # Text + Image CFG: 3x expansion
    #         bsz = bsz_with_cfg // 3
    #     else:
    #         raise ValueError(f"Invalid CFG config: cfg={cfg_scale}, cfg2={cfg2_scale}")
        
    #     # 计算尺寸
    #     vae_h = height // self.vae_wrapper.stride // self.latent_patch_size
    #     vae_w = width // self.vae_wrapper.stride // self.latent_patch_size
    #     total_tokens = vae_h * vae_w
    #     total_step = min(total_tokens, total_step)
        
    #     # ========== 4. 初始化生成调度器 ==========
    #     # 使用total_step作为期望的总步数
    #     scheduler = get_generation_scheduler(
    #         scheduler_type=scheduler_type,
    #         num_steps=total_step,
    #         total_tokens=total_tokens,
    #         height=vae_h,
    #         width=vae_w,
    #         seed=scheduler_seed
    #     )
        
    #     generation_schedule = scheduler.schedule()  # [[step1_tokens], [step2_tokens], ...]
    #     num_groups = len(generation_schedule)  # 实际的组数由调度器决定
        
    #     text_len = inputs_embeds.shape[1]
        
    #     # ========== 2. 初始化DynamicCache ==========
    #     past_key_values = DynamicCache()
        
    #     # 存储生成结果
    #     generated_latents_list = []
        
    #     if progress_bar:
    #         from tqdm.auto import tqdm
    #         group_iterator = tqdm(range(1, num_groups), desc="Generating with KV cache")
    #     else:
    #         group_iterator = range(1, num_groups)
        
    #     meta_queries, pos_embed = self.adaptive_metaquery(target_h=vae_h, target_w=vae_w)
    #     meta_queries = meta_queries+pos_embed
    #     meta_queries = meta_queries.to(device=self.device, dtype=self.dtype)
    #     pos_embed = pos_embed.to(device=self.device, dtype=self.dtype)

    #     # === 第一组：完整前向计算 ===
    #     # 使用调度器指定的token顺序
    #     first_step_tokens = generation_schedule[0]
    #     current_queries = meta_queries[:, first_step_tokens].expand(
    #         bsz_with_cfg, -1, -1
    #     ).clone().to(device=self.device, dtype=self.dtype)
            
    #     mllm_inputs = self.prepare_forward_input(
    #         queries=torch.cat([self.generate_embeding(bsz_with_cfg), current_queries] ,dim=1),
    #         inputs_embeds=inputs_embeds,
    #         attention_mask=attention_mask
    #     )
    #     position_ids = mllm_inputs["position_ids"]
        
    #     output = self.llm.model(
    #         **mllm_inputs,
    #         past_key_values=past_key_values,
    #         use_cache=True,
    #         return_dict=True
    #     )
        
    #     # cache会自动更新
    #     past_key_values = output.past_key_values
    #     cur_output = output.last_hidden_state[:, -current_queries.shape[1]:, :]
    #     cur_conditions = self.image_out_projector(cur_output)
    #     cur_conditions = cur_conditions.view(-1, cur_conditions.shape[-1])

    #     # 采样
    #     sampled_latents_flat = self.image_head.sample(
    #         c=cur_conditions,
    #         cfg=cfg_scale,
    #         cfg2=1.0,
    #         num_sampling_steps=num_steps,
    #         progress=False,
    #         **kwargs
    #     )


    #     sampled_latents_flat = layer_norm(sampled_latents_flat, sampled_latents_flat.size()[1:])
    #     sampled_latents = sampled_latents_flat.reshape(bsz, len(first_step_tokens), -1)
    #     generated_latents_list.append(sampled_latents)
        
    #     # 为下一轮准备interleaved序列
    #     current_queries = []
    #     # 添加第一组的query和sample
    #     if add_pos:
    #         current_queries.append(self.image_in_projector(sampled_latents).repeat(bsz_with_cfg // bsz, 1, 1)+ pos_embed[:, first_step_tokens].expand(bsz_with_cfg, -1, -1))
    #     else:
    #         current_queries.append(self.image_in_projector(sampled_latents).repeat(bsz_with_cfg // bsz, 1, 1))

    #     # ========== 3. 逐组生成 ==========
    #     for group_idx in group_iterator:
    #         # 使用调度器指定的token顺序
    #         current_step_tokens = generation_schedule[group_idx]
    #         current_group_size = len(current_step_tokens)
    #         input_querys = meta_queries[:, current_step_tokens].expand(bsz_with_cfg, -1, -1).clone()
            
    #         current_queries.append(input_querys)
    #         pos_start = sum(len(step) * 2 for step in generation_schedule[:group_idx])
    #         pos_end = pos_start + current_group_size

    #         current_queries = torch.cat(current_queries, dim=1)
    #         current_pos_ids = torch.tensor(
    #             [text_len+len(generated_latents_list)*current_queries.shape[1]+i for i in range(current_queries.shape[1])],
    #             dtype=torch.long,
    #             device=self.device
    #         ).unsqueeze(0).expand(bsz_with_cfg, -1)
            
    #         # ========== KV缓存策略 ==========
    #         # === 后续组：更新cache + 计算当前query ===
    #         output = self.llm.model(
    #             inputs_embeds=current_queries,
    #             position_ids=current_pos_ids,
    #             past_key_values=past_key_values,
    #             use_cache=True,
    #             return_dict=True
    #         )
            
    #         past_key_values = output.past_key_values
            
    #         # ========== 4. 提取输出并采样 ==========
    #         # 提取当前组的输出
    #         cur_output = output.last_hidden_state[:, -current_group_size:, :]
    #         cur_conditions = self.image_out_projector(cur_output)
    #         cur_conditions = cur_conditions.view(-1, cur_conditions.shape[-1])
            
                    
    #         if cfg_schedule == "linear":
    #             cfg_iter = 1 + (cfg_scale - 1) * (num_groups - len(generated_latents_list)) / num_groups
    #             cfg2_iter = 1 + (cfg2_scale - 1) * (num_groups - len(generated_latents_list)) / num_groups
    #         elif cfg_schedule == "constant":
    #             cfg_iter = cfg_scale
    #             cfg2_iter = cfg2_scale

    #         # 采样
    #         sampled_latents_flat = self.image_head.sample(
    #             c=cur_conditions,
    #             cfg=cfg_iter,
    #             cfg2=cfg2_iter,
    #             num_sampling_steps=num_steps,
    #             progress=False,
    #             **kwargs
    #         )


    #         if progress_bar:
    #             group_iterator.set_description(
    #                 f"Group {group_idx}/{num_groups-1} | CFG={cfg_iter:.2f}"
    #             )
            
    #         sampled_latents_flat = layer_norm(sampled_latents_flat, sampled_latents_flat.size()[1:])
    #         sampled_latents = sampled_latents_flat.reshape(bsz, current_group_size, -1)
    #         generated_latents_list.append(sampled_latents)
            
    #         # 为下一轮准备interleaved序列
    #         current_queries = []
    #         current_pos_ids = []
            
    #         # [query1, sample_latent1, query2, sample_latent3....]
    #         if add_pos:
    #             current_generates = self.image_in_projector(sampled_latents).repeat(bsz_with_cfg // bsz, 1, 1) + pos_embed[:, current_step_tokens].expand(bsz_with_cfg, -1, -1)
    #         else:
    #             current_generates = self.image_in_projector(sampled_latents).repeat(bsz_with_cfg // bsz, 1, 1)
    #         current_queries.append(current_generates)
    #         # current_pos_ids.extend([text_len+i for i in range(pos_start, pos_start+current_generates.shape[1])])

        
    #     # ========== 5. 合并和解码 ==========
    #     # 需要根据原始顺序重新排列生成的latents
    #     final_latents_seq = torch.cat(generated_latents_list, dim=1)
    #     # 创建逆映射，将调度顺序映射回原始顺序
    #     original_order = torch.zeros(total_tokens, dtype=torch.long, device=self.device)
    #     current_idx = 0
    #     for step_tokens in generation_schedule:
    #         for token_idx in step_tokens:
    #             original_order[token_idx] = current_idx
    #             current_idx += 1
        
    #     # 重新排列latents到原始顺序
    #     generated_latents_seq = final_latents_seq[:, original_order, :]
            
    #     generated_latents = self.unpatchify(
    #         generated_latents_seq,
    #         patch_size=self.latent_patch_size,
    #         h=vae_h,
    #         w=vae_w
    #     )
        
    #     generated_images = self.latents_to_pixels(z=generated_latents)
        
    #     # 调整尺寸
    #     cur_height, cur_width = generated_images.shape[-2:]
    #     if cur_height != height or cur_width != width:
    #         generated_images = F.interpolate(
    #             generated_images,
    #             size=(height, width),
    #             mode='bilinear',
    #             align_corners=False
    #         )
        
    #     return generated_images

    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        cfg_scale=4.5,
        cfg2_scale=1.0,
        num_steps=30,
        height=256,
        width=256,
        progress_bar=True,
        total_step = 256,
        cfg_schedule="linear",
        scheduler_type='random',
        scheduler_seed=None,
        add_pos=True,
        **kwargs
    ):
        """
        使用DynamicCache优化的MAR生成 - 新的interleaved模式
        
        核心思路：
        1. 初始化DynamicCache
        2. 第一组：计算完整 [text + first_query]，保存到cache
        3. 后续组：构造interleaved序列 [prev_query1, prev_sample1, ..., cur_query]
        
        KV更新策略：
        - text部分 [0:text_len]: 固定不变，直接从cache读取
        - interleaved部分: [prev_group_query1, prev_group_sample1, ..., cur_group_query]
        - position_ids策略: 为每个query和sample分配连续的位置ID
        
        支持的生成调度策略：
        - scheduler_type: 'sequential', 'random', 'spiral', 'exponential', 'cosine'
        - scheduler_seed: 随机种子，确保可复现性
        - total_step: 期望的总步数，用于部署

        add_pos: 开启gt 拼接位置
        """
        
        # ========== 1. 准备输入 ==========
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        
        bsz_with_cfg = attention_mask.shape[0]
        if cfg_scale == 1.0 and cfg2_scale == 1.0:
            # No expansion
            bsz = bsz_with_cfg
        elif cfg_scale > 1.0 and cfg2_scale == 1.0:
            # Text CFG: 2x expansion
            bsz = bsz_with_cfg // 2
        elif cfg_scale > 1.0 and cfg2_scale > 1.0:
            # Text + Image CFG: 3x expansion
            bsz = bsz_with_cfg // 3
        else:
            raise ValueError(f"Invalid CFG config: cfg={cfg_scale}, cfg2={cfg2_scale}")
        
        # 计算尺寸
        vae_h = height // self.vae_wrapper.stride // self.latent_patch_size
        vae_w = width // self.vae_wrapper.stride // self.latent_patch_size
        total_tokens = vae_h * vae_w
        if total_step < total_tokens:
            muti_mode = True
        else:
            muti_mode = False

        total_step = min(total_tokens, total_step)
        
        # ========== 4. 初始化生成调度器 ==========
        # 使用total_step作为期望的总步数
        scheduler = get_generation_scheduler(
            scheduler_type=scheduler_type,
            num_steps=total_step,
            total_tokens=total_tokens,
            height=vae_h,
            width=vae_w,
            seed=scheduler_seed
        )
        
        generation_schedule = scheduler.schedule()  # [[step1_tokens], [step2_tokens], ...]
        num_groups = len(generation_schedule)  # 实际的组数由调度器决定
        
        text_len = inputs_embeds.shape[1]
        
        # ========== 2. 初始化DynamicCache ==========
        past_key_values = DynamicCache()
        
        # 存储生成结果
        generated_latents_list = []
        
        if progress_bar:
            from tqdm.auto import tqdm
            group_iterator = tqdm(range(1, num_groups), desc="Generating with KV cache")
        else:
            group_iterator = range(1, num_groups)
        
        meta_queries, pos_embed = self.adaptive_metaquery(target_h=vae_h, target_w=vae_w)
        meta_queries = meta_queries+pos_embed
        meta_queries = meta_queries.to(device=self.device, dtype=self.dtype)
        pos_embed = pos_embed.to(device=self.device, dtype=self.dtype)

        # === 第一组：完整前向计算 ===
        # 使用调度器指定的token顺序
        first_step_tokens = generation_schedule[0]
        current_queries = meta_queries[:, first_step_tokens].expand(
            bsz_with_cfg, -1, -1
        ).clone().to(device=self.device, dtype=self.dtype)

        mllm_inputs = self.prepare_forward_input(
            queries=torch.cat([self.generate_embeding(bsz_with_cfg), current_queries] ,dim=1),
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )
        position_ids = mllm_inputs["position_ids"]
        
        output = self.llm.model(
            **mllm_inputs,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True
        )
        
        # cache会自动更新
        past_key_values = output.past_key_values
        cur_output = output.last_hidden_state[:, -current_queries.shape[1]:, :]
        cur_conditions = self.image_out_projector(cur_output)
        cur_conditions = cur_conditions.view(-1, cur_conditions.shape[-1])

        # 采样
        sampled_latents_flat = self.image_head.sample(
            c=cur_conditions,
            cfg=cfg_scale,
            cfg2=1.0,
            num_sampling_steps=num_steps,
            progress=False,
            **kwargs
        )


        sampled_latents_flat = layer_norm(sampled_latents_flat, sampled_latents_flat.size()[1:])
        sampled_latents = sampled_latents_flat.reshape(bsz, len(first_step_tokens), -1)
        generated_latents_list.append(sampled_latents)
        
        # 为下一轮准备interleaved序列
        current_queries = []
        # 添加第一组的query和sample
        # if add_pos:
        if muti_mode:
            cur_in = self.interleave_queries_latents(
                input_querys=meta_queries[:, first_step_tokens].expand(bsz_with_cfg, -1, -1),
                latents=self.image_in_projector(sampled_latents).repeat(bsz_with_cfg // bsz, 1, 1)+ pos_embed[:, first_step_tokens].expand(bsz_with_cfg, -1, -1)
            )
        else:
            cur_in = self.image_in_projector(sampled_latents).repeat(bsz_with_cfg // bsz, 1, 1)+ pos_embed[:, first_step_tokens].expand(bsz_with_cfg, -1, -1)
        current_queries.append(cur_in)
        
        # else:
            # current_queries.append(self.image_in_projector(sampled_latents).repeat(bsz_with_cfg // bsz, 1, 1))

        # ========== 3. 逐组生成 ==========
        for group_idx in group_iterator:
            # 使用调度器指定的token顺序
            current_step_tokens = generation_schedule[group_idx]
            current_group_size = len(current_step_tokens)
            input_querys = meta_queries[:, current_step_tokens].expand(bsz_with_cfg, -1, -1).clone()
            
            current_queries.append(input_querys)
            pos_start = sum(len(step) * 2 for step in generation_schedule[:group_idx])
            pos_end = pos_start + current_group_size

            current_queries = torch.cat(current_queries, dim=1)
            current_pos_ids = torch.tensor(
                [text_len+len(generated_latents_list)*current_queries.shape[1]+i for i in range(current_queries.shape[1])],
                dtype=torch.long,
                device=self.device
            ).unsqueeze(0).expand(bsz_with_cfg, -1)
            
            # ========== KV缓存策略 ==========
            # === 后续组：更新cache + 计算当前query ===
            output = self.llm.model(
                inputs_embeds=current_queries,
                position_ids=current_pos_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
            )
            
            past_key_values = output.past_key_values
            
            # ========== 4. 提取输出并采样 ==========
            # 提取当前组的输出
            cur_output = output.last_hidden_state[:, -current_group_size:, :]
            cur_conditions = self.image_out_projector(cur_output)
            cur_conditions = cur_conditions.view(-1, cur_conditions.shape[-1])
            
                    
            if cfg_schedule == "linear":
                cfg_iter = 1 + (cfg_scale - 1) * (num_groups - len(generated_latents_list)) / num_groups
                cfg2_iter = 1 + (cfg2_scale - 1) * (num_groups - len(generated_latents_list)) / num_groups
            elif cfg_schedule == "constant":
                cfg_iter = cfg_scale
                cfg2_iter = cfg2_scale

            # 采样
            sampled_latents_flat = self.image_head.sample(
                c=cur_conditions,
                cfg=cfg_iter,
                cfg2=cfg2_iter,
                num_sampling_steps=num_steps,
                progress=False,
                **kwargs
            )


            if progress_bar:
                group_iterator.set_description(
                    f"Group {group_idx}/{num_groups-1} | CFG={cfg_iter:.2f}"
                )
            
            sampled_latents_flat = layer_norm(sampled_latents_flat, sampled_latents_flat.size()[1:])
            sampled_latents = sampled_latents_flat.reshape(bsz, current_group_size, -1)
            generated_latents_list.append(sampled_latents)
            
            # 为下一轮准备interleaved序列
            current_queries = []
            current_pos_ids = []
            
            # [query1, sample_latent1, query2, sample_latent3....]
            if add_pos:
                current_generates = self.image_in_projector(sampled_latents).repeat(bsz_with_cfg // bsz, 1, 1) + pos_embed[:, current_step_tokens].expand(bsz_with_cfg, -1, -1)
            else:
                current_generates = self.image_in_projector(sampled_latents).repeat(bsz_with_cfg // bsz, 1, 1)

            if muti_mode:
                cur_in = self.interleave_queries_latents(
                    input_querys=meta_queries[:, current_step_tokens].expand(bsz_with_cfg, -1, -1),
                    latents=current_generates
                )
            else:
                cur_in = current_generates
            current_queries.append(cur_in)
            # current_pos_ids.extend([text_len+i for i in range(pos_start, pos_start+current_generates.shape[1])])

        
        # ========== 5. 合并和解码 ==========
        # 需要根据原始顺序重新排列生成的latents
        final_latents_seq = torch.cat(generated_latents_list, dim=1)
        # 创建逆映射，将调度顺序映射回原始顺序
        original_order = torch.zeros(total_tokens, dtype=torch.long, device=self.device)
        current_idx = 0
        for step_tokens in generation_schedule:
            for token_idx in step_tokens:
                original_order[token_idx] = current_idx
                current_idx += 1
        
        # 重新排列latents到原始顺序
        generated_latents_seq = final_latents_seq[:, original_order, :]
            
        generated_latents = self.unpatchify(
            generated_latents_seq,
            patch_size=self.latent_patch_size,
            h=vae_h,
            w=vae_w
        )
        
        generated_images = self.latents_to_pixels(z=generated_latents)
        
        # 调整尺寸
        cur_height, cur_width = generated_images.shape[-2:]
        if cur_height != height or cur_width != width:
            generated_images = F.interpolate(
                generated_images,
                size=(height, width),
                mode='bilinear',
                align_corners=False
            )
        
        return generated_images

    def get_model_summary(self):
        """获取模型摘要"""
        return {
            "architecture": "OpenUni-MAR-InternVL3-Modified",
            "mllm": "InternVL3",
            "vae_config": self.vae_wrapper.get_config_summary(),
            "diffusion_head": {
                "input_dim": self.mar_diffusion_head.input_dim,
                "vae_dim": self.vae_wrapper.latent_dim
            },
            "modifications": {
                "special_token": "removed",
                "aspect_ratio_token": "enabled",
                "truncation_mode": "2D_region_crop"
            }
        }
