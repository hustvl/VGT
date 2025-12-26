"""
UniLIP Tokenizer - 基于 InternVL3 + DC-AE 的图像重建模型
参考 UniLIP 项目的 tokenizer 实现，包装为统一接口
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from omegaconf import OmegaConf
from transformers import AutoModel
from diffusers.models import AutoencoderDC
from torch.distributions import Uniform
from collections import OrderedDict, namedtuple
import re
from timm.models.layers import trunc_normal_
import copy
from huggingface_hub import PyTorchModelHubMixin
from transformers import AutoTokenizer, AutoModel, AutoConfig, Siglip2VisionConfig, Siglip2VisionModel
from einops import rearrange, reduce
# from mmengine.dist import is_main_process, get_rank
IMAGENET_MEAN = (0.48145466, 0.4578275,  0.40821073)
IMAGENET_STD = (0.26862954, 0.26130258, 0.27577711)


import torch.distributed as dist

def is_dist_avail_and_initialized():
    """检查分布式是否可用且已初始化"""
    return dist.is_available() and dist.is_initialized()

def get_rank():
    """获取当前进程的 rank（非分布式则返回 0）"""
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def get_world_size():
    """获取总进程数（非分布式则返回 1）"""
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def is_main_process():
    """判断当前是否为主进程"""
    return get_rank() == 0

def barrier():
    """在分布式环境中进行同步，单机模式下跳过"""
    if is_dist_avail_and_initialized():
        dist.barrier()


def pixel_shuffle(x, scale_factor=0.5):
    """像素重排函数，用于调整特征图尺寸"""
    n, w, h, c = x.size()
    # N, W, H, C --> N, W, H * scale, C // scale
    x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
    # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
    x = x.permute(0, 2, 1, 3).contiguous()
    # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
    x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                int(c / (scale_factor * scale_factor)))
    x = x.permute(0, 2, 1, 3).contiguous()
    return x

def layer_norm_2d(input: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    2D Layer Normalization - 参考NextStep-1实现
    对每个spatial位置的所有channels做layer norm
    """
    # input.shape = (bsz, c, h, w)
    _input = input.permute(0, 2, 3, 1)  # (bsz, h, w, c)
    _input = F.layer_norm(_input, _input.size()[-1:], None, None, eps)
    _input = _input.permute(0, 3, 1, 2)  # (bsz, c, h, w)
    return _input

def add_stochastic_perturbation(latents: torch.Tensor, 
                               max_noise_strength: float = 0.1,
                               training: bool = True) -> torch.Tensor:
    """
    添加随机扰动 - 参考NextStep-1和σ-VAE
    z~ = z + α·ε, where α ~ U[0, γ] and ε ~ N(0, I)
    """
    if not training or max_noise_strength <= 0.0:
        return latents
    
    # α ~ U[0, γ]
    alpha_dist = Uniform(0, max_noise_strength)
    alpha = alpha_dist.sample((latents.shape[0],)).to(latents.device)
    alpha = alpha.view(-1, 1, 1, 1)  # [B, 1, 1, 1]
    
    # ε ~ N(0, I)
    epsilon = torch.randn_like(latents)
    
    # z~ = z + α·ε
    perturbed_latents = latents + alpha * epsilon
    
    return perturbed_latents

import torch.nn as nn

def random_init_fn(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

class ResBlock(nn.Module):
    """残差块，用于特征变换"""
    
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.mlp = nn.Sequential(
            nn.LayerNorm(channels, eps=1e-6),
            nn.Linear(channels, channels, bias=True),
            nn.GELU(),
            nn.Linear(channels, channels, bias=True),
        )

    def forward(self, x):
        return x + self.mlp(x)


from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

def disable_dropout(model):
    """
    将模型中所有 dropout 层彻底关闭（包括 Dropout、Dropout2d、Dropout3d、StochasticDepth、DropPath 等）。
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.p = 0.0
            print(f"Disabled dropout in {name}")
        elif hasattr(module, 'drop_prob'):  # 兼容 DropPath 或 StochasticDepth
            module.drop_prob = 0.0
            print(f"Disabled drop_path/stochastic depth in {name}")

class VLVAE_Qwen2_5_Tokenizer(nn.Module):
    """
    UniLIP Tokenizer - 基于 InternVL3 + DC-AE 的图像重建模型
    
    模型结构：
    1. InternVL3 ViT 编码器（冻结或可训练）
    2. 像素重排 + MLP 投影
    3. 3层残差块 + 降维 MLP
    4. DC-AE 解码器
    
    下采样倍率：28倍（patch_size=14，pixel_shuffle后变为28倍）
    """
    
    def __init__(self, 
                 mllm_path: str = "/mmu-vcg-hdd/guojiahao/pretrain/Qwen/Qwen2.5-VL-3B-Instruct",
                 dc_ae_path: str = "/mmu-vcg-hdd/guojiahao/pretrain/mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers",
                 checkpoint_path: Optional[str] = None,
                 embed_dim: int = 32,
                 encoder_norm: bool = False,
                 max_noise_strength: float = 0.1,
                 scale_embeding: float = 16.6,
                 stage1_ckpt: Optional[str] = '',
                 random_init: bool = False,
                 force_bf16: bool = False,
                 ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.max_noise_strength = max_noise_strength
        self.scale_embeding = scale_embeding
        self.encoder_norm = encoder_norm

        if is_main_process():
            print(f"scale_embeding:{scale_embeding}, max_noise_strength: {max_noise_strength}, encoder_norm: {encoder_norm}, random_init:{random_init}")
            # 1. 加载 InternVL3 模型
            print(f"Loading model from {mllm_path}")
            print(f"Loading DC-AE from {dc_ae_path}")
        # default: Load the model on the available device(s)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            mllm_path,
            # torch_dtype="auto",
            # low_cpu_mem_usage=True,
            # use_flash_attn=True,
            trust_remote_code=False,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        self.vision_config = model.config.vision_config
        # import pdb;pdb.set_trace()
        self.encoder = model.visual if hasattr(model, 'visual') else model.vision_model
        disable_dropout(self.encoder)

        self.teacher = copy.deepcopy(self.encoder)
        disable_dropout(self.teacher)
        
        llm_hidden_size = model.language_model.config.hidden_size
        
        # 2. 下游变换模块
        down_blocks = []
        for i in range(3):
            down_blocks.append(ResBlock(llm_hidden_size))
        self.down_blocks = nn.ModuleList(down_blocks)
        
        self.down_mlp = nn.Sequential(
            nn.LayerNorm(llm_hidden_size),
            nn.Linear(llm_hidden_size, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.scale_embeding = 1.0
        self.shift_embeding = 0.0
        # self.scale_embeding = nn.Parameter(torch.tensor(scale_embeding, dtype=torch.float32))
        # # shift 参数（可训练，初始化为0偏移）
        # self.shift_embeding = nn.Parameter(torch.ones(1).to(dtype=torch.float32))
        # trunc_normal_(self.shift_embeding, std=0.02)
        
        # 3. 加载 DC-AE 解码器
        if force_bf16:
            dc_ae = AutoencoderDC.from_pretrained(dc_ae_path, torch_dtype=torch.float16)
        else:
            dc_ae = AutoencoderDC.from_pretrained(dc_ae_path, torch_dtype=torch.float32)
        self.decoder = dc_ae.decoder

        if random_init:
            if is_main_process():
                print("AutoencoderDC random_init") 
                print("before",next(self.decoder.parameters()).mean()) 
            self.decoder.apply(random_init_fn)
            if is_main_process():
                print("after",next(self.decoder.parameters()).mean()) 
        
        self.register_buffer('vit_mean', torch.tensor(IMAGENET_MEAN), persistent=False)
        self.register_buffer('vit_std', torch.tensor(IMAGENET_STD), persistent=False)

        # 优化内存格式
        for name, param in self.decoder.named_parameters():
            if len(param.data.shape) == 4:
                param.data = param.data.to(memory_format=torch.channels_last)
        
        # 初始化权重
        self._init_weights()

        # 5. 加载预训练权重
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

        # # 如果有stage1_ckpt，加载预训练权重
        # if stage1_ckpt != '':
        #     msg = self.load_state_dict(torch.load(stage1_ckpt), strict=False)
        #     if is_main_process():
        #         print(f"load {stage1_ckpt}")
        #         print("Missing keys:", msg.missing_keys)
        #         print("Unexpected keys:", msg.unexpected_keys)

        # self.encoder = copy.deepcopy(self.teacher)
        self.encoder.requires_grad_(True)
        self.teacher.requires_grad_(False)
        del model

        if force_bf16:
            self.to(torch.bfloat16)
        # self.decoder.gradient_checkpointing_disable()
        if is_main_process():
            self._print_architecture_summary()

    def _print_architecture_summary(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n模型参数统计:")
        print(f"  总参数: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")

    def _init_weights(self):
        """初始化新增模块的权重"""
        for module in [self.down_blocks, self.down_mlp]:
            for m in module.modules():
                if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                    nn.init.trunc_normal_(m.weight.data, mean=0.0, std=0.02)
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.LayerNorm):
                    m.bias.data.zero_()
                    m.weight.data.fill_(1.0)
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载预训练权重"""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        msg = self.load_state_dict(checkpoint, strict=False)
        if is_main_process():
            print(f"Loading checkpoint from {checkpoint_path}")
            print(f"Loaded checkpoint: {msg}")

    def get_semantic_features(self,encoder, pixel_values):
        # pixel_values: imagenet 归一化
        b, c, h, w = pixel_values.shape
        device = pixel_values.device

        patch_size = self.vision_config.patch_size
        spatial_merge_size = self.vision_config.spatial_merge_size
        temporal_patch_size = self.vision_config.temporal_patch_size

        pixel_values = pixel_values[:, None].expand(b, temporal_patch_size, c, h, w)

        grid_t = 1
        grid_h, grid_w = h // patch_size, w // patch_size

        pixel_values = pixel_values.view(
            b,
            grid_t,
            temporal_patch_size,
            c,
            grid_h // spatial_merge_size,
            spatial_merge_size,
            patch_size,
            grid_w // spatial_merge_size,
            spatial_merge_size,
            patch_size,
        )

        pixel_values = rearrange(
            pixel_values, 'b t tp c h m p w n q -> (b t h w m n) (c tp p q)')

        image_grid_thw = torch.tensor([(grid_t, grid_h, grid_w)] * b).to(device).long()

        image_embeds = encoder(pixel_values, grid_thw=image_grid_thw)
        image_embeds = rearrange(image_embeds, '(b l) d -> b l d', b=b)

        return image_embeds, (grid_h, grid_w)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        编码图像到 latent 空间
        Args:
            x: [B, 3, H, W] 输入图像，范围 [-1, 1] 或 ImageNet 归一化
        Returns:
            latents: [B, 32, H//28, W//28] latent 特征
        """
        # import pdb;pdb.set_trace()
        if x.min() < 0.0: #[-1,1]
            x = (x + 1.0) / 2     # [0, 1]
        x = x - self.vit_mean.view(1, 3, 1, 1)
        x = x / self.vit_std.view(1, 3, 1, 1)

        vit_embeds, (grid_h, grid_w) = self.get_semantic_features(self.encoder, x)
        vit_embeds_orige = vit_embeds.clone()
        # //2后是真实的序列 shape
        grid_h = grid_h//2
        grid_w = grid_w//2
        # import pdb;pdb.set_trace()
        # 通过下游变换模块
        for block in self.down_blocks:
            vit_embeds = block(vit_embeds)
        vit_embeds = self.down_mlp(vit_embeds)
        # 转换为 [B, C, H, W] 格式
        vit_embeds = vit_embeds.permute(0, 2, 1).contiguous()
        b, c, hw = vit_embeds.shape
        assert grid_h*grid_w == hw, f"{grid_h}*{grid_w} != {hw}, x shape:{x.shape}"
        vit_embeds = vit_embeds.view(b, c, int(grid_h), int(grid_w))

        # print(vit_embeds.min().item())
        # print(vit_embeds.max().item())
        # print(vit_embeds.std().item())
        # exit(0)
        vit_embeds = vit_embeds * self.scale_embeding + self.shift_embeding  # !!!大致让 std=1.0

        # 2. Channel-wise normalization (NextStep-1风格)
        if self.encoder_norm:
            vit_embeds = layer_norm_2d(vit_embeds)
        
        # 3. 随机扰动（仅训练时）
        if self.training and self.max_noise_strength > 0:
            vit_embeds = add_stochastic_perturbation(
                vit_embeds, 
                max_noise_strength=self.max_noise_strength,
                training=self.training
            )

        if self.training and hasattr(self, 'teacher'):
            vit_embeds_teacher, _ = self.get_semantic_features(self.teacher, x)
            return vit_embeds.float(), (vit_embeds_orige.float(), vit_embeds_teacher.float())
        
        return vit_embeds.float()
    
    def decode(self, latents: torch.Tensor, target_size: Optional[Tuple[int, int]] = (448, 448)) -> torch.Tensor:
        """
        解码 latent 特征到图像
        Args:
            latents: [B, 32, H, W] latent 特征
            target_size: 目标图像尺寸 (H, W)，如果为 None 则根据 latent 尺寸推断
        Returns:
            images: [B, 3, H, W] 重建图像，范围 [-1, 1]
        """
        # DC-AE 解码
        latents = (latents-self.shift_embeding) / self.scale_embeding
        decoded = self.decoder(latents)
        
        # 如果指定了目标尺寸，则插值到目标尺寸
        if target_size is not None:
            decoded = F.interpolate(decoded, size=target_size, mode='bilinear', align_corners=False)
        
        return decoded
    
    
    def forward(self, x: torch.Tensor, target_size: Optional[Tuple[int, int]] = (448, 448)) -> torch.Tensor:
        """
        完整的编码-解码流程
        Args:
            x: [B, 3, H, W] 输入图像
            target_size: 目标输出尺寸
        Returns:
            reconstructed: [B, 3, H, W] 重建图像
        """
        latents = self.encode(x)
        reconstructed = self.decode(latents, target_size)
        return reconstructed
    
    @property
    def latent_dim(self) -> int:
        """latent 特征的通道数"""
        return self.embed_dim
    
    @property
    def stride(self) -> int:
        """下采样倍率"""
        return 28  # patch_size=14，pixel_shuffle后变为28倍
    
    def get_latent_size(self, image_size: int) -> int:
        """计算给定图像尺寸对应的 latent 尺寸"""
        return image_size // self.stride


from modeling.modules.base_model import BaseModel
import json
from omegaconf import OmegaConf
from pathlib import Path

from huggingface_hub import PyTorchModelHubMixin
from transformers import AutoTokenizer, AutoModel, AutoConfig, Siglip2VisionConfig, Siglip2VisionModel

class VLVAE_QWEN2_5_Stage2(BaseModel, PyTorchModelHubMixin, VLVAE_Qwen2_5_Tokenizer):
    def __init__(self, config):
        # 处理config格式
        if isinstance(config, dict):
            config = OmegaConf.create(config)
        
        # 从config中提取VLVAE_Qwen2_5_Tokenizer所需的参数
        # 假设config.model中包含了这些参数
        model_config = config.model
        
        # 初始化父类VLVAE_Qwen2_5_Tokenizer
        mllm_path = getattr(model_config, 'mllm_path', "/mmu-vcg-hdd/guojiahao/pretrain/Qwen/Qwen2.5-VL-3B-Instruct")
        VLVAE_Qwen2_5_Tokenizer.__init__(
            self,
            mllm_path=mllm_path,
            dc_ae_path=getattr(model_config, 'dc_ae_path', "/mmu-vcg-hdd/guojiahao/pretrain/mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers"),
            checkpoint_path=getattr(model_config, 'checkpoint_path', None),
            embed_dim=getattr(model_config, 'embed_dim', 32),
            encoder_norm=getattr(model_config, 'encoder_norm', False),
            max_noise_strength=getattr(model_config, 'max_noise_strength', 0.0),
            scale_embeding=getattr(model_config, 'scale_embeding', 1.0),
            stage1_ckpt=getattr(model_config, 'stage1_ckpt', 1.0),
            random_init=getattr(model_config, 'random_init', False),
            force_bf16=getattr(model_config, 'force_bf16', False),
        )
        
        # 保存config
        self.config = config
        # del self.teacher
        
        # 如果有stage1_ckpt，加载预训练权重
        if hasattr(model_config, 'stage1_ckpt') and model_config.stage1_ckpt != '':
            msg = self.load_state_dict(torch.load(model_config.stage1_ckpt), strict=False)
            if is_main_process():
                print(f"load {model_config.stage1_ckpt}")
                print("Missing keys:", msg.missing_keys)
                print("Unexpected keys:", msg.unexpected_keys)


    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        编码图像到 latent 空间
        Args:
            x: [B, 3, H, W] 输入图像，范围 [-1, 1] 或 ImageNet 归一化
        Returns:
            latents: [B, 32, H//28, W//28] latent 特征
        """
        # import pdb;pdb.set_trace()
        if x.min() < 0.0: #[-1,1]
            x = (x + 1.0) / 2     # [0, 1]
        x = x - self.vit_mean.view(1, 3, 1, 1)
        x = x / self.vit_std.view(1, 3, 1, 1)

        vit_embeds, (grid_h, grid_w) = self.get_semantic_features(self.encoder, x)
        distill_output = vit_embeds.clone()
        # //2后是真实的序列 shape
        grid_h = grid_h//2
        grid_w = grid_w//2
        # import pdb;pdb.set_trace()
        # 通过下游变换模块
        for block in self.down_blocks:
            vit_embeds = block(vit_embeds)
        vit_embeds = self.down_mlp(vit_embeds)
        # 转换为 [B, C, H, W] 格式
        vit_embeds = vit_embeds.permute(0, 2, 1).contiguous()
        b, c, hw = vit_embeds.shape
        assert grid_h*grid_w == hw, f"{grid_h}*{grid_w} != {hw}, x shape:{x.shape}"
        vit_embeds = vit_embeds.view(b, c, int(grid_h), int(grid_w))

        # print(vit_embeds.min().item())
        # print(vit_embeds.max().item())
        # print(vit_embeds.std().item())
        # exit(0)
        vit_embeds = vit_embeds * self.scale_embeding  # !!!大致让 std=1.0

        # 2. Channel-wise normalization (NextStep-1风格)
        if self.encoder_norm:
            vit_embeds = layer_norm_2d(vit_embeds)
        
        # 3. 随机扰动（仅训练时）
        if self.training and self.max_noise_strength > 0:
            vit_embeds = add_stochastic_perturbation(
                vit_embeds, 
                max_noise_strength=self.max_noise_strength,
                training=self.training
            )

        result_dict = {'distill_feat': distill_output}

        if self.training and hasattr(self, 'teacher'):
            vit_embeds_teacher = self.get_semantic_features(self.teacher, x)[0].detach()
            distill_loss = F.mse_loss(distill_output,vit_embeds_teacher,reduction="mean",)
            # import pdb;pdb.set_trace()
            result_dict["distill_loss"] = distill_loss
        
        return vit_embeds.float(), result_dict
    
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        解码 latent 特征到图像
        Args:
            latents: [B, 32, H, W] latent 特征
            target_size: 目标图像尺寸 (H, W)，如果为 None 则根据 latent 尺寸推断
        Returns:
            images: [B, 3, H, W] 重建图像，范围 [-1, 1]
        """
        # DC-AE 解码
        latents = latents / self.scale_embeding
        # with torch.cuda.amp.autocast(dtype=torch.float32):
        B,C,H,W = latents.shape
        decoded = self.decoder(latents)
        # 如果指定了目标尺寸，则插值到目标尺寸
        target_size_ = (H*self.stride, W*self.stride)
        # if target_size_ is not None:
        decoded = F.interpolate(decoded, size=target_size_, mode='bilinear', align_corners=False)

        return decoded


    def forward(self, vit_pixel_values):
        """
        适配接口：完整的前向传播
        
        Args:
            vit_pixel_values: 输入图像 tensor，shape [B, C, H, W]
            
        Returns:
            decode_pixel_values: 重建图像，shape [B, 3, H, W]
            result_dict: 包含额外信息的字典
        """
        # 编码
        vit_embeds, result_dict = self.encode(vit_pixel_values)
        # 解码
        decode_pixel_values = self.decode(vit_embeds)
        
        return decode_pixel_values, result_dict

    def _save_pretrained(self, save_directory: Path) -> None:
        """保存权重和配置到本地目录"""
        # 转换config为字典并保存
        dict_config = OmegaConf.to_container(self.config)
        file_path = Path(save_directory) / "config.json"
        with open(file_path, 'w') as json_file:
            json.dump(dict_config, json_file, indent=4)
        super()._save_pretrained(save_directory)