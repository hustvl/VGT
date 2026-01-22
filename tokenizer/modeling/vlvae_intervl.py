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
import copy
from modeling.quantizer.quantizer import DiagonalGaussianDistribution
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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

def random_init_fn(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
        
        
class VLVAE_interVL3_Tokenizer(nn.Module):
    """
    UniLIP Tokenizer - 基于 InternVL3 + DC-AE 的图像重建模型
    
    模型结构：
    1. InternVL3 ViT 编码器（冻结或可训练）
    2. 像素重排 + MLP 投影
    3. 3层残差块 + 降维 MLP
    4. DC-AE 解码器
    
    """
    
    def __init__(self, 
                 mllm_path: str = "/path/to/pretrain/OpenGVLab/InternVL3-1B",
                 dc_ae_path: str = "/path/to/pretrain/mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers",
                 embed_dim: int = 32,
                 encoder_norm: bool = False,
                 max_noise_strength: float = 0.1,
                 scale_embeding: float = 1.0,
                 kl: bool = False,
                 ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.encoder_norm = encoder_norm
        self.max_noise_strength = max_noise_strength
        self.scale_embeding = scale_embeding
        self.kl = kl
        print(f"encoder_norm:  {encoder_norm}, self.scale_embeding:{self.scale_embeding}")
        # 1. 加载 InternVL3 模型
        print(f"Loading InternVL3 from {mllm_path}")
        model = AutoModel.from_pretrained(
            mllm_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        self.encoder = model.vision_model
        disable_dropout(self.encoder)
        self.mlp1 = model.mlp1

        self.teacher = copy.deepcopy(self.encoder).requires_grad_(False)
        self.teacher_mlp1 = copy.deepcopy(self.mlp1).requires_grad_(False)

        llm_hidden_size = model.config.llm_config.hidden_size
        
        # 2. 下游变换模块
        down_blocks = []
        for i in range(3):
            down_blocks.append(ResBlock(llm_hidden_size))
        self.down_blocks = nn.ModuleList(down_blocks)
        
        if self.kl:
            self.down_mlp = nn.Sequential(
                nn.LayerNorm(llm_hidden_size),
                nn.Linear(llm_hidden_size, embed_dim*2),
                nn.GELU(),
                nn.Linear(embed_dim*2, embed_dim*2),
            )
            self.quantize = DiagonalGaussianDistribution
        else:
            self.down_mlp = nn.Sequential(
                nn.LayerNorm(llm_hidden_size),
                nn.Linear(llm_hidden_size, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )
        

        # 3. 加载 DC-AE 解码器
        print(f"Loading DC-AE from {dc_ae_path}")
        dc_ae = AutoencoderDC.from_pretrained(dc_ae_path, torch_dtype=torch.float32)
        self.decoder = dc_ae.decoder
        
        self.register_buffer('vit_mean', torch.tensor(IMAGENET_MEAN), persistent=False)
        self.register_buffer('vit_std', torch.tensor(IMAGENET_STD), persistent=False)

        # 优化内存格式
        for name, param in self.decoder.named_parameters():
            if len(param.data.shape) == 4:
                param.data = param.data.to(memory_format=torch.channels_last)
        
        # 初始化权重
        self._init_weights()
        del model
                    
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

    def get_semantic_features(self, encoder, mlp,  x):
        if x.min() < 0.0: #[-1,1]
            x = (x + 1.0) / 2     # [0, 1]
            x = x - self.vit_mean.view(1, 3, 1, 1)
            x = x / self.vit_std.view(1, 3, 1, 1)

        # ViT 编码
        vit_embeds = encoder.embeddings(x)
        for encoder_layer in encoder.encoder.layers:
            vit_embeds = encoder_layer(vit_embeds)
        
        # 去掉 CLS token，保留 patch tokens
        vit_embeds = vit_embeds[:, 1:, :].contiguous().float()
        
        # 重塑为网格形状
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        
        # 像素重排：空间尺寸 x2，通道 /4
        vit_embeds = pixel_shuffle(vit_embeds)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        
        # 通过 mlp1 投影
        vit_embeds = mlp(vit_embeds)

        return vit_embeds


    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        编码图像到 latent 空间
        Args:
            x: [B, 3, H, W] 输入图像，范围 [-1, 1] 或 ImageNet 归一化
        Returns:
            latents: [B, 32, H//28, W//28] latent 特征
        """
        
        vit_embeds = self.get_semantic_features(self.encoder,self.mlp1,x)
        distill_output = vit_embeds.clone()
        # 通过下游变换模块
        for block in self.down_blocks:
            vit_embeds = block(vit_embeds)
        vit_embeds = self.down_mlp(vit_embeds)
        # 转换为 [B, C, H, W] 格式
        vit_embeds = vit_embeds.permute(0, 2, 1).contiguous()
        b, c, hw = vit_embeds.shape
        vit_embeds = vit_embeds.view(b, c, int(math.sqrt(hw)), int(math.sqrt(hw)))

        vit_embeds = vit_embeds.float()*self.scale_embeding

        result_dict = {}
        if self.kl:
            mean, logvar = torch.chunk(vit_embeds, 2, dim=1)
            if self.encoder_norm:
                mean = layer_norm_2d(mean)

            z = torch.cat([mean, logvar], dim=1).contiguous()
            posteriors = self.quantize(z)
            z_quantized = posteriors.sample()
            result_dict.update(posteriors=posteriors)
        else:
            z_quantized = vit_embeds
            if self.encoder_norm:
                z_quantized = layer_norm_2d(z_quantized)
            
        
        # 3. 随机扰动（仅训练时）
        if self.training and self.max_noise_strength > 0:
            z_quantized = add_stochastic_perturbation(
                z_quantized, 
                max_noise_strength=self.max_noise_strength,
                training=self.training
            )

        if self.training:
            if hasattr(self, 'teacher'):
                vit_embeds_teacher = self.get_semantic_features(self.teacher, self.teacher_mlp1, x).detach()
                distill_loss = F.mse_loss(distill_output,vit_embeds_teacher,reduction="mean",)
                # import pdb;pdb.set_trace()
                result_dict["distill_loss"] = distill_loss
            else:
                result_dict["distill_loss"] = torch.zeros((), device=distill_output.device)   
        
        return z_quantized.float(), result_dict
    
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
        return image_size // self.stride  # UniLIP 的下采样倍率是 28
    
from modeling.modules.base_model import BaseModel
import json
from omegaconf import OmegaConf
from pathlib import Path

from huggingface_hub import PyTorchModelHubMixin
from transformers import AutoTokenizer, AutoModel, AutoConfig, Siglip2VisionConfig, Siglip2VisionModel

class VLVAE_InterVL3_Train(BaseModel, PyTorchModelHubMixin, VLVAE_interVL3_Tokenizer):
    def __init__(self, config):
        # 处理config格式
        if isinstance(config, dict):
            config = OmegaConf.create(config)
        
        # 从config中提取VLVAE_Qwen2_5_Tokenizer所需的参数
        # 假设config.model中包含了这些参数
        model_config = config.model
        
        # 初始化父类VLVAE_Qwen2_5_Tokenizer
        mllm_path = getattr(model_config, 'mllm_path', "/path/to/pretrain/OpenGVLab/InternVL3-1B")
        VLVAE_interVL3_Tokenizer.__init__(
            self,
            mllm_path=mllm_path,
            dc_ae_path=getattr(model_config, 'dc_ae_path', "/path/to/pretrain/mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers"),
            embed_dim=getattr(model_config, 'embed_dim', 32),
            encoder_norm=getattr(model_config, 'encoder_norm', False),
            max_noise_strength=getattr(model_config, 'max_noise_strength', 0.0),
            scale_embeding=getattr(model_config, 'scale_embeding', 1.0),
            kl=getattr(model_config, 'kl', False),
        )

        # 保存config
        self.config = config
        # del self.teacher
        
        self.to(torch.bfloat16)
        # 如果有stage1_ckpt，加载预训练权重
        if hasattr(model_config, 'checkpoint_path') and model_config.checkpoint_path != '':
            print(f"Loading checkpoint from {model_config.checkpoint_path}")
            checkpoint = torch.load(model_config.checkpoint_path, map_location="cpu")
            if "state_dict" in checkpoint:
                msg = self.load_state_dict(checkpoint["state_dict"], strict=False)
            else:
                msg = self.load_state_dict(checkpoint, strict=False)
            print(f"Loaded checkpoint: {msg}")
            
            if is_main_process():
                print(f"load {model_config.checkpoint_path}")
                print("Missing keys:", msg.missing_keys)
                print("Unexpected keys:", msg.unexpected_keys)


    def _save_pretrained(self, save_directory: Path) -> None:
        """保存权重和配置到本地目录"""
        # 转换config为字典并保存
        dict_config = OmegaConf.to_container(self.config)
        file_path = Path(save_directory) / "config.json"
        with open(file_path, 'w') as json_file:
            json.dump(dict_config, json_file, indent=4)
        super()._save_pretrained(save_directory)