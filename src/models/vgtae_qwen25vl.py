"""
VGTAE_Qwen25VL - Independent Image Codec
Image reconstruction model based on Qwen2.5-VL + DC-AE, can directly load checkpoint
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from transformers import Qwen2_5_VLForConditionalGeneration
from diffusers.models import AutoencoderDC
from torch.distributions import Uniform
from einops import rearrange
import copy

IMAGENET_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGENET_STD = (0.26862954, 0.26130258, 0.27577711)


def layer_norm_2d(input: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """2D Layer Normalization"""
    _input = input.permute(0, 2, 3, 1)  # (bsz, h, w, c)
    _input = F.layer_norm(_input, _input.size()[-1:], None, None, eps)
    _input = _input.permute(0, 3, 1, 2)  # (bsz, c, h, w)
    return _input


def add_stochastic_perturbation(latents: torch.Tensor, 
                               max_noise_strength: float = 0.1,
                               training: bool = True) -> torch.Tensor:
    """Add stochastic perturbation: z~ = z + α·ε"""
    if not training or max_noise_strength <= 0.0:
        return latents
    
    alpha_dist = Uniform(0, max_noise_strength)
    alpha = alpha_dist.sample((latents.shape[0],)).to(latents.device)
    alpha = alpha.view(-1, 1, 1, 1)
    epsilon = torch.randn_like(latents)
    perturbed_latents = latents + alpha * epsilon
    
    return perturbed_latents


def disable_dropout(model):
    """Disable all dropout layers"""
    for name, module in model.named_modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.p = 0.0
        elif hasattr(module, 'drop_prob'):
            module.drop_prob = 0.0


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


class VGTAE_Qwen25VL(nn.Module):
    """
    VGTAE_Qwen25VL - Independent Image Codec
    
    Functions:
    - encode(): Image -> latent features
    - decode(): Latent features -> image
    
    Usage example:
        model = VGTAE_Qwen25VL(checkpoint_path="path/to/pytorch_model.bin")
        latents = model.encode(images)
        reconstructed = model.decode(latents)
    """
    
    def __init__(self, 
                 mllm_path: str = "Qwen/Qwen2.5-VL-3B-Instruct",
                 dc_ae_path: str = "mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers",
                 checkpoint_path: Optional[str] = None,
                 embed_dim: int = 32,
                 encoder_norm: bool = True,
                 max_noise_strength: float = 0.0,
                 scale_embeding: float = 1.0,
                 freeze_encoder: bool = False,
                 device: str = "cuda",
                 dtype: torch.dtype = torch.bfloat16):
        """
        Args:
            mllm_path: Qwen2.5-VL model path
            dc_ae_path: DC-AE decoder path
            checkpoint_path: Pre-trained weights path (pytorch_model.bin)
            embed_dim: Latent feature dimension
            encoder_norm: Whether to use encoder normalization
            max_noise_strength: Noise strength during training
            scale_embeding: Embedding scaling coefficient
            freeze_encoder: Whether to freeze encoder
            device: Device
            dtype: Data type
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.encoder_norm = encoder_norm
        self.max_noise_strength = max_noise_strength
        self.scale_embeding = scale_embeding
        self.shift_embeding = 0.0
        self.device = device
        self.dtype = dtype
        
        print(f"Initializing VGTAE_Qwen25VL...")
        print(f"  encoder_norm: {encoder_norm}")
        print(f"  scale_embeding: {scale_embeding}")
        print(f"  freeze_encoder: {freeze_encoder}")
        
        # 1. Load Qwen2.5-VL encoder
        print(f"Loading Qwen2.5-VL from {mllm_path}")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            mllm_path,
            torch_dtype=dtype,
            trust_remote_code=False,
            attn_implementation="flash_attention_2",
        )
        
        self.vision_config = model.config.vision_config
        self.encoder = model.visual if hasattr(model, 'visual') else model.vision_model
        disable_dropout(self.encoder)
        
        if freeze_encoder:
            self.encoder.requires_grad_(False)
        
        llm_hidden_size = model.language_model.config.hidden_size
        
        # 2. Downstream transformation modules
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
        
        # 3. Load DC-AE decoder
        print(f"Loading DC-AE from {dc_ae_path}")
        dc_ae = AutoencoderDC.from_pretrained(dc_ae_path, torch_dtype=torch.float32)
        self.decoder = dc_ae.decoder
        
        # 注册归一化参数
        self.register_buffer('vit_mean', torch.tensor(IMAGENET_MEAN), persistent=False)
        self.register_buffer('vit_std', torch.tensor(IMAGENET_STD), persistent=False)
        
        # 优化内存格式
        for name, param in self.decoder.named_parameters():
            if len(param.data.shape) == 4:
                param.data = param.data.to(memory_format=torch.channels_last)
        
        # 初始化权重
        self._init_weights()
        
        # 清理临时变量
        del model, dc_ae
        
        # 4. 加载checkpoint
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        print(f"VGTAE_Qwen25VL initialized successfully!")
    
    def _init_weights(self):
        """Initialize weights for newly added modules"""
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
        """Load pre-trained weights"""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # 处理可能的key前缀
        state_dict = checkpoint
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        
        msg = self.load_state_dict(state_dict, strict=False)
        print(f"Missing keys: {msg.missing_keys}")
        print(f"Unexpected keys: {msg.unexpected_keys}")
    
    def get_semantic_features(self, pixel_values):
        """Extract semantic features"""
        b, c, h, w = pixel_values.shape
        device = pixel_values.device

        patch_size = self.vision_config.patch_size
        spatial_merge_size = self.vision_config.spatial_merge_size
        temporal_patch_size = self.vision_config.temporal_patch_size

        # 扩展时间维度
        pixel_values = pixel_values[:, None].expand(b, temporal_patch_size, c, h, w)

        grid_t = 1
        grid_h, grid_w = h // patch_size, w // patch_size

        # 重排为patches
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

        # 编码
        image_embeds = self.encoder(pixel_values, grid_thw=image_grid_thw)
        image_embeds = rearrange(image_embeds, '(b l) d -> b l d', b=b)

        return image_embeds, (grid_h, grid_w)
    
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode image to latent space
        
        Args:
            x: [B, 3, H, W] Input image, range [-1, 1] or ImageNet normalized
            
        Returns:
            latents: [B, embed_dim, H//stride, W//stride] latent features
        """
        # Normalize input
        if x.min() < 0.0:  # [-1,1]
            x = (x + 1.0) / 2  # [0, 1]
        x = x - self.vit_mean.view(1, 3, 1, 1)
        x = x / self.vit_std.view(1, 3, 1, 1)

        # Extract semantic features
        vit_embeds, (grid_h, grid_w) = self.get_semantic_features(x)
        
        # Actual sequence shape
        grid_h = grid_h // 2
        grid_w = grid_w // 2
        
        # Pass through downstream transformation modules
        for block in self.down_blocks:
            vit_embeds = block(vit_embeds)
        vit_embeds = self.down_mlp(vit_embeds)
        
        # Convert to [B, C, H, W] format
        vit_embeds = vit_embeds.permute(0, 2, 1).contiguous()
        b, c, hw = vit_embeds.shape
        assert grid_h * grid_w == hw, f"{grid_h}*{grid_w} != {hw}, x shape:{x.shape}"
        vit_embeds = vit_embeds.view(b, c, grid_h, grid_w)
        
        # Scale and shift
        vit_embeds = vit_embeds.float() * self.scale_embeding + self.shift_embeding
        
        # Channel-wise normalization
        if self.encoder_norm:
            vit_embeds = layer_norm_2d(vit_embeds)
        
        # Stochastic perturbation (training only)
        if self.training and self.max_noise_strength > 0:
            vit_embeds = add_stochastic_perturbation(
                vit_embeds, 
                max_noise_strength=self.max_noise_strength,
                training=self.training
            )
        
        return vit_embeds.float()
    
    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latent features to image
        
        Args:
            latents: [B, embed_dim, H, W] latent features
            
        Returns:
            images: [B, 3, H*stride, W*stride] reconstructed images, range [-1, 1]
        """
        # 反缩放
        latents = (latents - self.shift_embeding) / self.scale_embeding
        
        # DC-AE 解码
        B, C, H, W = latents.shape
        decoded = self.decoder(latents)
        
        # 插值到目标尺寸
        target_size = (H * self.stride, W * self.stride)
        decoded = F.interpolate(decoded, size=target_size, mode='bilinear', align_corners=False)
        
        return decoded
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Complete encoding-decoding pipeline
        
        Args:
            x: [B, 3, H, W] Input image
            
        Returns:
            reconstructed: [B, 3, H, W] reconstructed image
        """
        latents = self.encode(x)
        reconstructed = self.decode(latents)
        return reconstructed
    
    @property
    def latent_dim(self) -> int:
        """Number of channels in latent features"""
        return self.embed_dim
    
    @property
    def stride(self) -> int:
        return 28
    
    def get_latent_size(self, image_size: int) -> int:
        """Calculate latent size corresponding to given image size"""
        return image_size // self.stride