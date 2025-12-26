"""
VGTAE_InternVL3 - Independent Image Codec
Image reconstruction model based on InternVL3 + DC-AE, can directly load checkpoint
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from transformers import AutoModel
from diffusers.models import AutoencoderDC
from torch.distributions import Uniform

# ImageNet normalization constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def layer_norm_2d(input: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Apply 2D layer normalization.
    
    Args:
        input: Input tensor with shape (batch_size, channels, height, width)
        eps: Small value for numerical stability
        
    Returns:
        Normalized tensor
    """
    _input = input.permute(0, 2, 3, 1)  # (bsz, h, w, c)
    _input = F.layer_norm(_input, _input.size()[-1:], None, None, eps)
    _input = _input.permute(0, 3, 1, 2)  # (bsz, c, h, w)
    return _input


def add_stochastic_perturbation(
    latents: torch.Tensor, 
    max_noise_strength: float = 0.1,
    training: bool = True
) -> torch.Tensor:
    """
    Add stochastic perturbation to latents: z~ = z + α·ε
    where α ~ U[0, γ] and ε ~ N(0, I)
    
    Args:
        latents: Input latent tensor
        max_noise_strength: Maximum noise strength γ
        training: Whether in training mode
        
    Returns:
        Perturbed latents
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


def pixel_shuffle(x: torch.Tensor, scale_factor: float = 0.5) -> torch.Tensor:
    """
    Pixel shuffle operation for feature map resizing.
    
    Args:
        x: Input tensor with shape (N, W, H, C)
        scale_factor: Scale factor for spatial dimensions
        
    Returns:
        Shuffled tensor
    """
    n, w, h, c = x.size()
    # N, W, H, C --> N, W, H * scale, C // scale
    x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
    # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
    x = x.permute(0, 2, 1, 3).contiguous()
    # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
    x = x.view(
        n, 
        int(h * scale_factor), 
        int(w * scale_factor),
        int(c / (scale_factor * scale_factor))
    )
    x = x.permute(0, 2, 1, 3).contiguous()
    return x


def disable_dropout(model):
    """Disable all dropout layers in the model."""
    for name, module in model.named_modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.p = 0.0
        elif hasattr(module, 'drop_prob'):  # For DropPath or StochasticDepth
            module.drop_prob = 0.0


class ResBlock(nn.Module):
    """Residual block for feature transformation."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.mlp = nn.Sequential(
            nn.LayerNorm(channels, eps=1e-6),
            nn.Linear(channels, channels, bias=True),
            nn.GELU(),
            nn.Linear(channels, channels, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x)


class VGTAE_InternVL3(nn.Module):
    """
    VGTAE_InternVL3 - Independent Image Codec
    
    Architecture:
    1. InternVL3 ViT encoder (frozen or trainable)
    2. Pixel shuffle + MLP projection
    3. 3-layer residual blocks + dimension reduction MLP
    4. DC-AE decoder
    
    Downsampling ratio: 28x (patch_size=14, after pixel_shuffle becomes 28x)
    
    Functions:
    - encode(): Image -> latent features
    - decode(): Latent features -> image
    
    Usage example:
        model = VGTAE_InternVL3(checkpoint_path="path/to/pytorch_model.bin")
        latents = model.encode(images)
        reconstructed = model.decode(latents)
    """
    
    def __init__(
        self, 
        mllm_path: str = "OpenGVLab/InternVL3-1B",
        dc_ae_path: str = "mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers",
        checkpoint_path: Optional[str] = None,
        embed_dim: int = 32,
        encoder_norm: bool = False,
        max_noise_strength: float = 0.0,
        scale_embeding: float = 1.0,
        freeze_encoder: bool = True,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16
    ):
        """
        Initialize VGTAE_InternVL3 model.
        
        Args:
            mllm_path: InternVL3 model path or HuggingFace repo
            dc_ae_path: DC-AE decoder path or HuggingFace repo
            checkpoint_path: Pre-trained weights path (pytorch_model.bin)
            embed_dim: Latent feature dimension
            encoder_norm: Whether to apply layer norm on encoder output
            max_noise_strength: Noise strength during training (0.0 = disabled)
            scale_embeding: Embedding scaling coefficient
            freeze_encoder: Whether to freeze encoder weights
            device: Device to load model on
            dtype: Data type for model weights
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.encoder_norm = encoder_norm
        self.max_noise_strength = max_noise_strength
        self.scale_embeding = scale_embeding
        self.device = device
        self.dtype = dtype
        
        print(f"Initializing VGTAE_InternVL3...")
        print(f"  encoder_norm: {encoder_norm}")
        print(f"  freeze_encoder: {freeze_encoder}")
        print(f"  scale_embeding: {scale_embeding}")
        
        # 1. Load InternVL3 encoder
        print(f"Loading InternVL3 from {mllm_path}")
        model = AutoModel.from_pretrained(
            mllm_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            # use_flash_attn=True,
            trust_remote_code=True,
            # attn_implementation="flash_attention_2",
        )
        
        self.encoder = model.vision_model
        disable_dropout(self.encoder)
        self.mlp1 = model.mlp1
        
        if freeze_encoder:
            self.encoder.requires_grad_(False)
            self.mlp1.requires_grad_(False)
        
        llm_hidden_size = model.config.llm_config.hidden_size
        
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
        
        # Register normalization buffers
        self.register_buffer('vit_mean', torch.tensor(IMAGENET_MEAN), persistent=False)
        self.register_buffer('vit_std', torch.tensor(IMAGENET_STD), persistent=False)
        
        # Optimize memory format
        for name, param in self.decoder.named_parameters():
            if len(param.data.shape) == 4:
                param.data = param.data.to(memory_format=torch.channels_last)
        
        # Initialize weights
        self._init_weights()
        
        # Clean up temporary variables
        del model, dc_ae
        
        # 4. Load checkpoint if provided
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        print(f"VGTAE_InternVL3 initialized successfully!")
    
    def _init_weights(self):
        """Initialize weights for newly added modules."""
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
        """
        Load pre-trained weights from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Handle different checkpoint formats
        state_dict = checkpoint
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        
        msg = self.load_state_dict(state_dict, strict=False)
        print(f"Missing keys: {msg.missing_keys}")
        print(f"Unexpected keys: {msg.unexpected_keys}")
    
    def get_semantic_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract semantic features from InternVL3 encoder.
        
        Args:
            x: Input image tensor, ImageNet normalized
            
        Returns:
            vit_embeds: Encoded features
        """
        # ViT encoding
        vit_embeds = self.encoder.embeddings(x)
        for encoder_layer in self.encoder.encoder.layers:
            vit_embeds = encoder_layer(vit_embeds)
        
        # Remove CLS token, keep patch tokens
        vit_embeds = vit_embeds[:, 1:, :].contiguous().float()
        
        # Reshape to grid format
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        
        # Pixel shuffle: spatial size x2, channels /4
        vit_embeds = pixel_shuffle(vit_embeds)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        
        # Project through mlp1
        vit_embeds = self.mlp1(vit_embeds)
        
        return vit_embeds
    
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode image to latent space.
        
        Args:
            x: [B, 3, H, W] Input image, range [-1, 1] or ImageNet normalized
            
        Returns:
            latents: [B, embed_dim, H//28, W//28] latent features
        """
        # Normalize input to ImageNet format
        if x.min() < 0.0:  # Assume [-1, 1] range
            x = (x + 1.0) / 2  # Convert to [0, 1]
        x = x - self.vit_mean.view(1, 3, 1, 1)
        x = x / self.vit_std.view(1, 3, 1, 1)
        
        # Extract semantic features
        vit_embeds = self.get_semantic_features(x)
        
        # Pass through downstream transformation modules
        for block in self.down_blocks:
            vit_embeds = block(vit_embeds)
        vit_embeds = self.down_mlp(vit_embeds)
        
        # Convert to [B, C, H, W] format
        vit_embeds = vit_embeds.permute(0, 2, 1).contiguous()
        b, c, hw = vit_embeds.shape
        vit_embeds = vit_embeds.view(b, c, int(math.sqrt(hw)), int(math.sqrt(hw)))
        
        # Scale embeddings
        vit_embeds = vit_embeds.float() * self.scale_embeding
        
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
        Decode latent features to image.
        
        Args:
            latents: [B, embed_dim, H, W] latent features
            
        Returns:
            images: [B, 3, H*28, W*28] reconstructed images, range [-1, 1]
        """
        # Unscale embeddings
        latents = latents / self.scale_embeding
        
        # DC-AE decoding
        B, C, H, W = latents.shape
        decoded = self.decoder(latents)
        
        # Interpolate to target size
        target_size = (H * self.stride, W * self.stride)
        decoded = F.interpolate(
            decoded, 
            size=target_size, 
            mode='bilinear', 
            align_corners=False
        )
        
        return decoded
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Complete encoding-decoding pipeline.
        
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
        """Number of channels in latent features."""
        return self.embed_dim
    
    @property
    def stride(self) -> int:
        """Downsampling stride factor."""
        return 28  # patch_size=14, after pixel_shuffle becomes 28x
    
    def get_latent_size(self, image_size: int) -> int:
        """
        Calculate latent size corresponding to given image size.
        
        Args:
            image_size: Input image size (assumes square image)
            
        Returns:
            Latent feature size
        """
        return image_size // self.stride

if __name__ == "__main__":
    import os
    from PIL import Image
    import torchvision.transforms as transforms
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 初始化模型
    print("Initializing VGTAE_InternVL3 model...")
    model = VGTAE_InternVL3(
        mllm_path = "OpenGVLab/InternVL3-1B",
        dc_ae_path = "mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers",
        checkpoint_path = "./pytorch_model.bin", # from https://huggingface.co/hustvl/vgt_internvl3_1_6B_pretrain 
        encoder_norm=True,
        device=device,
        dtype=torch.bfloat16
    )
    model.to(device)
    model.eval()
    
    # 定义图像预处理：中心裁剪到448x448，归一化到[-1, 1]
    transform = transforms.Compose([
        transforms.CenterCrop(448),
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化到[-1, 1]
    ])
    
    # 读取测试图像
    test_image_path = "./VGT/test.jpeg"
    if not os.path.exists(test_image_path):
        print(f"Error: Test image not found at {test_image_path}")
        exit(1)
    
    print(f"Loading test image from: {test_image_path}")
    image = Image.open(test_image_path).convert("RGB")
    
    # 预处理图像
    input_tensor = transform(image).unsqueeze(0).to(device, dtype=torch.bfloat16)  # [1, 3, 448, 448]
    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Input tensor range: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")
    
    # 模型重建
    print("Running model reconstruction...")
    with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
        # 编码
        latents = model.encode(input_tensor)
        print(f"Latent shape: {latents.shape}")
        
        # 解码
        reconstructed = model.decode(latents)
        print(f"Reconstructed shape: {reconstructed.shape}")
        print(f"Reconstructed range before clamp: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")
        
        # Clamp到[-1, 1]
        reconstructed = torch.clamp(reconstructed, -1.0, 1.0)
        print(f"Reconstructed range after clamp: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")
    
    # 转换为PIL图像并保存
    def tensor_to_pil(tensor):
        """将tensor转换为PIL图像"""
        # 从[-1, 1]转换到[0, 1]
        tensor = (tensor + 1.0) / 2.0
        tensor = torch.clamp(tensor, 0.0, 1.0)
        
        # 转换为numpy
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)  # 移除batch维度
        tensor = tensor.float().cpu().permute(1, 2, 0).numpy()
        
        # 转换为PIL图像
        tensor = (tensor * 255).astype('uint8')
        return Image.fromarray(tensor)
    
    # 保存原始图像和重建图像
    original_pil = tensor_to_pil(input_tensor.float())
    reconstructed_pil = tensor_to_pil(reconstructed)
    
    # 保存到当前目录
    original_save_path = "original_448x448.png"
    reconstructed_save_path = "reconstructed_448x448.png"
    
    original_pil.save(original_save_path)
    reconstructed_pil.save(reconstructed_save_path)
    
    print(f"Original image saved to: {original_save_path}")
    print(f"Reconstructed image saved to: {reconstructed_save_path}")
    print("Test completed successfully!")
