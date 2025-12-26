"""
DINOv3特征提取器 - 基于timm官方实现
支持ConvNeXt和ViT架构，正确处理Register Tokens
"""

import torch
import torch.nn as nn
import timm
from typing import Optional, Dict
import os
from einops import rearrange

class DINOv3FeatureExtractor(nn.Module):
    """
    DINOv3特征提取器 - 基于timm实现
    
    支持模型:
    - vit_base_patch16_dinov3.lvd1689m (ViT-B/16 with 4 register tokens, 256x256)
    - convnext_large.dinov3_lvd1689m (ConvNeXt-Large, 224x224)
    
    特征输出:
    - ViT: 返回 patch_features (B, num_patches, dim) 和 cls_token (B, dim)
    - ConvNeXt: 返回 patch_features (B, num_patches, dim) 和 pooled_features (B, dim)
    """
    def __init__(self, 
                 model_name: str = "vit_base_patch16_dinov3.lvd1689m",
                 checkpoint_path: Optional[str] = None,
                 freeze: bool = True):
        super().__init__()
        
        self.model_name = model_name
        self.freeze = freeze
        
        # 判断模型架构类型
        self.is_vit = "vit" in model_name.lower()
        self.is_convnext = "convnext" in model_name.lower()
        
        # ViT模型的Register Token数量 (DINOv3使用4个)
        self.num_register_tokens = 4 if self.is_vit else 0
        
        # 加载模型
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            print(f"从本地路径加载模型: {checkpoint_path}")
            self.model = timm.create_model(
                model_name,
                pretrained=False,
                num_classes=0,  # 移除分类头
            )
            # 加载权重
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            elif 'model' in checkpoint:
                checkpoint = checkpoint['model']
            self.model.load_state_dict(checkpoint, strict=False)
        else:
            print(f"从timm加载预训练模型: {model_name}")
            self.model = timm.create_model(
                model_name,
                pretrained=True,
                num_classes=0,
            )
        
        # 获取模型数据配置
        self.data_config = timm.data.resolve_model_data_config(self.model)
        self.transforms = timm.data.create_transform(**self.data_config, is_training=False)
        
        # 获取特征维度和图像尺寸
        self.feature_dim = self.model.num_features
        self.img_size = self.data_config['input_size'][-1]  # (C, H, W) -> H
        
        # 获取patch信息
        if self.is_vit:
            self.patch_size = self.model.patch_embed.patch_size[0]
            self.num_patches = (self.img_size // self.patch_size) ** 2
        elif self.is_convnext:
            # ConvNeXt: 通过forward_features获取特征图尺寸
            # 临时测试以确定输出尺寸
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, self.img_size, self.img_size)
                dummy_features = self.model.forward_features(dummy_input)
                if len(dummy_features.shape) == 4:  # (B, C, H, W)
                    self.feature_map_size = dummy_features.shape[-1]
                    self.num_patches = self.feature_map_size ** 2
                    self.patch_size = self.img_size // self.feature_map_size
        
        # 冻结模型参数
        if freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        
        print(f"dinov3 dim: {self.feature_dim}")
        # if self.is_vit:
        #     print(f"  Register Tokens: {self.num_register_tokens}")
    
    def preprocess_single_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        预处理单张图像
        Args:
            image: (3, H, W) 或 (H, W, 3) PIL格式的tensor，范围[0, 1]
        Returns:
            processed: (3, img_size, img_size) 预处理后的图像
        """
        # 如果是(H, W, 3)格式，转换为(3, H, W)
        if image.shape[0] != 3:
            image = image.permute(2, 0, 1)
        
        # 转换为PIL Image进行预处理
        # timm的transform期望PIL Image或(C, H, W) tensor
        return self.transforms(image)
    
    def preprocess_batch(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        批量预处理图像
        Args:
            pixel_values: (B, 3, H, W) 像素值，范围[-1, 1]或[0, 1]
        Returns:
            processed: (B, 3, img_size, img_size) 预处理后的图像
        """
        # 确保输入在[0, 1]范围内
        if pixel_values.min() < 0:
            # 从[-1, 1]转换到[0, 1]
            pixel_values = (pixel_values + 1.0) / 2.0
        
        # 对批次中每张图像应用transform
        return self.transforms(pixel_values)
    
    def extract_features_vit(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        提取ViT模型的特征
        Args:
            x: (B, 3, H, W) 预处理后的图像
        Returns:
            features_dict: 包含 patch_features, cls_token, register_tokens 的字典
        """
        # 使用forward_features获取所有token
        # 输出形状: (B, num_tokens, feature_dim)
        # num_tokens = 1(CLS) + 4(register) + num_patches
        features = self.model.forward_features(x)  # (B, 1+4+num_patches, feature_dim)
        
        # 分离特殊tokens和patch features
        # 顺序: [CLS, REG1, REG2, REG3, REG4, patch1, patch2, ...]
        num_special_tokens = 1 + self.num_register_tokens  # 1 CLS + 4 Register
        
        cls_token = features[:, 0]  # (B, feature_dim)
        patch_features = features[:, num_special_tokens:]  # (B, num_patches, feature_dim)
        features = rearrange(
            patch_features, 
            'b (h w) c -> b c h w', 
            h=self.img_size // self.patch_size, w=self.img_size // self.patch_size
        )
        result = {
            'features': features, #[B, C, 16, 16]
            'cls_token': cls_token, #[B, C]
            "register_tokens": features[:, 1:num_special_tokens],
        }
        
        return result
    
    def extract_features_convnext(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        提取ConvNeXt模型的特征
        Args:
            x: (B, 3, H, W) 预处理后的图像
        Returns:
            features_dict: 包含 patch_features, cls_token 的字典
        """
        # 使用forward_features获取特征图
        # 输出形状: (B, feature_dim, H, W)
        features = self.model.forward_features(x)  # (B, C, H, W)
        
        B, C, H, W = features.shape
        
        # 使用forward_head获取池化特征（作为CLS token的替代）
        # pre_logits=True 返回池化后未经过分类头的特征
        pooled_features = self.model.forward_head(features, pre_logits=True)  # (B, feature_dim)
        
        return {
            'features': features, #(B, C, 7, 7)
            'cls_token': pooled_features ##(B, C)
        }
    
    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        Args:
            pixel_values: (B, 3, H, W) 输入图像，范围[-1, 1]或[0, 1]
        Returns:
            output_dict: 包含以下键的字典
                - 'patch_features': (B, num_patches, feature_dim) patch特征
                - 'cls_token': (B, feature_dim) CLS token (ViT) 或 pooled特征 (ConvNeXt)
                - 'register_tokens': (B, 4, feature_dim) Register tokens (仅ViT且return_all_tokens=True)
        """
        # 预处理
        processed_images = self.preprocess_batch(pixel_values)
        
        # 提取特征
        with torch.no_grad() if self.freeze else torch.enable_grad():
            if self.is_vit:
                return self.extract_features_vit(processed_images)
            else:  # ConvNeXt
                return self.extract_features_convnext(processed_images)
    
    def get_feature_info(self) -> Dict[str, any]:
        """获取特征信息"""
        info = {
            "model_name": self.model_name,
            "architecture": "ViT" if self.is_vit else "ConvNeXt",
            "feature_dim": self.feature_dim,
            "num_patches": self.num_patches,
            "patch_size": self.patch_size,
            "img_size": self.img_size,
        }
        
        if self.is_vit:
            info["num_register_tokens"] = self.num_register_tokens
            info["num_special_tokens"] = 1 + self.num_register_tokens
        
        return info


def create_dinov3_extractor(
    model_name: str = "vit_base_patch16_dinov3.lvd1689m",
    checkpoint_path: Optional[str] = None,
    device: str = "cuda"
) -> DINOv3FeatureExtractor:
    """
    创建DINOv3特征提取器的便捷函数
    
    Args:
        model_name: timm模型名称
            ViT模型: 
                - 'vit_small_patch16_dinov3.lvd1689m' (256x256)
                - 'vit_base_patch16_dinov3.lvd1689m' (256x256)
                - 'vit_large_patch16_dinov3.lvd1689m' (256x256)
            ConvNeXt模型: 
                - 'convnext_tiny.dinov3_lvd1689m' (224x224)
                - 'convnext_small.dinov3_lvd1689m' (224x224)
                - 'convnext_base.dinov3_lvd1689m' (224x224)
                - 'convnext_large.dinov3_lvd1689m' (224x224)
        checkpoint_path: 本地模型权重路径（可选）
        device: 设备
        return_all_tokens: 是否返回所有tokens（包括register tokens）
    
    Returns:
        extractor: DINOv3特征提取器
    """
    extractor = DINOv3FeatureExtractor(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        freeze=True
    )
    
    return extractor.to(device)

from torch import nn
import torch
import torch.nn.functional as F
from math import *
from transformers import SiglipModel
from transformers import AutoModel

def convert_image_to_patches(image: "torch.Tensor", patch_size: int) -> "torch.Tensor":
    batch_size, num_channels, image_height, image_width = image.shape
    num_patches_height = image_height // patch_size
    num_patches_width = image_width // patch_size
    patched_image = image.reshape(batch_size, num_channels, num_patches_height, patch_size, num_patches_width, patch_size)
    patched_image = patched_image.permute(0, 2, 4, 3, 5, 1)
    patched_image = patched_image.reshape(batch_size, num_patches_height * num_patches_width, -1)
    return patched_image
    
class SigLIP2wNorm(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name

        # -------------------------------------------------------
        # Load SigLIP-2 model (google/siglip2-so400m-patch16-naflex)
        # -------------------------------------------------------
        self.model = AutoModel.from_pretrained(model_name).vision_model
        # -------------------------------------------------------
        # Remove affine of final LayerNorm
        self.model.post_layernorm.elementwise_affine = False
        self.model.post_layernorm.weight = None
        self.model.post_layernorm.bias = None

        # Model configs
        cfg = self.model.config
        self.hidden_size = cfg.hidden_size
        self.patch_size = cfg.patch_size

        # 输出维度
        self.feature_dim = self.hidden_size

        # Freeze encoder
        for p in self.model.parameters():
            p.requires_grad = False

    def preprocess(self, images):
        B, C, H, W = images.shape
        assert H % self.patch_size == 0
        assert W % self.patch_size == 0
        images = convert_image_to_patches(images, self.patch_size)
        grid_h = H // self.patch_size
        grid_w = W // self.patch_size
        attention_mask = torch.ones((B, grid_h*grid_w), device=images.device)
        spatial_shapes = torch.tensor([grid_h, grid_w], device=images.device)
        spatial_shapes = spatial_shapes.reshape(1, -1).repeat(B, 1)
        return images, attention_mask, spatial_shapes
    
    @torch.no_grad()
    def forward(self, x):
        """
        x: (B, C, H, W) in range [-1,1] or any range
        return [B, L, C]
        """
        if x.shape[-1] != 256:
            x = F.interpolate(x,size=(256, 256),mode="bilinear",align_corners=False)
        # import pdb;pdb.set_trace()
        images, attention_mask, spatial_shapes = self.preprocess(x)
        h, w = spatial_shapes[0]
        outputs = self.model(images, attention_mask, spatial_shapes)
        image_features = outputs.last_hidden_state
        
        # feats = image_features[:, 1:, :]
        return image_features