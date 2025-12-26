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
        
        print(f"DINOv3特征提取器初始化完成:")
        print(f"  模型: {model_name}")
        print(f"  架构: {'ViT' if self.is_vit else 'ConvNeXt'}")
        print(f"  特征维度: {self.feature_dim}")
        print(f"  图像尺寸: {self.img_size}x{self.img_size}")
        print(f"  Patch大小: {self.patch_size}")
        print(f"  Patch数量: {self.num_patches}")
        if self.is_vit:
            print(f"  Register Tokens: {self.num_register_tokens}")
        print(f"  冻结参数: {freeze}")
    
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


def test_dinov3_extractor():
    """测试DINOv3特征提取器"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 80)
    print("测试 ViT 模型 (带Register Tokens)")
    print("=" * 80)
    
    # 测试ViT模型
    vit_extractor = create_dinov3_extractor(
        model_name="vit_base_patch16_dinov3.lvd1689m",
        device=device,
    )
    
    # 测试输入 - 任意尺寸，会自动resize到模型需要的尺寸
    batch_size = 2
    test_images = torch.randn(batch_size, 3, 384, 384).to(device)
    
    # 提取特征
    with torch.no_grad():
        vit_features = vit_extractor(test_images)
    
    print(f"\n输入形状: {test_images.shape}")
    print(f"Patch特征形状: {vit_features['patch_features'].shape}")
    print(f"CLS token形状: {vit_features['cls_token'].shape}")
    if 'register_tokens' in vit_features and vit_features['register_tokens'] is not None:
        print(f"Register tokens形状: {vit_features['register_tokens'].shape}")
    print(f"\n特征信息:")
    for key, value in vit_extractor.get_feature_info().items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 80)
    # print("测试 ConvNeXt 模型")
    # print("=" * 80)
    
    # # 测试ConvNeXt模型
    # convnext_extractor = create_dinov3_extractor(
    #     model_name="convnext_large.dinov3_lvd1689m",
    #     device=device
    # )
    
    # # 提取特征
    # with torch.no_grad():
    #     convnext_features = convnext_extractor(test_images)
    
    # print(f"\n输入形状: {test_images.shape}")
    # print(f"Patch特征形状: {convnext_features['patch_features'].shape}")
    # print(f"Pooled特征形状 (作为CLS token): {convnext_features['cls_token'].shape}")
    # print(f"\n特征信息:")
    # for key, value in convnext_extractor.get_feature_info().items():
    #     print(f"  {key}: {value}")
    
    print("\n" + "=" * 80)
    print("测试不同输入尺寸")
    print("=" * 80)
    
    # 测试不同尺寸的输入
    test_sizes = [(512, 512), (128, 128), (256, 384)]
    for h, w in test_sizes:
        test_img = torch.randn(1, 3, h, w).to(device)
        with torch.no_grad():
            features = vit_extractor(test_img)
        print(f"输入 {h}x{w} -> Patch特征: {features['patch_features'].shape}")
    
    return vit_features, convnext_features


if __name__ == "__main__":
    test_dinov3_extractor()