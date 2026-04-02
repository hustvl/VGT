"""
DINOv3 feature extractor based on the official timm implementation.
Supports both ConvNeXt and ViT backbones, including register token handling.
"""

import torch
import torch.nn as nn
import timm
from typing import Optional, Dict
import os
from einops import rearrange

class DINOv3FeatureExtractor(nn.Module):
    """
    DINOv3 feature extractor implemented with timm.

    Supported models:
    - vit_base_patch16_dinov3.lvd1689m (ViT-B/16 with 4 register tokens, 256x256)
    - convnext_large.dinov3_lvd1689m (ConvNeXt-Large, 224x224)

    Feature outputs:
    - ViT: returns patch_features (B, num_patches, dim) and cls_token (B, dim)
    - ConvNeXt: returns patch_features (B, num_patches, dim) and pooled_features (B, dim)
    """
    def __init__(self, 
                 model_name: str = "vit_base_patch16_dinov3.lvd1689m",
                 checkpoint_path: Optional[str] = None,
                 freeze: bool = True):
        super().__init__()
        
        self.model_name = model_name
        self.freeze = freeze
        
        # Detect the backbone family.
        self.is_vit = "vit" in model_name.lower()
        self.is_convnext = "convnext" in model_name.lower()
        
        # Number of register tokens for ViT backbones (DINOv3 uses 4).
        self.num_register_tokens = 4 if self.is_vit else 0
        
        # Load the backbone.
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            print(f"Loading model from local checkpoint: {checkpoint_path}")
            self.model = timm.create_model(
                model_name,
                pretrained=False,
                num_classes=0,  # Remove the classification head.
            )
            # Load checkpoint weights.
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            elif 'model' in checkpoint:
                checkpoint = checkpoint['model']
            self.model.load_state_dict(checkpoint, strict=False)
        else:
            print(f"Loading pretrained timm model: {model_name}")
            self.model = timm.create_model(
                model_name,
                pretrained=True,
                num_classes=0,
            )
        
        # Resolve data config and preprocessing transforms.
        self.data_config = timm.data.resolve_model_data_config(self.model)
        self.transforms = timm.data.create_transform(**self.data_config, is_training=False)
        
        # Cache feature width and input image size.
        self.feature_dim = self.model.num_features
        self.img_size = self.data_config['input_size'][-1]  # (C, H, W) -> H
        
        # Infer patch layout metadata.
        if self.is_vit:
            self.patch_size = self.model.patch_embed.patch_size[0]
            self.num_patches = (self.img_size // self.patch_size) ** 2
        elif self.is_convnext:
            # For ConvNeXt, probe forward_features to determine the output map size.
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, self.img_size, self.img_size)
                dummy_features = self.model.forward_features(dummy_input)
                if len(dummy_features.shape) == 4:  # (B, C, H, W)
                    self.feature_map_size = dummy_features.shape[-1]
                    self.num_patches = self.feature_map_size ** 2
                    self.patch_size = self.img_size // self.feature_map_size
        
        # Optionally freeze backbone parameters.
        if freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        
        print(f"dinov3 dim: {self.feature_dim}")
        # if self.is_vit:
        #     print(f"  Register Tokens: {self.num_register_tokens}")
    
    def preprocess_single_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Preprocess a single image tensor.

        Args:
            image: Tensor in ``(3, H, W)`` or ``(H, W, 3)`` format, with values in ``[0, 1]``.

        Returns:
            processed: Preprocessed image tensor of shape ``(3, img_size, img_size)``.
        """
        # Convert (H, W, 3) input to (3, H, W) if needed.
        if image.shape[0] != 3:
            image = image.permute(2, 0, 1)
        
        # timm transforms accept PIL images or tensors in (C, H, W) format.
        return self.transforms(image)
    
    def preprocess_batch(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Preprocess a batch of image tensors.

        Args:
            pixel_values: Tensor of shape ``(B, 3, H, W)`` with values in ``[-1, 1]`` or ``[0, 1]``.

        Returns:
            processed: Preprocessed tensor of shape ``(B, 3, img_size, img_size)``.
        """
        # Normalize the input range to [0, 1] when necessary.
        if pixel_values.min() < 0:
            # Map values from [-1, 1] to [0, 1].
            pixel_values = (pixel_values + 1.0) / 2.0
        
        # Apply the transform to the full batch.
        return self.transforms(pixel_values)
    
    def extract_features_vit(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features from a ViT backbone.

        Args:
            x: Preprocessed images with shape ``(B, 3, H, W)``.

        Returns:
            Dictionary containing patch features, cls token, and register tokens.
        """
        # forward_features returns all tokens with shape (B, num_tokens, feature_dim).
        # num_tokens = 1(CLS) + 4(register) + num_patches
        features = self.model.forward_features(x)  # (B, 1+4+num_patches, feature_dim)
        
        # Token layout: [CLS, REG1, REG2, REG3, REG4, patch1, patch2, ...].
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
        Extract features from a ConvNeXt backbone.

        Args:
            x: Preprocessed images with shape ``(B, 3, H, W)``.

        Returns:
            Dictionary containing feature maps and pooled features.
        """
        # forward_features returns a feature map of shape (B, feature_dim, H, W).
        features = self.model.forward_features(x)  # (B, C, H, W)
        
        B, C, H, W = features.shape
        
        # Use the pooled representation as the CLS-token analogue.
        pooled_features = self.model.forward_head(features, pre_logits=True)  # (B, feature_dim)
        
        return {
            'features': features, #(B, C, 7, 7)
            'cls_token': pooled_features ##(B, C)
        }
    
    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Run preprocessing and feature extraction.

        Args:
            pixel_values: Input images of shape ``(B, 3, H, W)`` with values in ``[-1, 1]`` or ``[0, 1]``.

        Returns:
            A dictionary containing the extracted features.
        """
        # Preprocess inputs before feature extraction.
        processed_images = self.preprocess_batch(pixel_values)
        
        # Extract backbone features.
        with torch.no_grad() if self.freeze else torch.enable_grad():
            if self.is_vit:
                return self.extract_features_vit(processed_images)
            else:  # ConvNeXt
                return self.extract_features_convnext(processed_images)
    
    def get_feature_info(self) -> Dict[str, any]:
        """Return metadata about the configured feature extractor."""
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
    Convenience factory for the DINOv3 feature extractor.

    Args:
        model_name: timm model name.
            ViT models:
                - 'vit_small_patch16_dinov3.lvd1689m' (256x256)
                - 'vit_base_patch16_dinov3.lvd1689m' (256x256)
                - 'vit_large_patch16_dinov3.lvd1689m' (256x256)
            ConvNeXt models:
                - 'convnext_tiny.dinov3_lvd1689m' (224x224)
                - 'convnext_small.dinov3_lvd1689m' (224x224)
                - 'convnext_base.dinov3_lvd1689m' (224x224)
                - 'convnext_large.dinov3_lvd1689m' (224x224)
        checkpoint_path: Optional local checkpoint path.
        device: Target device.

    Returns:
        Configured DINOv3 feature extractor.
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

        # Output feature dimension.
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
