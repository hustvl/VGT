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
        
        print("DINOv3 feature extractor initialized:")
        print(f"  Model: {model_name}")
        print(f"  Architecture: {'ViT' if self.is_vit else 'ConvNeXt'}")
        print(f"  Feature dim: {self.feature_dim}")
        print(f"  Image size: {self.img_size}x{self.img_size}")
        print(f"  Patch size: {self.patch_size}")
        print(f"  Patch count: {self.num_patches}")
        if self.is_vit:
            print(f"  Register Tokens: {self.num_register_tokens}")
        print(f"  Frozen parameters: {freeze}")
    
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


def test_dinov3_extractor():
    """Smoke test for the DINOv3 feature extractor."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 80)
    print("Testing ViT model with register tokens")
    print("=" * 80)
    
    # Test the ViT model.
    vit_extractor = create_dinov3_extractor(
        model_name="vit_base_patch16_dinov3.lvd1689m",
        device=device,
    )
    
    # Test input with an arbitrary size; it will be resized automatically.
    batch_size = 2
    test_images = torch.randn(batch_size, 3, 384, 384).to(device)
    
    # Extract features.
    with torch.no_grad():
        vit_features = vit_extractor(test_images)
    
    print(f"\nInput shape: {test_images.shape}")
    print(f"Patch feature shape: {vit_features['patch_features'].shape}")
    print(f"CLS token shape: {vit_features['cls_token'].shape}")
    if 'register_tokens' in vit_features and vit_features['register_tokens'] is not None:
        print(f"Register token shape: {vit_features['register_tokens'].shape}")
    print("\nFeature info:")
    for key, value in vit_extractor.get_feature_info().items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 80)
    # print("Testing ConvNeXt model")
    # print("=" * 80)
    
    # # Test the ConvNeXt model.
    # convnext_extractor = create_dinov3_extractor(
    #     model_name="convnext_large.dinov3_lvd1689m",
    #     device=device
    # )
    
    # # Extract features.
    # with torch.no_grad():
    #     convnext_features = convnext_extractor(test_images)
    
    # print(f"\nInput shape: {test_images.shape}")
    # print(f"Patch feature shape: {convnext_features['patch_features'].shape}")
    # print(f"Pooled feature shape (used as CLS token): {convnext_features['cls_token'].shape}")
    # print("\nFeature info:")
    # for key, value in convnext_extractor.get_feature_info().items():
    #     print(f"  {key}: {value}")
    
    print("\n" + "=" * 80)
    print("Testing different input sizes")
    print("=" * 80)
    
    # Test several input sizes.
    test_sizes = [(512, 512), (128, 128), (256, 384)]
    for h, w in test_sizes:
        test_img = torch.randn(1, 3, h, w).to(device)
        with torch.no_grad():
            features = vit_extractor(test_img)
        print(f"Input {h}x{w} -> patch feature shape: {features['patch_features'].shape}")
    
    return vit_features, convnext_features


if __name__ == "__main__":
    test_dinov3_extractor()
