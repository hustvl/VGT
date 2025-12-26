"""This file contains perceptual loss module using LPIPS and ConvNeXt-S.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License. 
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import os


def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def pixel_shuffle(x, scale_factor=0.5):
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


class DistillLoss(torch.nn.Module):
    def __init__(self, model_name: str = "OpenGVLab/InternVL3-1B"):
        """Initializes the Distill class.

        Args:
            model_name: A string, the path of the distillation loss model to use.

        """
        super().__init__()
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            use_flash_attn=False,
            trust_remote_code=True)
        self.ref_vit = model.vision_model
        self.ref_mlp1 = model.mlp1

        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor, out_feat: torch.Tensor):
        """Computes the perceptual loss.

        Args:
            input: A tensor of shape (B, C, H, W), the input image. Normalized to [0, 1].
            target: A tensor of shape (B, C, H, W), the target image. Normalized to [0, 1].

        Returns:
            A scalar tensor, the perceptual loss.
        """
        # x is in [0,1], need imgnet normalize
        # imgnetnorm
        std = torch.tensor([0.229,0.224,0.225]).to(x.device)
        mean = torch.tensor([0.485,0.456,0.406]).to(x.device)
        mean = mean[None, :, None, None]
        std = std[None, :, None, None]
        x = (x - mean) / std
        # Always in eval mode.
        self.eval()
        loss = 0.
        with torch.no_grad():
            vit_embeds = self.ref_vit.embeddings(x)
            for idx, encoder_layer in enumerate(self.ref_vit.encoder.layers):
                vit_embeds = encoder_layer(vit_embeds)
            vit_embeds = vit_embeds[:,1:,:].contiguous().float()
            h = w = int(vit_embeds.shape[1] ** 0.5)
            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
            vit_embeds = pixel_shuffle(vit_embeds)
            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
            vit_embeds = self.ref_mlp1(vit_embeds)
        target_feat = vit_embeds

        distill_loss = F.mse_loss(
            out_feat,
            target_feat,
            reduction="mean",
        )
        loss += distill_loss

        return loss


class SigLIP2_DistillLoss(torch.nn.Module):
    def __init__(self, model_name: str = "OpenGVLab/InternVL3-1B"):
        """Initializes the Distill class.

        Args:
            model_name: A string, the path of the distillation loss model to use.

        """
        super().__init__()
        model = AutoModel.from_pretrained(
            model_name, 
            # attn_implementation="eager",
        )
        self.ref_vit = model.vision_model

        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor, out_feat: torch.Tensor):
        """Computes the perceptual loss.

        Args:
            input: A tensor of shape (B, C, H, W), the input image. Normalized to [0, 1].
            target: A tensor of shape (B, C, H, W), the target image. Normalized to [0, 1].

        Returns:
            A scalar tensor, the perceptual loss.
        """
        """Computes the perceptual loss.

        Args:
            input: A tensor of shape (B, C, H, W), the input image. Normalized to [0, 1].
            target: A tensor of shape (B, C, H, W), the target image. Normalized to [0, 1].

        Returns:
            A scalar tensor, the perceptual loss.
        """
        # x is in [0,1]

        x = (x - 0.5) / 0.5
        self.eval()
        loss = 0.
        with torch.no_grad():
            vit_embeds = self.ref_vit(x, output_hidden_states=True).last_hidden_state
        target_feat = vit_embeds

        distill_loss = F.mse_loss(
            out_feat,
            target_feat,
            reduction="mean",
        )
        loss += distill_loss

        return loss




class Dinov3_DistillLoss(torch.nn.Module):
    def __init__(self, model_name: str = "vit_large_patch16_dinov3.lvd1689m"):
        """Initializes the Distill class.

        Args:
            model_name: A string, the path of the distillation loss model to use.

        """
        super().__init__()
        import timm
        self.ref_vit = timm.create_model(model_name, pretrained=True,num_classes=0,)

        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor, out_feat: torch.Tensor):
        """Computes the perceptual loss.

        Args:
            input: A tensor of shape (B, C, H, W), the input image. Normalized to [0, 1].
            target: A tensor of shape (B, C, H, W), the target image. Normalized to [0, 1].

        Returns:
            A scalar tensor, the perceptual loss.
        """
        """Computes the perceptual loss.

        Args:
            input: A tensor of shape (B, C, H, W), the input image. Normalized to [0, 1].
            target: A tensor of shape (B, C, H, W), the target image. Normalized to [0, 1].

        Returns:
            A scalar tensor, the perceptual loss.
        """
        # x is in [0,1]

        x = (x - 0.5) / 0.5
        # import pdb;pdb.set_trace()
        # print("distill_loss after", x.min(), x.max(), x.sum())
        self.eval()
        loss = 0.
        with torch.no_grad():
            vit_embeds = self.ref_vit.forward_features(x)[:, 5:]
        target_feat = vit_embeds

        distill_loss = F.mse_loss(
            out_feat,
            target_feat,
            reduction="mean",
        )
        loss += distill_loss

        return loss


from transformers import Dinov2WithRegistersModel
class Dinohuf_DistillLoss(torch.nn.Module):
    def __init__(self, model_name: str = "vit_large_patch16_dinov3.lvd1689m"):
        """Initializes the Distill class.

        Args:
            model_name: A string, the path of the distillation loss model to use.

        """
        super().__init__()
        try:
            self.ref_vit = Dinov2WithRegistersModel.from_pretrained(model_name, local_files_only=True)
        except (OSError, ValueError, AttributeError):
            self.ref_vit = Dinov2WithRegistersModel.from_pretrained(model_name, local_files_only=False)
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor, out_feat: torch.Tensor):
        """Computes the perceptual loss.

        Args:
            input: A tensor of shape (B, C, H, W), the input image. Normalized to [0, 1].
            target: A tensor of shape (B, C, H, W), the target image. Normalized to [0, 1].

        Returns:
            A scalar tensor, the perceptual loss.
        """
        """Computes the perceptual loss.

        Args:
            input: A tensor of shape (B, C, H, W), the input image. Normalized to [0, 1].
            target: A tensor of shape (B, C, H, W), the target image. Normalized to [0, 1].

        Returns:
            A scalar tensor, the perceptual loss.
        """
        # x is in [0,1]

        x = (x - 0.5) / 0.5
        # import pdb;pdb.set_trace()
        # print("distill_loss after", x.min(), x.max(), x.sum())
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False) # !!Dino要求输入 224
        self.eval()
        loss = 0.
        with torch.no_grad():
            vit_embeds = self.ref_vit(x, output_hidden_states=True).last_hidden_state[:, 5:]
        target_feat = vit_embeds

        distill_loss = F.mse_loss(
            out_feat,
            target_feat,
            reduction="mean",
        )
        loss += distill_loss

        return loss




from ..internvl3.modeling_internvl_chat import InternVLChatModel
class InternViT_DistillLoss(torch.nn.Module):
    def __init__(self, model_name: str = "vit_large_patch16_dinov3.lvd1689m"):
        """Initializes the Distill class.

        Args:
            model_name: A string, the path of the distillation loss model to use.

        """
        super().__init__()
        self.ref_vit = InternVLChatModel.from_pretrained(
            pretrained_model_name_or_path=model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
        )
        # only train vit
        del self.ref_vit.language_model

        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor, out_feat: torch.Tensor):
        """Computes the perceptual loss.

        Args:
            input: A tensor of shape (B, C, H, W), the input image. Normalized to [0, 1].
            target: A tensor of shape (B, C, H, W), the target image. Normalized to [0, 1].

        Returns:
            A scalar tensor, the perceptual loss.
        """
        """Computes the perceptual loss.

        Args:
            input: A tensor of shape (B, C, H, W), the input image. Normalized to [0, 1].
            target: A tensor of shape (B, C, H, W), the target image. Normalized to [0, 1].

        Returns:
            A scalar tensor, the perceptual loss.
        """
        # x is in [0,1]

        x = (x - 0.5) / 0.5
        # import pdb;pdb.set_trace()
        # print("distill_loss after", x.min(), x.max(), x.sum())
        # x = x.to(dtype=self.ref_vit.dtype)
        # print(f"DistillLoss dtype16: {x.min()} {x.max()} {x.sum()}, {x.dtype} \n")
        # import pdb;pdb.set_trace()
        self.eval()
        loss = 0.
        with torch.no_grad():
            vit_embeds = self.ref_vit.extract_feature(x)
        target_feat = vit_embeds

        distill_loss = F.mse_loss(
            out_feat,
            target_feat,
            reduction="mean",
        )
        loss += distill_loss

        return loss

from einops import rearrange, reduce
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
class VLVAE_QWEN2_5_DistillLoss(torch.nn.Module):
    def __init__(self, model_name: str = "OpenGVLab/InternVL3-1B"):
        """Initializes the Distill class.

        Args:
            model_name: A string, the path of the distillation loss model to use.

        """
        super().__init__()
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, 
            # attn_implementation="eager",
        )
        self.ref_vit = model.visual
        self.vision_config = model.config.vision_config

        IMAGENET_MEAN = (0.48145466, 0.4578275,  0.40821073)
        IMAGENET_STD = (0.26862954, 0.26130258, 0.27577711)
        self.register_buffer('vit_mean', torch.tensor(IMAGENET_MEAN), persistent=False)
        self.register_buffer('vit_std', torch.tensor(IMAGENET_STD), persistent=False)

        for param in self.parameters():
            param.requires_grad = False
    
    def get_semantic_features(self,encoder, pixel_values):
        # [0, 1]
        pixel_values = pixel_values - self.vit_mean.view(1, 3, 1, 1)
        pixel_values = pixel_values / self.vit_std.view(1, 3, 1, 1)

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

        return image_embeds

    def forward(self, x: torch.Tensor, out_feat: torch.Tensor):
        """Computes the perceptual loss.

        Args:
            input: A tensor of shape (B, C, H, W), the input image. Normalized to [0, 1].
            target: A tensor of shape (B, C, H, W), the target image. Normalized to [0, 1].

        Returns:
            A scalar tensor, the perceptual loss.
        """
        """Computes the perceptual loss.

        Args:
            input: A tensor of shape (B, C, H, W), the input image. Normalized to [0, 1].
            target: A tensor of shape (B, C, H, W), the target image. Normalized to [0, 1].

        Returns:
            A scalar tensor, the perceptual loss.
        """
        # x is in [0,1]

        # x = (x - 0.5) / 0.5
        self.eval()
        loss = 0.
        with torch.no_grad():
            vit_embeds = self.get_semantic_features(self.ref_vit, x)
        target_feat = vit_embeds

        distill_loss = F.mse_loss(out_feat,target_feat,reduction="mean",)
        loss += distill_loss
        import pdb;pdb.set_trace()

        return loss