import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional
from timm.models.layers import trunc_normal_

try:
    from mmengine.dist import is_main_process
except ImportError:
    def is_main_process() -> bool:
        """Fallback function when mmengine is not available."""
        return True


class Learnable2DPosEncoding(nn.Module):
    """
    Learnable 2D Position Encoding for truncation mode with center cropping.
    Supports learnable encoding and center crop mode only.
    """
    
    def __init__(
        self,
        embed_dim: int,
        init_res: Tuple[int, int] = (64, 64),
        crop_mode: str = "center"
    ):
        """
        Initialize Learnable2DPosEncoding.
        
        Args:
            embed_dim: Embedding dimension
            init_res: Initial resolution (height, width)
            crop_mode: Cropping mode ("center")
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.init_res = init_res
        self.crop_mode = crop_mode  # "center"
        
        assert crop_mode == "center", f"Only center crop mode supported, got: {crop_mode}"
        
        # Basic 2D grid parameters - learnable encoding
        self.grid_embed = nn.Parameter(
            torch.zeros(1, embed_dim, init_res[0], init_res[1])
        )
        trunc_normal_(self.grid_embed, std=0.02)
        if is_main_process():
            print(f"Initializing Learnable2DPosEncoding - Mode: Learnable, Base size: {init_res}, Crop mode: {crop_mode}")
    
    def forward(
        self,
        H: int,
        W: int,
        base_metaquery: Optional[torch.Tensor] = None,
        mode: str = "truncation"
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for position encoding in truncation mode.
        
        Args:
            H: Target feature map height
            W: Target feature map width
            base_metaquery: Base metaquery with shape [1, C, init_H, init_W] (optional)
            mode: Should be "truncation"
            
        Returns:
            Tuple of (position embedding, truncated metaquery)
        """
        assert mode == "truncation", f"Only truncation mode supported, got: {mode}"
        
        # 2D truncation mode: center crop
        pos_embed = self._2d_truncate(H, W, self.grid_embed)
        
        if base_metaquery is not None:
            truncated_metaquery = self._2d_truncate(H, W, base_metaquery)
        else:
            truncated_metaquery = None
        
        return pos_embed, truncated_metaquery
    
    def _2d_truncate(self, H: int, W: int, feature_map: torch.Tensor) -> torch.Tensor:
        """
        Truncate target region from 2D feature map using center crop.
        
        Args:
            H: Target height
            W: Target width
            feature_map: Feature map with shape [1, C, orig_H, orig_W]
            
        Returns:
            Truncated result with shape [1, H*W, C]
        """
        _, C, orig_H, orig_W = feature_map.shape
        
        # Ensure target size does not exceed original size
        if H > orig_H or W > orig_W:
            raise ValueError(f"Target size({H}, {W}) exceeds original size({orig_H}, {orig_W})")
        
        # Center crop
        start_h = (orig_H - H) // 2
        start_w = (orig_W - W) // 2
        
        # Truncate target region
        truncated_2d = feature_map[:, :, start_h:start_h+H, start_w:start_w+W]  # [1, C, H, W]
        
        # Reshape to sequence format
        result = truncated_2d.permute(0, 2, 3, 1).reshape(1, H * W, C)  # [1, H*W, C]
        return result


class AdaptiveMetaQuery(nn.Module):
    """
    Simplified AdaptiveMetaQuery supporting only truncation mode with center cropping.
    """
    
    def __init__(
        self,
        mllm_embed_dim: int = 768,
        mode: str = "truncation",  # Only truncation mode
        max_size: Tuple[int, int] = (64, 64),   # Max size for truncation
        crop_mode: str = "center",  # Only center crop
        encoding_type: str = "learnable"  # Only learnable encoding
    ):
        """
        Initialize AdaptiveMetaQuery.
        
        Args:
            mllm_embed_dim: MLLM embedding dimension
            mode: Generation mode ("truncation")
            max_size: Max size for truncation mode
            crop_mode: Cropping mode ("center")
            encoding_type: Encoding type ("learnable")
        """
        super().__init__()
        
        assert mode == "truncation", f"Only truncation mode supported, got: {mode}"
        assert crop_mode == "center", f"Only center crop mode supported, got: {crop_mode}"
        assert encoding_type == "learnable", f"Only learnable encoding supported, got: {encoding_type}"
        
        self.mllm_embed_dim = mllm_embed_dim
        self.mode = mode
        self.max_size = max_size
        self.crop_mode = crop_mode
        self.encoding_type = encoding_type
        
        if is_main_process():
            print(f"Initializing AdaptiveMetaQuery - Mode: {mode}, Crop mode: {crop_mode}, Encoding type: {encoding_type}")
        
        self._init_truncation_mode()

    def _init_truncation_mode(self) -> None:
        """Initialize truncation mode."""
        max_h, max_w = self.max_size
        max_num_queries = max_h * max_w
        
        if is_main_process():
            print(f"Truncation mode - Max size: {max_h}x{max_w} = {max_num_queries} tokens, Crop mode: {self.crop_mode}")
        
        # Max size metaquery parameters (2D grid format)
        self.max_metaquery = nn.Parameter(
            torch.zeros(1, self.mllm_embed_dim, max_h, max_w)
        )
        nn.init.normal_(self.max_metaquery, std=1 / math.sqrt(self.mllm_embed_dim))

        # Position encoding
        self.pos_encoding = Learnable2DPosEncoding(
            self.mllm_embed_dim, init_res=self.max_size, 
            crop_mode=self.crop_mode
        )
    
    def _truncation_forward(self, target_h: int, target_w: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for 2D truncation mode."""
        max_h, max_w = self.max_size
        
        if target_h > max_h or target_w > max_w:
            raise ValueError(f"Target size({target_h}, {target_w}) exceeds max size limit({max_h}, {max_w})")

        # Use 2D truncation to get metaquery and position encoding
        pos_embed, metaquery = self.pos_encoding(
            target_h, target_w, self.max_metaquery, mode="truncation"
        )
        pos_embed = pos_embed.to(device=metaquery.device, dtype=metaquery.dtype)

        return metaquery, pos_embed
    
    def forward(
        self, 
        target_h: int, 
        target_w: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate MetaQuery for specified resolution.
        
        Args:
            target_h: Target VAE height
            target_w: Target VAE width
            
        Returns:
            Tuple of (metaquery, pos_embed)
        """
        return self._truncation_forward(target_h, target_w)