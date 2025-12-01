import os
import torch
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from einops import rearrange
import numpy as np
import torch
import argparse
from xtuner.registry import BUILDER
from xtuner.model.utils import guess_load_checkpoint
from mmengine.config import Config
from einops import rearrange
from PIL import Image
from mmengine.runner.checkpoint import load_checkpoint
from mmengine.logging import MMLogger
import matplotlib.pyplot as plt
import sys
import os
from typing import Any



import torch
from typing import List, Tuple
import torch
from collections import OrderedDict
from mmengine.logging import MMLogger
from mmengine.runner import load_checkpoint


def pad_input_ids(
    input_ids: List[torch.Tensor],
    batch_first: bool = True,
    padding_value: int = 0,
    padding_mode: str = "left"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad a batch of input_ids, returns (padded_input_ids, attention_mask)

    Args:
        input_ids (List[Tensor]): Token id tensors for each sample, shape [seq_len]
        batch_first (bool): Whether batch dimension is first, default True => [B, T]
        padding_value (int): Padding value (e.g., pad_token_id)
        padding_mode (str): 'left' or 'right', determines padding direction

    Returns:
        Tuple[Tensor, Tensor]:
            - padded_input_ids: [B, T] or [T, B]
            - attention_mask: Same shape, padding positions are False, others are True
    """

    # Validate inputs
    if len(input_ids) == 0:
        raise ValueError("input_ids cannot be empty")

    if padding_mode not in ("left", "right"):
        raise ValueError(f"padding_mode must be 'left' or 'right', but got {padding_mode}")

    max_len = max(x.size(0) for x in input_ids)

    padded_inputs = []
    attention_masks = []

    for ids in input_ids:
        seq_len = ids.size(0)
        pad_len = max_len - seq_len

        pad_tensor = torch.full((pad_len,), padding_value, dtype=ids.dtype, device=ids.device)
        mask_pad = torch.zeros(pad_len, dtype=torch.bool, device=ids.device)
        mask_content = torch.ones(seq_len, dtype=torch.bool, device=ids.device)

        if padding_mode == "left":
            padded = torch.cat([pad_tensor, ids], dim=0)
            mask = torch.cat([mask_pad, mask_content], dim=0)
        else:  # right padding
            padded = torch.cat([ids, pad_tensor], dim=0)
            mask = torch.cat([mask_content, mask_pad], dim=0)

        padded_inputs.append(padded)
        attention_masks.append(mask)

    padded_input_ids = torch.stack(padded_inputs, dim=0 if batch_first else 1)
    attention_mask = torch.stack(attention_masks, dim=0 if batch_first else 1)

    return padded_input_ids, attention_mask

def load_checkpoint_with_ema(model, checkpoint_path, use_ema=True, map_location='cpu', strict=False, logger=None):
    """
    Load checkpoint and prioritize EMA weights if available.

    Args:
        model (nn.Module): Target model
        checkpoint_path (str): Checkpoint path
        use_ema (bool): Whether to prioritize loading EMA weights
        map_location (str): Map location
        strict (bool): Whether to strictly match keys
        logger (MMLogger): Logger instance

    Returns:
        dict: Checkpoint dictionary returned after loading
    """
    if logger is None:
        logger = MMLogger.get_instance(name='load_checkpoint', log_level='INFO')
    
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    
    # Prioritize EMA weights
    if use_ema and 'ema_state_dict' in checkpoint:
        state_dict = checkpoint['ema_state_dict']
        logger.info(f"Loading EMA weights from {checkpoint_path}")
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        logger.info(f"Loading regular weights from {checkpoint_path}")
    else:
        state_dict = checkpoint
        logger.info(f"Loading checkpoint as raw state_dict from {checkpoint_path}")
    
    # Remove module prefix
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k[len('module.'):] if k.startswith('module.') else k
        new_state_dict[new_key] = v
    
    # Load weights
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        logger.warning(f"Missing keys: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected}")
    
    return checkpoint
    
