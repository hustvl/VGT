# from https://github.com/wusize/OpenUni
import torch
import torch.nn.functional as F
import random
import numpy as np
from xtuner.utils import DEFAULT_PAD_TOKEN_INDEX, IGNORE_INDEX
from typing import Dict, Sequence, List, Tuple, Union
from torch.nn.utils.rnn import pad_sequence
from functools import partial
from dataclasses import dataclass
from torchvision import transforms
from mmengine.logging import print_log
from .utils import pad_input_ids


def smart_resize_and_crop(image_tensor: torch.Tensor, 
                         target_size: Tuple[int, int],
                         mode: str = 'center_crop') -> torch.Tensor:

    '''
    Intelligent cropping and scaling, maintaining aspect ratio
    
    Args:
        image_tensor: Image tensor in [C, H, W] format
        target_size: (width, height) target size
        mode: Cropping mode ('center_crop', 'random_crop', 'pad_resize')
    
    Returns:
        Processed image tensor [C, target_height, target_width]
    '''

    if len(image_tensor.shape) != 3:
        raise ValueError(f"Expected 3D tensor [C,H,W], got shape {image_tensor.shape}")
    
    C, H, W = image_tensor.shape
    target_w, target_h = target_size
    
    scale_w = target_w / W
    scale_h = target_h / H
    
    if mode == 'pad_resize':
        # Padding after proportional scaling (keep the complete image content)
        scale = min(scale_w, scale_h)
        new_w, new_h = int(W * scale), int(H * scale)
        
        image_tensor = image_tensor.unsqueeze(0)  # [1, C, H, W]
        image_tensor = F.interpolate(image_tensor, size=(new_h, new_w), 
                                   mode='bilinear', align_corners=False)
        image_tensor = image_tensor.squeeze(0)  # [C, new_h, new_w]

        pad_h = target_h - new_h
        pad_w = target_w - new_w
        padding = (pad_w // 2, pad_w - pad_w // 2,  # left, right
                  pad_h // 2, pad_h - pad_h // 2)  # top, bottom
        image_tensor = F.pad(image_tensor, padding, mode='constant', value=0)
        
    elif mode in ['center_crop', 'random_crop']:
        scale = max(scale_w, scale_h)
        new_w, new_h = max(target_w, int(W * scale)), max(target_h, int(H * scale))
        
        image_tensor = image_tensor.unsqueeze(0)  # [1, C, H, W]
        image_tensor = F.interpolate(image_tensor, size=(new_h, new_w), 
                                   mode='bilinear', align_corners=False)
        image_tensor = image_tensor.squeeze(0)  # [C, new_h, new_w]

        if mode == 'center_crop':
            start_h = (new_h - target_h) // 2
            start_w = (new_w - target_w) // 2
        elif mode == 'random_crop':
            start_h = random.randint(0, max(0, new_h - target_h))
            start_w = random.randint(0, max(0, new_w - target_w))
        
        image_tensor = image_tensor[:, start_h:start_h+target_h, start_w:start_w+target_w]
    else:
        raise ValueError(f"Unknown crop mode: {mode}")
    
    return image_tensor


def collate_func_img2img(instances: Sequence[Dict],
                         pad_index: int = DEFAULT_PAD_TOKEN_INDEX):
    pixel_values_src_list, pixel_values_list, input_ids_list = [], [], []
    for instance in instances:
        pixel_values_src_ = instance.pop('pixel_values_src')
        if isinstance(pixel_values_src_, torch.Tensor):
            pixel_values_src_ = [pixel_values_src_]
        pixel_values_src_list += pixel_values_src_
        pixel_values_list.append(instance.pop('pixel_values'))
        input_ids_list.append(instance.pop('input_ids'))

    ori_length = [len(ids) for ids in input_ids_list]
    pad_length = max(ori_length)
    attention_mask = torch.zeros(len(instances), pad_length, dtype=torch.bool)
    input_ids = torch.full(size=(len(instances), pad_length),
                           fill_value=pad_index, dtype=torch.long)

    # left padding for editing
    for i, length in enumerate(ori_length):
        attention_mask[i, -length:] = True
        input_ids_i = input_ids_list[i]
        if not isinstance(input_ids_i, torch.Tensor):
            input_ids_i = torch.tensor(input_ids_i, dtype=torch.long)
        input_ids[i, -length:] = input_ids_i

    pixel_values = torch.stack(pixel_values_list)
    pixel_values_src = torch.stack(pixel_values_src_list)

    data_dict = dict(input_ids=input_ids, attention_mask=attention_mask,
                     pixel_values=pixel_values, pixel_values_src=pixel_values_src)

    return {'data': data_dict, 'data_samples': None}


def collate_func_gen(instances: Sequence[Dict],
                    resolutions: Union[List[Tuple[int, int]], Tuple[int, int]] = None,
                    resolutions_ratio: List[float] = None,
                    crop_mode: str = 'center_crop',
                    pad_index: int = DEFAULT_PAD_TOKEN_INDEX):
    """
    The collate function for text-to-image generation tasks, supporting multi-resolution and left padding
    """

    # 分辨率处理
    if resolutions is None:
        target_resolution = (512, 512)
    elif isinstance(resolutions, tuple):
        target_resolution = resolutions
    else:
        target_resolution = select_resolution(resolutions, resolutions_ratio)
    
    pixel_values, input_ids = [], []

    for example in instances:
        pixel_value = example.pop('pixel_values')
        if isinstance(pixel_value, torch.Tensor):
            pixel_value = smart_resize_and_crop(pixel_value, target_resolution, crop_mode)
        pixel_values.append(pixel_value)
        
        ids = example.pop('input_ids')
        if not isinstance(ids, torch.Tensor):
            ids = torch.tensor(ids, dtype=torch.long)
        input_ids.append(ids)

    input_ids, attention_mask = pad_input_ids(
        input_ids,
        batch_first=True,
        padding_value=pad_index,
        padding_mode="right"
    )

    data_dict = dict(
        pixel_values=torch.stack(pixel_values),
        input_ids=input_ids,
        attention_mask=attention_mask,
        batch_resolution=target_resolution
    )

    return {'data': data_dict, 'data_samples': None}


def collate_func_gen_tokens(instances: Sequence[Dict],
                            pad_index: int = DEFAULT_PAD_TOKEN_INDEX):
    image_tokens, input_ids, input_lengths = [], [], []
    for example in instances:
        image_tokens.append(example.pop('image_tokens'))
        input_lengths.append(len(example['input_ids']))
        input_ids.append(example.pop('input_ids'))

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_index)
    attention_mask = torch.zeros_like(input_ids).bool()
    for i in range(len(input_ids)):
        attention_mask[i, :input_lengths[i]] = True

    data_dict = dict(image_tokens=torch.stack(image_tokens),
                     input_ids=input_ids,
                     attention_mask=attention_mask)

    return {'data': data_dict, 'data_samples': None}


def collate_func_gen_latents(instances: Sequence[Dict],
                             pad_index: int = DEFAULT_PAD_TOKEN_INDEX):
    image_latents, input_ids, input_lengths = [], [], []
    for example in instances:
        image_latents.append(example.pop('image_latents'))
        input_lengths.append(len(example['input_ids']))
        input_ids.append(example.pop('input_ids'))

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_index)
    attention_mask = torch.zeros_like(input_ids).bool()
    for i in range(len(input_ids)):
        attention_mask[i, :input_lengths[i]] = True

    data_dict = dict(image_latents=torch.stack(image_latents),
                     input_ids=input_ids,
                     attention_mask=attention_mask)

    return {'data': data_dict, 'data_samples': None}


def collate_func_und(instances, pad_index=DEFAULT_PAD_TOKEN_INDEX):
    input_ids_list, labels_list, pixel_values_list = [], [], []

    for sample in instances:
        input_ids_list.append(torch.LongTensor(sample['input_ids']))
        labels_list.append(torch.LongTensor(sample['labels']))

        if 'pixel_values' in sample:
            pixel_values_list.append(sample['pixel_values'])

    ori_length = [len(input_ids_) for input_ids_ in input_ids_list]
    # right padding
    if len(instances) > 1:
        input_ids = pad_sequence(
            input_ids_list, batch_first=True, padding_value=pad_index)
        labels = pad_sequence(
            labels_list, batch_first=True, padding_value=IGNORE_INDEX)
    else:
        input_ids = torch.stack(input_ids_list)
        labels = torch.stack(labels_list)

    attention_mask = torch.zeros_like(input_ids).bool()
    for i, length in enumerate(ori_length):
        attention_mask[i, :length] = True        # right padding

    data_dict = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'pixel_values': torch.stack(pixel_values_list) if len(pixel_values_list) > 0 else None
    }

    return {'data': data_dict, 'data_samples': None}


class CollateConcat(object):
    def __init__(self, collate_fns, keys):
        self.keys = keys
        self.collate_fns = {}
        for key, collate_fn in zip(keys, collate_fns):
            func = collate_fn.pop('type')
            self.collate_fns[key] = partial(func, **collate_fn)

    def __call__(self, data_samples):
        data_samples = [data_sample for data_sample in data_samples if len(data_sample) > 0]
        data_dict = {}
        key = data_samples[0]['type']
        data_dict[key] = self.collate_fns[key](data_samples)['data']

        return {'data': data_dict, 'data_samples': None}
