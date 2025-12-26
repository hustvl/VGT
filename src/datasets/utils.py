# Modified from rom https://github.com/wusize/OpenUni
import copy
import random
from xtuner.dataset.utils import get_bos_eos_token_ids
from xtuner.utils import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX
import json
from typing import List, Tuple
import torch

INPUT_IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
OUTPUT_IMAGE_TOKEN_INDEX = -300



def pad_input_ids(
    input_ids: List[torch.Tensor],
    batch_first: bool = True,
    padding_value: int = 0,
    padding_mode: str = "left"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对一批 input_ids 执行 padding，返回 (padded_input_ids, attention_mask)

    Args:
        input_ids (List[Tensor]): 每个样本的 token id 张量，形状 [seq_len]
        batch_first (bool): 是否 batch 维在前，默认 True => [B, T]
        padding_value (int): padding 的填充值（例如 pad_token_id）
        padding_mode (str): 'left' 或 'right'，决定填充方向

    Returns:
        Tuple[Tensor, Tensor]:
            - padded_input_ids: [B, T] 或 [T, B]
            - attention_mask: 同形状，padding 为 False，其余为 True
    """

    # 校验输入
    if len(input_ids) == 0:
        raise ValueError("input_ids 不能为空")

    if padding_mode not in ("left", "right"):
        raise ValueError(f"padding_mode 必须是 'left' 或 'right'，但得到 {padding_mode}")

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

def crop2square(pil_img):
    width, height = pil_img.width, pil_img.height

    if width > height:
        y0, y1 = 0, height
        x0 = random.randint(0, width - height)    # [0, w - h]
        x1 = x0 + height    # [h, w]
    else:
        x0, x1 = 0, width
        y0 = random.randint(0, height - width)   # [0, h - w]
        y1 = y0 + width     # [w, h]

    return pil_img.crop(box=(x0, y0, x1, y1))


def load_jsonl(json_file):
    with open(json_file) as f:
        lines = f.readlines()
    data = []
    for line in lines:
        data.append(json.loads(line))
    return data


def encode_fn(example,
              tokenizer,
              max_length=None,
              image_length=1,
              input_ids_with_output=True,
              with_image_token=False,
              truncation='right'):
    """We only support the following three scenarios:

    1. Incremental pretraining dataset.
        example['conversation'] = [
                {
                    'input': '',
                    'output': '### Human: Can you write xxx'
                }
            ]

    2. Single-turn conversation dataset.
        example['conversation'] = [
                {
                    'input': 'Give three tips for staying healthy.',
                    'output': '1.Eat a balanced diet xxx'
                }
            ]

    3. Multi-turn conversation dataset.
        example['conversation'] = [
                {
                    'input': 'Give three tips for staying healthy.',
                    'output': '1.Eat a balanced diet xxx'
                },
                {
                    'input': 'Please expand on the second point.',
                    'output': 'Here is an expanded explanation of the xxx'
                }
            ]
    """
    bos_token_id, eos_token_id = get_bos_eos_token_ids(tokenizer)
    is_multi_turn_conversation = len(example['conversation']) > 1
    if is_multi_turn_conversation:
        assert input_ids_with_output

    input_ids, labels = [], []
    next_needs_bos_token = True
    for single_turn_conversation in example['conversation']:
        input = single_turn_conversation['input']
        if DEFAULT_IMAGE_TOKEN in input and with_image_token:
            chunk_encode = [
                tokenizer.encode(chunk, add_special_tokens=False)
                for chunk in input.split(DEFAULT_IMAGE_TOKEN)
            ]
            assert len(chunk_encode) == 2
            input_encode = []
            for idx, cur_chunk_encode in enumerate(chunk_encode):
                input_encode.extend(cur_chunk_encode)
                if idx != len(chunk_encode) - 1:
                    # input_encode.append(IMAGE_TOKEN_INDEX)
                    input_encode += [IMAGE_TOKEN_INDEX] * image_length
        else:
            input_encode = tokenizer.encode(input, add_special_tokens=False)
        if next_needs_bos_token:
            input_ids += bos_token_id
            labels += [IGNORE_INDEX] * len(bos_token_id)
        input_ids += input_encode
        labels += [IGNORE_INDEX] * len(input_encode)
        if input_ids_with_output and 'output' in single_turn_conversation:
            # Add output
            output_with_loss = single_turn_conversation.get(
                'output_with_loss', True)
            output = single_turn_conversation['output']
            if DEFAULT_IMAGE_TOKEN in output and with_image_token:
                chunk_encode = [
                    tokenizer.encode(chunk, add_special_tokens=False)
                    for chunk in output.split(DEFAULT_IMAGE_TOKEN)
                ]
                assert len(chunk_encode) == 2
                output_encode = []
                for idx, cur_chunk_encode in enumerate(chunk_encode):
                    output_encode.extend(cur_chunk_encode)
                    if idx != len(chunk_encode) - 1:
                        output_encode += [IMAGE_TOKEN_INDEX] * image_length
            else:
                output_encode = tokenizer.encode(output, add_special_tokens=False)
            # output_encode = tokenizer.encode(output, add_special_tokens=False)
            input_ids += output_encode
            if output_with_loss:
                labels += copy.deepcopy(output_encode)
            else:
                labels += [IGNORE_INDEX] * len(output_encode)
            # Add EOS_TOKEN (with loss)
            if single_turn_conversation.get('need_eos_token', True):
                next_needs_bos_token = True
                input_ids += eos_token_id
                if output_with_loss:
                    labels += copy.deepcopy(eos_token_id)
                else:
                    labels += [IGNORE_INDEX] * len(eos_token_id)
            else:
                next_needs_bos_token = False
            # Add SEP (without loss)
            sep = single_turn_conversation.get('sep', '')
            if sep != '':
                sep_encode = tokenizer.encode(sep, add_special_tokens=False)
                input_ids += sep_encode
                labels += [IGNORE_INDEX] * len(sep_encode)

    if max_length is not None and len(input_ids) > max_length:
        if truncation == 'right':
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]
        elif truncation == 'left':
            input_ids = input_ids[-max_length:]
            labels = labels[-max_length:]
        else:
            assert truncation is None
    return {'input_ids': input_ids, 'labels': labels}
