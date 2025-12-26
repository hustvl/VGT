from mmengine.config import read_base
from mmengine.dataset import InfiniteSampler
from src.datasets.collate_functions import collate_func_gen_latents, collate_func_img2img, collate_func_gen
from src.datasets.text2image.blip3_o import BLIP3oDataset
from xtuner.dataset import ConcatDataset

with read_base():
    from .processors import prompt_template, tokenizer, image_size, pad_index

max_length = 256

blip3060k_dataset = dict(type=BLIP3oDataset,
               image_size=image_size,
               data_path='data/BLIP3o-60k/data.json',
               cap_folder='data/BLIP3o-60k/raw',
               image_folder='data/BLIP3o-60k/raw',
               unconditional=0.1,
               prompt_template=prompt_template,
               ceph_folder=None,
               ceph_config=None,
               tokenizer=tokenizer,
               max_length=max_length)

echo4o_t2i_100k_dataset = dict(type=CaptionDataset,
               image_size=image_size,
               cap_source='caption',
               data_path='data/echo-4o-image/data.json',
               cap_folder='data/echo-4o-image',
               image_folder='data/echo-4o-image',
               unconditional=0.1,
               prompt_template=prompt_template,
               ceph_folder=None,
               ceph_config=None,
               tokenizer=tokenizer,
               max_length=max_length
            )

share4o_t2i_45k_dataset =  dict(type=CaptionDataset,
               image_size=image_size,
               cap_source='caption',
               data_path='data/sharegpt-4o-image/data.json',
               cap_folder='data/sharegpt-4o-image',
               image_folder='data/sharegpt-4o-image',
               unconditional=0.1,
               prompt_template=prompt_template,
               ceph_folder=None,
               ceph_config=None,
               tokenizer=tokenizer,
               max_length=max_length
            )

dataset = dict(
    type=ConcatDataset,
    datasets=[
    blip3o60k_dataset, 
    echo4o_t2i_100k_dataset,
    share4o_t2i_45k_dataset
    ],
)

train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    pin_memory=True,
    dataset=dataset,
    sampler=dict(type=InfiniteSampler, shuffle=True),
    collate_fn=dict(type=collate_func_gen,
                    resolutions=(image_size, image_size),  # fix
                    pad_index=pad_index)
)
