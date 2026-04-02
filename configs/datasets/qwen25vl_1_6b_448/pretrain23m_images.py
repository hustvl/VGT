from src.datasets.collate_functions import collate_func_gen_latents, collate_func_img2img, collate_func_gen
from mmengine.config import read_base
from mmengine.dataset import InfiniteSampler
from xtuner.dataset import ConcatDataset
from src.datasets.text2image.caption_datasets import CaptionDataset

with read_base():
    from .processors import prompt_template, tokenizer, image_size, pad_index

max_length = 256

megalith10m_dataset = dict(type=CaptionDataset,
               image_size=image_size,
               cap_source='caption_internlm2_short',
               cap_folder='data/megalith-10m/captions',
               data_path='data/megalith-10m/megalith10m_all.json',
               image_folder='data/megalith-10m/raw',
            #    image_latents_folder=f'data/megalith-10m/raw_dc32_{image_size}',
               unconditional=0.1,
               prompt_template=prompt_template,
               ceph_folder=None,
               ceph_config=None,
               tokenizer=tokenizer,
               max_length=max_length)

redcaps5m_latents = dict(type=CaptionDataset,
               image_size=image_size,
               cap_source='re_caption',
               cap_folder='data/redcaps5m/raw',
               data_path='data/redcaps5m/data.json',
               image_folder='data/redcaps5m/raw',
            #    image_latents_folder=f'data/redcaps5m/raw_dc32_{image_size}',
               unconditional=0.1,
               prompt_template=prompt_template,
               ceph_folder=None,
               ceph_config=None,
               tokenizer=tokenizer,
               max_length=max_length)

laion6m_dataset = dict(type=CaptionDataset,
               image_size=image_size,
               cap_source='re_caption',
               cap_folder='data/laion6m/raw',
               data_path='data/laion6m/data.json',
               image_folder='data/laion6m/raw',
            #    image_latents_folder=f'data/laion6m/raw_dc32_{image_size}',
               unconditional=0.1,
               prompt_template=prompt_template,
               ceph_folder=None,
               ceph_config=None,
               tokenizer=tokenizer,
               max_length=max_length)

t2i_2m = dict(type=CaptionDataset,
              image_size=image_size,
              cap_source='prompt',
              data_path='data/text-to-image-2M/data/data_512_2M.json',
              cap_folder='data/text-to-image-2M/raw/data_512_2M',
              image_folder='data/text-to-image-2M/raw/data_512_2M',
            #   image_latents_folder=f'data/text-to-image-2M/raw/data_512_2M_dc32_{image_size}',
              unconditional=0.1,
              prompt_template=prompt_template,
              ceph_folder=None,
              ceph_config=None,
              tokenizer=tokenizer,
              max_length=max_length)

t2i_10k = dict(type=CaptionDataset,
               image_size=image_size,
               cap_source='prompt',
               data_path='data/text-to-image-2M/data/data_1024_10K.json',
               cap_folder='data/text-to-image-2M/raw/data_1024_10K',
               image_folder='data/text-to-image-2M/raw/data_1024_10K',
            #    image_latents_folder=f'data/text-to-image-2M/raw/data_1024_10K_dc32_{image_size}',
               unconditional=0.1,
               prompt_template=prompt_template,
               ceph_folder=None,
               ceph_config=None,
               tokenizer=tokenizer,
               max_length=max_length,
               )

midjourney_23M_dataset = dict(type=CaptionDataset,
               image_size=image_size,
               cap_source='caption',
               data_path='data/midjourney-23m/data.json',
               cap_folder='data/midjourney-23m',
               image_folder='data/midjourney-23m',
               unconditional=0.1,
               prompt_template=prompt_template,
               ceph_folder=None,
               ceph_config=None,
               tokenizer=tokenizer,
               max_length=max_length
            )

imagenet1k_1M_dataset = dict(type=CaptionDataset,
               image_size=image_size,
               cap_source='caption',
               data_path='data/imagenet1k-t2i/data.json',
               cap_folder='data/imagenet1k-t2i',
               image_folder='data/imagenet1k-t2i',
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
         megalith10m_dataset, laion6m_dataset, midjourney_23M_dataset, t2i_2m, imagenet1k_1M_dataset],
)


train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    pin_memory=True,
    dataset=dataset,
    sampler=dict(type=InfiniteSampler, shuffle=True),
    collate_fn=dict(type=collate_func_gen,
                    resolutions=(image_size, image_size),  # fix
                    pad_index=pad_index)
)
