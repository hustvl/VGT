
# Midjourney Captioned 23M

This dataset is from [deepghs/midjourney_captioned_23m_full](https://huggingface.co/datasets/deepghs/midjourney_captioned_23m_full).
It contains 23 million Midjourney images with captions, organized in tar archives and parquet metadata files.

## Download

Download the dataset by:
```shell
cd /path/to/VGT
huggingface-cli download deepghs/midjourney_captioned_23m_full --local-dir data/midjourney-23m/raw --repo-type dataset
```

```text
VGT/
├── data
    ├── midjourney-23m
        ├── raw
            ├── images
                ├── (subdirectories with .tar files)
            ├── table-1.parquet
            ├── table-2.parquet
            ├── table-3.parquet
            ├── table-4.parquet
            ├── table-5.parquet
        ├── images
        ├── data.json
```

## Extract Images from Tar Files

The images are stored in tar files across multiple subdirectories. We need to extract them:

```shell
cd data/midjourney-23m/raw/images
vim extract.py
```

Write the following into extract.py:

```python
import multiprocessing as mp
import argparse
import os
from tqdm import tqdm
from glob import glob
import subprocess


def single_process(tar_list, output_dir):
    for tar_file in tqdm(tar_list):
        # Extract directly to output directory
        subprocess.run(["tar", "-xf", tar_file, "-C", output_dir, "--no-same-owner"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=-1, type=int)
    parser.add_argument('--num-processes', default=8, type=int)
    args = parser.parse_args()

    # Create output directory
    output_dir = '../../images'
    os.makedirs(output_dir, exist_ok=True)

    # Recursively find all tar files
    tar_files = sorted(glob('**/*.tar', recursive=True))
    
    if args.end == -1:
        args.end = len(tar_files)

    tar_files = tar_files[args.start:args.end]

    num_tars = len(tar_files)
    num_processes = args.num_processes
    num_tars_per_process = num_tars // num_processes
    res = num_tars % num_processes
    if res > 0:
        num_processes += 1

    processes = [mp.Process(target=single_process,
                            args=(tar_files[process_id * num_tars_per_process:
                                            (process_id + 1) * num_tars_per_process]
                                  if process_id < num_processes - 1
                                  else tar_files[process_id * num_tars_per_process:],
                                  output_dir))
                 for process_id in range(num_processes)]

    # Run processes
    for p in processes:
        p.start()

    # Exit the completed processes
    for p in processes:
        p.join()

```

Then run the python file to extract all tar files:

```shell
python extract.py --num-processes 8
# python extract.py --num-processes 8 --start 0 --end 100   # you can also process a part of the .tars in a single task and launch many tasks   
```

## Process Parquet Metadata

After extracting images, we need to process the parquet files to create the data.json:

```shell
cd data/midjourney-23m
vim process_metadata.py
```

Write the following into process_metadata.py:

```python
import pandas as pd
from glob import glob
from tqdm import tqdm
import json
import os


def process_parquet_files():
    """Process all parquet files and create data.json"""
    parquet_files = sorted(glob('raw/table-*.parquet'))
    
    all_data = []
    
    for parquet_file in tqdm(parquet_files, desc="Processing parquet files"):
        df = pd.read_parquet(parquet_file)
        
        for idx, row in df.iterrows():
            filename = row['filename']
            prompt = row['prompt']
            width = row['width']
            height = row['height']
            
            # Check if image exists
            image_path = f"images/{filename}"
            if os.path.exists(image_path):
                all_data.append({
                    'image': image_path,
                    'caption': prompt,
                    'width': int(width),
                    'height': int(height)
                })
    
    # Save data.json
    with open('data.json', 'w') as f:
        json.dump(all_data, f, indent=2)
    
    print(f"Processed {len(all_data)} image-caption pairs")
    print(f"Saved to data.json")


if __name__ == '__main__':
    process_parquet_files()

```

Then run the python file:

```shell
python process_metadata.py
```

## Set config

```python
from src.datasets.text2image.caption_datasets import CaptionDataset
from mmengine.config import read_base
from mmengine.dataset import InfiniteSampler
from xtuner.dataset import ConcatDataset

with read_base():
    from .processors import prompt_template, tokenizer, image_size, pad_index

max_length = 128

dataset = dict(type=CaptionDataset,
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
               max_length=max_length)
```
