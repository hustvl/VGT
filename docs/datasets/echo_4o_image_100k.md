
# Echo-4o-Image

This dataset is from [Yejy53/Echo-4o-Image](https://huggingface.co/datasets/Yejy53/Echo-4o-Image).
It contains instruction-following images and surreal fantasy images with detailed captions. We only use the text-to-image (t2i) task type data.

## Download

Download the dataset by:
```shell
cd /path/to/VGT
huggingface-cli download Yejy53/Echo-4o-Image --local-dir data/echo-4o-image/raw --repo-type dataset
```

```text
VGT/
├── data
    ├── echo-4o-image
        ├── raw
            ├── Instruction-Following-Image
                ├── images
                    ├── *.tar.gz
                ├── Instruction-Following-Image.jsonl
            ├── Surrel-Fantasy-Image
                ├── images
                    ├── *.tar.gz
                ├── conflict.jsonl
        ├── images
        ├── data.json
```

## Extract Images from Tar.gz Files

The images are stored in tar.gz files. We need to extract them to a single directory:

```shell
cd data/echo-4o-image
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
        # Extract tar.gz files directly to output directory
        subprocess.run(["tar", "-xzf", tar_file, "-C", output_dir, "--no-same-owner"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-processes', default=8, type=int)
    args = parser.parse_args()

    # Create output directory
    output_dir = 'images'
    os.makedirs(output_dir, exist_ok=True)

    # Find all tar.gz files from both subdirectories
    tar_files = []
    tar_files.extend(glob('raw/Instruction-Following-Image/images/*.tar.gz'))
    tar_files.extend(glob('raw/Surrel-Fantasy-Image/images/*.tar.gz'))
    tar_files = sorted(tar_files)
    
    print(f"Found {len(tar_files)} tar.gz files to extract")

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
    
    print(f"Extraction completed. Images saved to {output_dir}")

```

Then run the python file to extract all tar.gz files:

```shell
python extract.py --num-processes 8
```

## Process JSONL Metadata

After extracting images, we need to process the jsonl files to create data.json, filtering only t2i task type:

```shell
cd data/echo-4o-image
vim process_metadata.py
```

Write the following into process_metadata.py:

```python
import json
from tqdm import tqdm
import os


def process_jsonl_files():
    """Process all jsonl files and create data.json, filtering only t2i tasks"""
    jsonl_files = [
        'raw/Instruction-Following-Image/Instruction-Following-Image.jsonl',
        'raw/Surrel-Fantasy-Image/conflict.jsonl'
    ]
    
    all_data = []
    
    for jsonl_file in jsonl_files:
        print(f"Processing {jsonl_file}...")
        
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"Reading {os.path.basename(jsonl_file)}"):
                data = json.loads(line.strip())
                
                # Only process t2i task type
                if data.get('task_type') == 't2i':
                    output_image = data.get('output_image', '')
                    instruction = data.get('instruction', '')
                    
                    # Extract filename from path
                    # e.g., "/Echo-4o-Image/Surrel-Fantasy-Image/images/37547.jpg" -> "37547.jpg"
                    if output_image:
                        filename = os.path.basename(output_image)
                        image_path = f"images/{filename}"
                        
                        # Check if image exists
                        if os.path.exists(image_path):
                            all_data.append({
                                'image': image_path,
                                'caption': instruction,
                                'type': data.get('type', '')
                            })
                        else:
                            # Try without extension change
                            possible_extensions = ['.jpg', '.png', '.jpeg', '.webp']
                            found = False
                            base_name = os.path.splitext(filename)[0]
                            
                            for ext in possible_extensions:
                                test_path = f"images/{base_name}{ext}"
                                if os.path.exists(test_path):
                                    all_data.append({
                                        'image': test_path,
                                        'caption': instruction,
                                        'type': data.get('type', '')
                                    })
                                    found = True
                                    break
    
    # Save data.json
    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nProcessed {len(all_data)} t2i image-caption pairs")
    print(f"Saved to data.json")


if __name__ == '__main__':
    process_jsonl_files()

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
               data_path='data/echo-4o-image/data.json',
               cap_folder='data/echo-4o-image',
               image_folder='data/echo-4o-image',
               unconditional=0.1,
               prompt_template=prompt_template,
               ceph_folder=None,
               ceph_config=None,
               tokenizer=tokenizer,
               max_length=max_length)
```
