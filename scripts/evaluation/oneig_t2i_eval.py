import json
import os
import copy
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from PIL import Image
from xtuner.registry import BUILDER
from mmengine.config import Config
from accelerate import Accelerator
from accelerate.utils import gather_object
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
import megfile
import inflect
from src.utils import load_checkpoint_with_ema
from src.utils import _generate_text_to_image

p = inflect.engine()

def create_image_gallery(images, rows=2, cols=2):
    assert len(images) >= rows * cols, "Not enough images provided!"
    
    img_height, img_width = images[0].size
    
    # Create a blank image as the gallery background
    gallery_width = cols * img_width
    gallery_height = rows * img_height
    gallery_image = Image.new("RGB", (gallery_width, gallery_height))
    
    # Paste each image onto the gallery canvas
    for row in range(rows):
        for col in range(cols):
            img = images[row * cols + col]
            x_offset = col * img_width
            y_offset = row * img_height
            gallery_image.paste(img, (x_offset, y_offset))
    
    return gallery_image

class OneIGDataset(Dataset):
    def __init__(self, data_path, language='en'):
        # Read CSV file
        self.df = pd.read_csv(data_path, dtype=str)
        self.language = language
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data_dict = {
            'id': row['id'],
            'category': row['category'],
            'prompt': row[f'prompt_{self.language}'],
            'idx': idx
        }
        return data_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--config', help='config file path.')
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--data', default='./data/OneIG-Benchmark/OneIG-Bench.csv', type=str)
    parser.add_argument('--output', default='oneig_output', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--language', default='en', choices=['en', 'zh'], help='Language for prompts')
    parser.add_argument("--cfg_prompt", type=str, default=None)
    parser.add_argument("--cfg_scale", type=float, default=4.5)
    parser.add_argument('--num_steps', type=int, default=30)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid_rows", type=int, default=2, help="Number of rows in the image grid")
    parser.add_argument("--grid_cols", type=int, default=2, help="Number of columns in the image grid")
    
    args = parser.parse_args()
    print(args)
    
    accelerator = Accelerator()
    message = [f"Hello this is GPU {accelerator.process_index}"]
    messages = gather_object(message)
    accelerator.print(f"Number of gpus: {accelerator.num_processes}")
    accelerator.print(messages)
    
    config = Config.fromfile(args.config)
    print(f'Device: {accelerator.device}', flush=True)
    
    # category to subfolder name mapping
    class_item = {
        "Anime_Stylization": "anime",
        "Portrait": "human", 
        "General_Object": "object",
        "Text_Rendering": "text",
        "Knowledge_Reasoning": "reasoning",
        "Multilingualism": "multilingualism"
    }
    
    dataset = OneIGDataset(data_path=args.data, language=args.language)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            drop_last=False,
                            collate_fn=lambda x: x)
    
    model = BUILDER.build(config.model)
    if args.checkpoint is not None:
        load_checkpoint_with_ema(model, args.checkpoint, use_ema=True, map_location='cpu', strict=False)
    model = model.to(device=accelerator.device)
    model = model.to(model.dtype)
    model.eval()
    
    dataloader = accelerator.prepare(dataloader)
    
    print(f'Number of samples: {len(dataloader)}', flush=True)

    print(f'out dir: {args.output}', flush=True)
    if args.cfg_prompt is None:
        cfg_prompt = model.prompt_template['CFG']
    else:
        cfg_prompt = model.prompt_template['GENERATION'].format(input=args.cfg_prompt.strip())
    cfg_prompt = model.prompt_template['INSTRUCTION'].format(input=cfg_prompt)
    
    if accelerator.is_main_process:
        os.makedirs(args.output, exist_ok=True)
        # Create subdirectories for each category
        for category in class_item.values():
            os.makedirs(os.path.join(args.output, category), exist_ok=True)
    
    generator = torch.Generator(device=model.device).manual_seed(args.seed)
    
    for batch_idx, data_samples in tqdm(enumerate(dataloader), disable=not accelerator.is_main_process):
        device_idx = accelerator.process_index
        
        prompts = []
        categories = []
        ids = []
        
        for data_sample in data_samples:
            prompt = data_sample['prompt'].strip()
            prompts.append(prompt)
            categories.append(data_sample['category'])
            ids.append(data_sample['id'])
        
        # Generate 4 images per prompt (for 2x2 grid)
        num_images_per_prompt = args.grid_rows * args.grid_cols
        prompts = prompts * num_images_per_prompt
        categories = categories * num_images_per_prompt
        ids = ids * num_images_per_prompt
        
        inputs = model.prepare_batch_text_conditions(prompts)
        
        images = model.generate(**inputs, progress_bar=False,
                                cfg_scale=args.cfg_scale, num_steps=args.num_steps,
                                generator=generator, height=args.height, width=args.width)
        
        images = rearrange(images, '(n b) c h w -> b n h w c', n=num_images_per_prompt)
        images = torch.clamp(
            127.5 * images + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()
        
        # Save samples to disk as individual .webp files (grid format)
        for image, category, sample_id in zip(images, categories, ids):
            # Create category subdirectory
            category_dir = class_item[category]
            output_dir = os.path.join(args.output, category_dir, args.model_name)
            
            # Create gallery image from individual images
            pil_images = [Image.fromarray(img) for img in image]
            gallery_image = create_image_gallery(pil_images, args.grid_rows, args.grid_cols)
            
            # Save as webp file
            file_path = megfile.smart_path_join(output_dir, f"{sample_id}.webp")
            with megfile.smart_open(file_path, "wb") as f:
                gallery_image.save(f)