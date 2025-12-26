import json
import os
import copy
import torch
import argparse
from tqdm import tqdm
from xtuner.registry import BUILDER
from mmengine.config import Config
from accelerate import Accelerator
from accelerate.utils import gather_object
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from einops import rearrange
from src.utils import load_checkpoint_with_ema
from cleanfid import fid


class MJHQ30K(Dataset):
    def __init__(self, meta_data_path):
        """
        Args:
            meta_data_path: path to meta_data.json file
        """
        with open(meta_data_path, 'r') as f:
            self.data = json.load(f)
        
        # Convert to list format for easier indexing
        self.items = [(img_id, info) for img_id, info in self.data.items()]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_id, info = self.items[idx]
        data_dict = {
            'img_id': img_id,
            'prompt': info['prompt'],
            'category': info['category'],
            'idx': idx
        }
        return data_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('config', help='config file path.')
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--meta_data', default='meta_data.json', type=str, 
                        help='Path to meta_data.json file')
    parser.add_argument('--ref_dir', default='mjhq30k_imgs', type=str,
                        help='Path to reference images directory')
    parser.add_argument('--output', default='output_mjhq', type=str)
    parser.add_argument("--cfg_prompt", type=str, default=None)
    parser.add_argument("--cfg_scale", type=float, default=4.5)
    parser.add_argument("--total_step", type=int, default=256)
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=1,
                        help='Number of samples to generate per prompt')

    args = parser.parse_args()
    print(args)
    
    accelerator = Accelerator()
    message = [f"Hello this is GPU {accelerator.process_index}"]
    messages = gather_object(message)
    accelerator.print(f"Number of gpus: {accelerator.num_processes}")
    accelerator.print(messages)

    config = Config.fromfile(args.config)
    print(f'Device: {accelerator.device}', flush=True)

    # Load dataset
    dataset = MJHQ30K(meta_data_path=args.meta_data)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=lambda x: x
    )

    # Build and load model
    model = BUILDER.build(config.model)
    if args.checkpoint is not None:
        load_checkpoint_with_ema(model, args.checkpoint, use_ema=True, map_location='cpu', strict=False)
    model = model.to(device=accelerator.device)
    model = model.to(model.dtype)
    model.eval()

    dataloader = accelerator.prepare(dataloader)
    print(f'Number of batches: {len(dataloader)}', flush=True)

    # Prepare CFG prompt if needed
    if args.cfg_prompt is None:
        cfg_prompt = model.prompt_template['CFG']
    else:
        cfg_prompt = model.prompt_template['GENERATION'].format(input=args.cfg_prompt.strip())
    cfg_prompt = model.prompt_template['INSTRUCTION'].format(input=cfg_prompt)

    # Create output directory structure
    if accelerator.is_main_process:
        os.makedirs(args.output, exist_ok=True)
        # Create category subdirectories
        categories = set([item[1]['category'] for item in dataset.items])
        for category in categories:
            os.makedirs(os.path.join(args.output, category), exist_ok=True)

    accelerator.wait_for_everyone()

    generator = torch.Generator(device=model.device).manual_seed(args.seed)

    # Generate images
    for batch_idx, data_samples in tqdm(enumerate(dataloader), total=len(dataloader), disable=not accelerator.is_main_process):
        prompts = []
        for data_sample in data_samples:
            prompt = copy.deepcopy(data_sample['prompt'].strip())
            prompts.append(prompt)

        # Replicate prompts if generating multiple samples per prompt
        prompts = prompts * args.num_samples

        # Prepare inputs
        inputs = model.prepare_batch_text_conditions(prompts)
        
        # Generate images
        images = model.generate(
            **inputs, 
            progress_bar=False,
            cfg_scale=args.cfg_scale, 
            num_steps=args.num_steps,
            generator=generator, 
            height=args.height, 
            width=args.width, 
            total_step=args.total_step
        )
        
        # Rearrange if multiple samples per prompt
        if args.num_samples > 1:
            images = rearrange(images, '(n b) c h w -> b n h w c', n=args.num_samples)
        else:
            images = rearrange(images, 'b c h w -> b () h w c')
        
        # Convert to uint8
        images = torch.clamp(127.5 * images + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()

        # Save images with the same naming as reference images
        for image_samples, data_sample in zip(images, data_samples):
            name = data_sample['img_id']  # This is the key from meta_data.json
            category = data_sample['category']
            sub_path = os.path.join(args.output, category)
            
            # For single sample, save directly with name as filename
            if args.num_samples == 1:
                Image.fromarray(image_samples[0]).save('{}/{}.png'.format(sub_path, name))
            else:
                # For multiple samples, save with suffix
                for i, sub_image in enumerate(image_samples):
                    Image.fromarray(sub_image).save('{}/{}_{:02d}.png'.format(sub_path, name, i))

    try:
        accelerator.wait_for_everyone()
    except Exception as e:
        print(f"wait_for_everyone: {e}")

    # Calculate FID score on main process
    if accelerator.is_main_process:
        print("\n" + "="*50)
        print("Generation completed. Calculating FID scores...")
        print("="*50 + "\n")
        
        # Calculate overall FID
        try:
            overall_fid = fid.compute_fid(args.ref_dir, args.output)
            print(f"Overall FID Score: {overall_fid:.4f}")
        except Exception as e:
            print(f"Error calculating overall FID: {e}")
        
        # # Calculate per-category FID
        # categories = ['animals', 'art', 'fashion', 'food', 'indoor', 
        #              'landscape', 'logo', 'people', 'plants', 'vehicles']
        
        # print("\nPer-Category FID Scores:")
        # print("-" * 50)
        # category_scores = {}
        # for category in categories:
        #     ref_category_dir = os.path.join(args.ref_dir, category)
        #     gen_category_dir = os.path.join(args.output, category)
            
        #     if os.path.exists(ref_category_dir) and os.path.exists(gen_category_dir):
        #         try:
        #             category_fid = fid.compute_fid(ref_category_dir, gen_category_dir)
        #             category_scores[category] = category_fid
        #             print(f"{category:12s}: {category_fid:.4f}")
        #         except Exception as e:
        #             print(f"{category:12s}: Error - {e}")
        
        # Save results to JSON
        results = {
            'overall_fid': overall_fid if 'overall_fid' in locals() else None,
            # 'category_fid': category_scores,
            'config': {
                'checkpoint': args.checkpoint,
                'cfg_scale': args.cfg_scale,
                'num_steps': args.num_steps,
                'height': args.height,
                'width': args.width,
                'seed': args.seed,
                'num_samples': args.num_samples
            }
        }
        
        results_path = os.path.join(args.output, 'fid_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "="*50)
        print(f"Results saved to: {results_path}")
        print("="*50)