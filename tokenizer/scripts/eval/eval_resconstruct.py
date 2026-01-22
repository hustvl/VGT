"""This file contains a class to evaluate the reconstruction results.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License. 
"""

import warnings

from typing import Sequence, Optional, Mapping, Text
import numpy as np
from scipy import linalg
import torch
import torch.nn.functional as F

import os
import argparse
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from accelerate import Accelerator
from accelerate.utils import set_seed

from evaluator import Evaluator
import sys
import os
from pathlib import Path
from omegaconf import OmegaConf
import torch
from datasets import load_dataset

# Import necessary modules
from modeling.modules import EMAModel, loss_map
from modeling import model_map

def load_ae_model(config_path, checkpoint_path, accelerator):
    """Load RQAE model for image reconstruction evaluation
    
    Args:
        config_path: Path to YAML configuration file
        checkpoint_path: Path to model checkpoint file
        accelerator: Accelerator object
    
    Returns:
        model: Loaded model
    """
    
    # Load configuration file
    config = OmegaConf.load(config_path)
    
    # Get model type
    model_type = config.model.name
    if model_type not in model_map:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create model
    model_cls = model_map[model_type]
    config.model.checkpoint_path = checkpoint_path
    model = model_cls(config)
    
    # Load checkpoint
    accelerator.print(f"Loading checkpoint from {checkpoint_path}")
    # checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # # Handle different checkpoint formats
    # if isinstance(checkpoint, dict):
    #     if 'state_dict' in checkpoint:
    #         state_dict = checkpoint['state_dict']
    #     elif 'model' in checkpoint:
    #         state_dict = checkpoint['model']
    #     else:
    #         state_dict = checkpoint
    # else:
    #     state_dict = checkpoint
    
    # Load weights
    # msg = model.load_state_dict(state_dict, strict=True)
    # accelerator.print(f"Load checkpoint message: {msg}")
    
    # Set to evaluation mode
    model.eval()
    
    accelerator.print(f"Model loaded successfully")
    return model


def load_model(model_path, device):
    """General model loading function (for backward compatibility)
    
    Args:
        model_path: Path to model file
        device: Device (cuda/cpu)
    
    Returns:
        model: Loaded model
    """
    import torch
    
    if model_path.endswith('.pth') or model_path.endswith('.pt'):
        model = torch.load(model_path, map_location=device)
        if hasattr(model, 'eval'):
            model.eval()
        return model
    else:
        raise ValueError(f"Unsupported model format: {model_path}")


class HuggingFaceImageNetDataset(Dataset):
    """HuggingFace ImageNet dataset wrapper for streaming validation data"""
    
    def __init__(self, transform=None, use_auth_token=True):
        """
        Args:
            transform: Image transformation
            use_auth_token: Whether to use HuggingFace authentication token
        """
        self.transform = transform
        self.use_auth_token = use_auth_token
        
        # Load ImageNet validation dataset from HuggingFace
        self.dataset = load_dataset(
            "imagenet-1k", 
            streaming=True, 
            split="validation", 
            use_auth_token=use_auth_token
        )

        # Convert streaming dataset to list for indexing
        self.data_list = list(self.dataset)
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        example = self.data_list[idx]
        image = example["image"].convert('RGB')
        label = example["label"]
        
        if self.transform:
            image = self.transform(image)
        
        # Return image and a placeholder path (since streaming dataset doesn't have paths)
        return image, f"imagenet_val_{idx:08d}"


def create_transform(image_size):
    """Create image preprocessing transformation"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),  # Center crop to square
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
    )


def _process_buffer(all_original_images, all_reconstructed_images, accelerator, evaluator, eval_batch_size):
    """Process all collected image data for evaluation
    
    Args:
        all_original_images: List of all original images
        all_reconstructed_images: List of all reconstructed images
        accelerator: Accelerator object
        evaluator: Evaluator object
        eval_batch_size: Evaluation batch size
    """
    if len(all_original_images) == 0:
        return
    
    # Concatenate all batches on CPU
    all_original_images = torch.cat(all_original_images, dim=0)
    all_reconstructed_images = torch.cat(all_reconstructed_images, dim=0)
    
    # Gather from all processes
    all_original_images = accelerator.gather(all_original_images)
    all_reconstructed_images = accelerator.gather(all_reconstructed_images)
    
    # Only evaluate on main process
    if accelerator.is_main_process:
        total_bs = all_original_images.shape[0]
        # accelerator.print(f"Processing {total_bs} images for evaluation")
        
        # Process in batches to avoid memory issues
        for i in range(0, total_bs, eval_batch_size):
            end_idx = min(i + eval_batch_size, total_bs)
            batch_original_images = all_original_images[i:end_idx].to(accelerator.device)
            batch_reconstructed_images = all_reconstructed_images[i:end_idx].to(accelerator.device)
            
            # Update evaluator
            evaluator.update(batch_original_images, batch_reconstructed_images, None)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Image reconstruction quality evaluation script')
    parser.add_argument('--config_path', type=str, required=True,
                        help='Path to YAML configuration file')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to model checkpoint file')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--model_image_size', type=int, default=448,
                        help='Model input image size')
    parser.add_argument('--eval_image_size', type=int, default=256,
                        help='Evaluation image size')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device type (cuda/cpu)')
    parser.add_argument('--enable_rfid', action='store_true', default=True,
                        help='Enable rFID evaluation')
    parser.add_argument('--enable_inception_score', action='store_true', default=True,
                        help='Enable Inception Score evaluation')
    parser.add_argument('--buffer_size', type=int, default=4,
                        help='Buffer size for collecting multi-GPU data')
    parser.add_argument('--use_auth_token', action='store_true', default=True,
                        help='Use HuggingFace authentication token')
    
    args = parser.parse_args()
    
    eval_batch_size = args.batch_size
    # Initialize Accelerator
    accelerator = Accelerator()
    
    # Set random seed
    set_seed(42)
    
    # Load model
    accelerator.print(f"Loading model config: {args.config_path}")
    accelerator.print(f"Loading model checkpoint: {args.checkpoint_path}")
    model = load_ae_model(args.config_path, args.checkpoint_path, accelerator)
    
    # Create dataset and dataloader
    accelerator.print(f"Loading ImageNet validation dataset from HuggingFace")
    transform = create_transform(args.model_image_size)
    dataset = HuggingFaceImageNetDataset(transform=transform, use_auth_token=args.use_auth_token)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    accelerator.print(f"Dataset contains {len(dataset)} images")
    
    # Prepare model and dataloader with accelerator
    model, dataloader = accelerator.prepare(model, dataloader)
    
    # Initialize evaluator (only on main process)
    evaluator = None
    if accelerator.is_main_process:
        evaluator = Evaluator(
            device=accelerator.device,
            enable_rfid=args.enable_rfid,
            enable_inception_score=args.enable_inception_score,
            enable_codebook_usage_measure=False,
            enable_codebook_entropy_measure=False
        )
    
    # Start evaluation
    accelerator.print("Starting evaluation...")
    model.eval()
    
    # Collect data from all batches
    all_original_images = []
    all_reconstructed_images = []
    
    import tqdm

    # Wait for all processes to be ready
    accelerator.wait_for_everyone()

    with torch.no_grad():
        for batch_idx, (real_images, image_paths) in tqdm.tqdm(
            enumerate(dataloader), 
            total=len(dataloader),
            disable=not accelerator.is_local_main_process
        ):
            real_images = real_images.to(torch.bfloat16)

            # Model forward pass
            with torch.no_grad():
                model_output = model(real_images)
                if isinstance(model_output, tuple):
                    fake_images = model_output[0]
                elif isinstance(model_output, dict):
                    fake_images = model_output.get('reconstructed', model_output.get('reconstruction', model_output))
                else:
                    fake_images = model_output

            # Resize and normalize
            real_images = F.interpolate(real_images, size=(args.eval_image_size, args.eval_image_size), mode='bilinear').to(torch.float32)
            fake_images = F.interpolate(fake_images, size=(args.eval_image_size, args.eval_image_size), mode='bilinear').to(torch.float32)
            real_images = (real_images + 1.0) / 2.0

            mean = torch.tensor([0.485, 0.456, 0.406], device=accelerator.device)[None, :, None, None]
            std = torch.tensor([0.229, 0.224, 0.225], device=accelerator.device)[None, :, None, None]
            real_images = (real_images - mean) / std
            fake_images = torch.clamp(fake_images, -1.0, 1.0)

            all_original_images.append(real_images)
            all_reconstructed_images.append(fake_images)

            if len(all_original_images) >= args.buffer_size // args.batch_size:
                _process_buffer(
                    all_original_images, 
                    all_reconstructed_images, 
                    accelerator, 
                    evaluator, 
                    eval_batch_size
                )
                accelerator.wait_for_everyone()  # Synchronize all processes
                all_original_images, all_reconstructed_images = [], []

    # Process remaining data
    if len(all_original_images) > 0:
        _process_buffer(
            all_original_images, 
            all_reconstructed_images, 
            accelerator, 
            evaluator, 
            eval_batch_size
        )

    # Wait for all ranks to finish
    accelerator.wait_for_everyone()
    
    # Get evaluation results (only on main process)
    if accelerator.is_main_process and evaluator is not None:
        results = evaluator.result()
        
        # Print results to console
        accelerator.print("\nEvaluation Results:")
        accelerator.print("=" * 50)
        for metric, value in results.items():
            accelerator.print(f"{metric}: {value:.6f}")
        accelerator.print("=" * 50)

        # Save results to CSV file
        import os
        import csv

        # Build output path: same directory as checkpoint, filename is metrics.csv
        checkpoint_dir = os.path.dirname(args.checkpoint_path)
        metrics_file = os.path.join(checkpoint_dir, "metrics.csv")

        # Prepare one row of data: keys as header, values as data
        fieldnames = list(results.keys())
        row = {k: f"{v:.6f}" if isinstance(v, float) else str(v) for k, v in results.items()}

        # Write to CSV (create file and write header if it doesn't exist)
        file_exists = os.path.isfile(metrics_file)
        with open(metrics_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)


if __name__ == "__main__":
    main()
