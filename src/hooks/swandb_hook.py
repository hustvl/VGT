"""
Simplified SwanDB Hook for VGT Training
Only keeps core text-to-image visualization and logging functionality
"""

import os
import torch
import swanlab
from typing import Dict, Any, Optional
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.registry import HOOKS
from mmengine.dist import is_main_process, get_rank, get_world_size
from mmengine.logging import print_log
from PIL import Image
from einops import rearrange
from datetime import datetime
import math
from src.utils import _generate_text_to_image

def _create_image_grid(images: list, grid_size: int = None) -> Image.Image:
    """
    Create a grid of images
    
    Args:
        images: List of PIL Images (all same size)
        grid_size: Number of images per row (auto-calculate if None)
    
    Returns:
        Grid image as PIL Image
    """
    if not images:
        raise ValueError("No images to create grid")
    
    num_images = len(images)
    
    # Auto-calculate grid size if not provided
    if grid_size is None:
        grid_size = math.ceil(math.sqrt(num_images))
    
    # Calculate grid dimensions
    cols = grid_size
    rows = math.ceil(num_images / cols)
    
    # Get single image size (assume all images are same size)
    img_width, img_height = images[0].size
    
    # Create blank canvas
    grid_width = cols * img_width
    grid_height = rows * img_height
    grid_image = Image.new('RGB', (grid_width, grid_height), color=(255, 255, 255))
    
    # Paste images into grid
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = col * img_width
        y = row * img_height
        grid_image.paste(img, (x, y))
    
    return grid_image


@HOOKS.register_module()
class SwanDBHook(Hook):
    """
    Simplified SwanDB Hook for VGT Training
    
    Features:
    1. Training loss and metrics logging
    2. Text-to-image visualization at specified intervals
    3. Only runs on main process (rank 0)
    """
    
    def __init__(self,
                 api_key: str,
                 project: str = "vgt-training",
                 experiment_name: Optional[str] = None,
                 log_interval: int = 100,
                 sample_interval: int = 1000,
                 sample_tasks: list = None,
                 host: Optional[str] = None):
        """
        Args:
            api_key: SwanDB API key
            project: SwanDB project name
            experiment_name: Experiment name (optional)
            log_interval: Logging interval (iterations)
            sample_interval: Sampling visualization interval (iterations)
            sample_tasks: List of text-to-image tasks for visualization
            host: SwanDB server address (optional)
        """
        self.api_key = api_key
        self.project = project
        self.experiment_name = experiment_name
        self.log_interval = log_interval
        self.sample_interval = sample_interval
        self.host = host
        
        # Default sample tasks
        if sample_tasks is None:
            self.sample_tasks = [
                {
                    "task_type": "T2I",
                    "prompt": ["A beautiful landscape with mountains and rivers"],
                    "cfg_scale": 3.5,
                    "num_steps": 50,
                    "temperature": 1.0,
                    "height": 512,
                    "width": 512,
                    "grid_size": 1,
                }
            ]
        else:
            self.sample_tasks = sample_tasks
        
        self.swandb_run = None
        self.is_initialized = False
        
    def before_run(self, runner: Runner):
        """Initialize SwanDB - only on main process"""
        if not is_main_process():
            return
            
        try:
            # Login to SwanDB
            if self.host:
                swanlab.login(api_key=self.api_key, host=self.host)
            else:
                swanlab.login(api_key=self.api_key)
            
            # Prepare config
            config = {
                'model_architecture': 'VGT-InternVL3',
                'max_iters': getattr(runner.train_loop, 'max_iters', None),
                'learning_rate': getattr(runner.optim_wrapper.optimizer, 'param_groups', [{}])[0].get('lr', None),
                'batch_size': getattr(runner.train_dataloader, 'batch_size', None),
                'world_size': get_world_size(),
                'sample_tasks_count': len(self.sample_tasks),
            }
            
            # Initialize SwanDB
            self.swandb_run = swanlab.init(
                project=self.project,
                experiment_name=self.experiment_name,
                config=config
            )
            
            self.is_initialized = True
            runner.logger.info(f"SwanDB initialized (rank 0/{get_world_size()})")
            
        except Exception as e:
            runner.logger.error(f"SwanDB initialization failed: {e}")
            self.is_initialized = False
    
    def after_train_iter(self, 
                        runner: Runner,
                        batch_idx: int,
                        data_batch: Any = None,
                        outputs: Optional[dict] = None):
        """Hook after training iteration - only on main process"""
        if not is_main_process():
            return
        
        current_iter = runner.iter
        
        # Log training metrics (only if SwanDB is initialized)
        if self.is_initialized and current_iter % self.log_interval == 0:
            self._log_training_metrics(runner, outputs)
        
        # Sample visualization (always save locally, upload to SwanDB if initialized)
        if current_iter % self.sample_interval == 0 and current_iter > 0:
            self._log_sample_visualization(runner, current_iter)

    def _log_training_metrics(self, runner: Runner, outputs: Optional[dict]):
        """Log training metrics - only on main process"""
        if not is_main_process():
            return
            
        try:
            log_dict = {
                'iteration': runner.iter,
            }
            
            # Log losses
            if outputs is not None:
                for key, value in outputs.items():
                    if 'loss' in key.lower():
                        if torch.is_tensor(value):
                            log_dict[key] = value.item()
                        else:
                            log_dict[key] = value
            
            # Log learning rate
            if hasattr(runner.optim_wrapper, 'optimizer'):
                for i, param_group in enumerate(runner.optim_wrapper.optimizer.param_groups):
                    log_dict[f'lr_group_{i}'] = param_group['lr']
            
            # Log GPU memory usage
            if torch.cuda.is_available():
                log_dict['gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1024**3  # GB
                log_dict['gpu_memory_reserved'] = torch.cuda.memory_reserved() / 1024**3  # GB
            
            swanlab.log(log_dict, step=runner.iter)
            
        except Exception as e:
            runner.logger.error(f"Failed to log training metrics: {e}")
    
    def _save_images_locally(self, runner: Runner, images: list, task_info: list, current_iter: int):
        """Save generated images to local directory with timestamp naming"""
        try:
            # Create save directory
            work_dir = runner.work_dir
            if not work_dir:
                runner.logger.warning("No work_dir found, skipping local image save")
                return
            
            save_dir = os.path.join(work_dir, "generated_images", f"iter_{current_iter:06d}")
            os.makedirs(save_dir, exist_ok=True)
            
            # Get current timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save each image
            for idx, (img, info) in enumerate(zip(images, task_info)):
                # Create filename with timestamp
                filename = f"{timestamp}_sample_{idx:03d}.png"
                save_path = os.path.join(save_dir, filename)
                
                # Save image
                img.save(save_path)
            
            runner.logger.info(f"Saved {len(images)} images to {save_dir}")
            
        except Exception as e:
            runner.logger.error(f"Failed to save images locally: {e}")
    
    def _log_sample_visualization(self, runner: Runner, current_iter: int):
        """Log sample visualization - text-to-image generation"""
        if not is_main_process():
            return
            
        try:
            runner.logger.info(f"Starting T2I visualization (iter: {current_iter})")
            
            # Get model
            model = runner.model
            actual_model = model.module if hasattr(model, 'module') else model
            
            # Swap to EMA parameters if available
            ema_hook = next((h for h in runner.hooks if "EMAHook" in h.__class__.__name__), None)
            if ema_hook is not None:
                ema_hook._swap_ema_parameters()
            
            # Disable gradient checkpointing for visualization
            was_gradient_checkpointing = False
            if hasattr(actual_model, 'use_activation_checkpointing'):
                was_gradient_checkpointing = actual_model.use_activation_checkpointing
                if was_gradient_checkpointing:
                    actual_model.gradient_checkpointing_disable()
            
            # Set model to eval mode
            was_training = model.training
            model.eval()
            
            # Generate images for each task
            all_generated_images = []
            all_task_info = []
            
            for task_idx, task in enumerate(self.sample_tasks):
                if task.get('task_type') != 'T2I':
                    continue
                    
                runner.logger.info(f"Executing task {task_idx + 1}/{len(self.sample_tasks)}: T2I")
                
                # Extract prompts and prepare task parameters
                prompts = task.get('prompt', [])
                if not isinstance(prompts, list):
                    prompts = [prompts]
                
                # Prepare generation parameters
                gen_params = {
                    'prompts': prompts,
                    'cfg_scale': task.get('cfg_scale', 3.5),
                    'num_steps': task.get('num_steps', 50),
                    'height': task.get('height', 512),
                    'width': task.get('width', 512),
                    'temperature': task.get('temperature', 1.0),
                    'grid_size': task.get('grid_size', 1),
                }
                
                # Add any additional parameters
                for key in ['scheduler_type', 'total_step', 'acc_ratio']:
                    if key in task:
                        gen_params[key] = task[key]
                
                # Generate images
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
                    generated_images = _generate_text_to_image(actual_model, **gen_params)
                
                # Create grid image for this task
                grid_image = _create_image_grid(generated_images)
                
                # Create task info
                task_info = (f"Task{task_idx+1}_T2I | "
                           f"Prompts: {len(prompts)} | "
                           f"CFG: {gen_params['cfg_scale']} | "
                           f"Steps: {gen_params['num_steps']} | "
                           f"Size: {gen_params['width']}x{gen_params['height']}")
                
                all_generated_images.append(grid_image)
                all_task_info.append(task_info)
            
            # Log to SwanDB and save locally
            if all_generated_images:
                # Always save to local directory
                self._save_images_locally(runner, all_generated_images, all_task_info, current_iter)
                
                # Upload to SwanDB only if initialized
                if self.is_initialized:
                    swandb_images = []
                    for img, info in zip(all_generated_images, all_task_info):
                        swandb_img = swanlab.Image(img, caption=info)
                        swandb_images.append(swandb_img)
                    
                    swanlab.log({
                        "text2image_samples": swandb_images
                    }, step=current_iter)
                    
                    runner.logger.info(f"T2I visualization completed, generated {len(swandb_images)} images (uploaded to SwanDB)")
                else:
                    runner.logger.info(f"T2I visualization completed, generated {len(all_generated_images)} images (saved locally only)")
            
            # Restore model state
            if was_training:
                model.train()
            if was_gradient_checkpointing:
                actual_model.gradient_checkpointing_enable()
            if ema_hook is not None:
                ema_hook._swap_ema_parameters()
                
        except Exception as e:
            runner.logger.error(f"Sample visualization failed: {e}")
            import traceback
            runner.logger.error(traceback.format_exc())
    
    def after_run(self, runner: Runner):
        """Cleanup after training - only on main process"""
        if not is_main_process() or not self.is_initialized:
            return
            
        try:
            if self.swandb_run is not None:
                swanlab.finish()
                runner.logger.info("SwanDB run finished")
        except Exception as e:
            runner.logger.error(f"Failed to finish SwanDB run: {e}")
