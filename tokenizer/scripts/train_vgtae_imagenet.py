"""Training script for TiTok.

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

Reference:
    https://github.com/huggingface/open-muse
"""
import math
import os
import sys
from pathlib import Path

from accelerate.utils import set_seed
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs

import torch
from omegaconf import OmegaConf

from utils.logger import setup_logger
from utils.train_vgtae_imagenet_utils import (
    get_config, 
    create_model_and_loss_module,
    create_optimizer, create_lr_scheduler, create_dataloader,
    create_evaluator, auto_resume, save_checkpoint, 
    train_one_epoch)


def main():
    torch.hub.set_dir("./models/hub")

    config = get_config()
    # Enable TF32 on Ampere GPUs.
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    output_dir = config.experiment.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Whether logging to Wandb or Tensorboard.
    tracker = "tensorboard"
    if config.training.enable_wandb:
        tracker = "wandb"
        os.environ['WANDB_API_KEY'] = config.training.wandb_api_key
        os.environ['WANDB_PROJECT'] = config.experiment.project
        os.environ['WANDB_DIR'] = config.experiment.output_dir
        os.environ['WANDB_NAME'] = config.experiment.name

    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with=tracker,
        project_dir=output_dir,
        split_batches=False,
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=config.training.ddp_find_unused_parameters)
        ]
    )

    logger = setup_logger(
        name=config.experiment.project, 
        log_level="INFO",
        output_file=f"{output_dir}/logger_rank_{accelerator.process_index}.txt",
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(config.experiment.project)
        config_path = Path(output_dir) / "config.yaml"
        logger.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)
        logger.info(f"Config:\n{OmegaConf.to_yaml(config)}")

    # If passed along, set the training seed now.
    if config.training.seed is not None:
        set_seed(config.training.seed, device_specific=True)
        
    accelerator.wait_for_everyone()

    model, ema_model, loss_module = create_model_and_loss_module(
        config, logger, accelerator, model_type=config.model.name)

    optimizer, discriminator_optimizer = create_optimizer(config, logger, model, loss_module, model_type=config.model.name)

    lr_scheduler, discriminator_lr_scheduler = create_lr_scheduler(
        config, logger, accelerator, optimizer, discriminator_optimizer)

    train_dataloader, eval_dataloader = create_dataloader(config, logger, accelerator)

    # Set up evaluator.
    evaluator = create_evaluator(config, logger, accelerator)

    # Prepare everything with accelerator.
    logger.info("Preparing model, optimizer, scheduler, and dataloaders")
    # The dataloader are already aware of distributed training, so we don't need to prepare them.
    (
        model, 
        loss_module, 
        optimizer, 
        discriminator_optimizer, 
        lr_scheduler, 
        discriminator_lr_scheduler,
        train_dataloader,
        eval_dataloader,
    ) = accelerator.prepare(
        model, 
        loss_module, 
        optimizer, 
        discriminator_optimizer, 
        lr_scheduler, 
        discriminator_lr_scheduler,
        train_dataloader,
        eval_dataloader,
    )
    if config.training.use_ema:
        ema_model.to(accelerator.device)

    # if accelerator.is_main_process:
    #     for name, param in accelerator.unwrap_model(model).named_parameters():
    #         print(name, param.requires_grad, param.device)
    # import pdb; pdb.set_trace()
    # total_batch_size_without_accum = config.training.per_gpu_batch_size * accelerator.num_processes
    # num_batches = math.ceil(
    #     config.experiment.max_train_examples / total_batch_size_without_accum)
    # num_update_steps_per_epoch = math.ceil(num_batches / config.training.gradient_accumulation_steps)
    # num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)

    total_batch_size_without_accum = config.training.per_gpu_batch_size * accelerator.num_processes
    num_batches = math.ceil(
        config.experiment.max_train_examples / total_batch_size_without_accum)
    num_update_steps_per_epoch = math.ceil(num_batches / config.training.gradient_accumulation_steps)
    num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)

    # Start training.
    logger.info("***** Running training *****")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")
    logger.info(f"  Instantaneous batch size per gpu = { config.training.per_gpu_batch_size}")
    logger.info(f"""  Total train batch size (w. parallel, distributed & accumulation) = {(
        config.training.per_gpu_batch_size *
        accelerator.num_processes *
        config.training.gradient_accumulation_steps)}""")
    global_step = 0
    first_epoch = 0

    global_step, first_epoch = auto_resume(
        config, logger, accelerator, ema_model, num_update_steps_per_epoch,
        strict=True)

    for current_epoch in range(first_epoch, num_train_epochs):
        accelerator.print(f"Epoch {current_epoch}/{num_train_epochs-1} started.")
        global_step = train_one_epoch(config, logger, accelerator,
                            model, ema_model, loss_module,
                            optimizer, discriminator_optimizer,
                            lr_scheduler, discriminator_lr_scheduler,
                            train_dataloader, eval_dataloader,
                            evaluator,
                            global_step)
        # Stop training if max steps is reached.
        if global_step >= config.training.max_train_steps:
            accelerator.print(
                f"Finishing training: Global step is >= Max train steps: {global_step} >= {config.training.max_train_steps}"
            )
            break

    accelerator.wait_for_everyone()
    # Save checkpoint at the end of training.
    save_checkpoint(model, output_dir, accelerator, global_step, logger=logger)
    # Save the final trained checkpoint
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        if config.training.use_ema:
            ema_model.copy_to(model.parameters())
        model.save_pretrained_weight(output_dir)
    accelerator.end_training()


if __name__ == "__main__":
    main()