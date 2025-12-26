from mmengine.config import read_base
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR, OptimWrapper, MultiStepLR
from xtuner.engine.runner import TrainLoop
from src.optimisers.custom_adamw import CustomAdamW
from src.hooks.swandb_hook import SwanDBHook
from src.hooks.ema_hook import EMAHook_fix
from mmengine.hooks import EMAHook
with read_base():
    from ..models.vgt_internvl3_0_6B_448px import model
    from ..datasets.internvl3_1b_448.fintune_images import train_dataloader

train_dataloader.batch_size = 32
model.use_activation_checkpointing = False # If the GPU memory is insufficient, turn it on
model.repa_loss_weight = 0.5

# Scheduler & Optimizer
accumulative_counts = 1
dataloader_num_workers = 4
max_iters = 5000
ema_begin_iter = max_iters
optim_type = CustomAdamW
lr = 1e-4
betas = (0.9, 0.95)
weight_decay = 0.05
max_norm = 1.0
warmup_ratio = 0.01

# Save
save_steps = 2000
save_total_limit = -1  # Maximum checkpoints to keep (-1 means unlimited)


optim_wrapper = dict(
    type=OptimWrapper, 
    optimizer=dict(type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
)

param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=False,
        begin=0,
        end=warmup_ratio * max_iters),
    dict(
        type=CosineAnnealingLR,
        eta_min=5e-5,
        by_epoch=False,
        begin=warmup_ratio * max_iters,
        end=max_iters)
]
# train, val, test setting
train_cfg = dict(type=TrainLoop, max_iters=max_iters)

t2i_tasks = [
dict(
        task_type="T2I",
        prompt=[
            "A stack of pancakes with a glossy, caramel-like syrup on top is presented. The pancakes are golden brown and appear to be freshly cooked. The syrup is dripping down the sides of the pancakes, creating a shiny, reflective surface. The image is captioned 'SILVER DOLLAR PANCAKES GLUTEN FREE & LOW CARB' and includes a watermark at the bottom right corner.",
            "Aerial view of a lush green mountainous landscape with a winding path leading through the hills. The terrain is covered with dense vegetation, and the sky is partly cloudy.",
            "A close-up shot of a young white snow fox with alert ears and intelligent eyes, standing amid soft falling snow.",
            "The image features a close-up of a dog with a brown coat and a white patch on its chest, looking to the left against a warm brown background.",
            "A young East Asian girl with braided hair and a floral shirt poses against a plain white background, looking directly at the camera with a slight smile.",
            "A young woman with her hair in a bun sits in a boat, wearing a striped off-the-shoulder dress, with a blurred background of trees and water.",
            "A fair-skinned woman with blonde hair and a white orchid in her hair is looking at the camera with a slight smile against a white background.",
            "A close-up shot features a young woman with long, ombre hair and blue eyes, wearing a dark gray shirt against a muted gray background.",
            ],
        cfg_scale=3.5,
        num_steps=30,
        temperature=1.0,
        height=448,
        width=448,
        total_step= 256,
    ),
dict(
    task_type="T2I",
    prompt=[
            "A stack of pancakes with a glossy, caramel-like syrup on top is presented. The pancakes are golden brown and appear to be freshly cooked. The syrup is dripping down the sides of the pancakes, creating a shiny, reflective surface. The image is captioned 'SILVER DOLLAR PANCAKES GLUTEN FREE & LOW CARB' and includes a watermark at the bottom right corner.",
            "Aerial view of a lush green mountainous landscape with a winding path leading through the hills. The terrain is covered with dense vegetation, and the sky is partly cloudy.",
            "A close-up shot of a young white snow fox with alert ears and intelligent eyes, standing amid soft falling snow.",
            "The image features a close-up of a dog with a brown coat and a white patch on its chest, looking to the left against a warm brown background.",
            "A young East Asian girl with braided hair and a floral shirt poses against a plain white background, looking directly at the camera with a slight smile.",
            "A young woman with her hair in a bun sits in a boat, wearing a striped off-the-shoulder dress, with a blurred background of trees and water.",
            "A fair-skinned woman with blonde hair and a white orchid in her hair is looking at the camera with a slight smile against a white background.",
            "A close-up shot features a young woman with long, ombre hair and blue eyes, wearing a dark gray shirt against a muted gray background.",
            ],
    cfg_scale=1.0,
    num_steps=30,
    temperature=1.0,
    height=448,
    width=448,
    total_step= 256,
),
dict(
    task_type="T2I",
    prompt=[
        "A cute cat sitting in a garden",
        "A young woman reading a book on a wooden bench in a park",
        "Abstract art with colorful patterns",
        "A peaceful forest scene with sunlight",
        "A futuristic robot in a laboratory"
    ],
    cfg_scale=3.5,
    num_steps=30,
    temperature=1.0,
    height=448,
    width=448,
    total_step= 256,
),
dict(
    task_type="T2I",
    prompt=[
        "A young East Asian girl with braided hair and a floral shirt poses against a plain white background, looking directly at the camera with a slight smile.",
    ],
    cfg_scale=6.5,
    num_steps=30,
    temperature=1.0,
    height=448,
    width=448,
    total_step= 256,
),
]

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit),
    ema=dict(
        type=EMAHook_fix,
        momentum=0.0002,
        begin_iter = ema_begin_iter,
        update_buffers=True,
        priority=49,
    ),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
    # wandb visualization hook
        wandb=dict(
        type=SwanDBHook,
        api_key='xxxxxxx',
        project='vgt-training',
        experiment_name='vgt_internvl3_0_6B_448px_sft',
        log_interval=100,
        sample_interval=2000,
        sample_tasks=t2i_tasks,
    ),
)
# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
visualizer = None

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = "./work_dirs/vgt_internvl3_0_6B_448px_pretrain/iter_100000.pth"

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(by_epoch=False)

