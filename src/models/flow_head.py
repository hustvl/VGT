# Modified from https://github.com/stepfun-ai/NextStep-1
import math
from typing import Optional

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from mmengine.dist import is_main_process, get_rank
import torch.nn.functional as F
def modulate(x, shift, scale=None):
    if shift is None:
        return x * (1 + scale)
    return x * (1 + scale) + shift

def build_mlp(hidden_size, projector_dim, z_dim):
    """构建REPA投影头MLP"""
    return nn.Sequential(
        nn.Linear(hidden_size, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, z_dim),
    )
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element. These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq.to(self.mlp[0].weight.dtype))
        return t_emb

class ResBlock(nn.Module):
    def __init__(self, channels, mlp_ratio=1.0):
        super().__init__()
        self.channels = channels
        self.intermediate_size = int(channels * mlp_ratio)

        self.in_ln = nn.LayerNorm(self.channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.channels, self.intermediate_size),
            nn.SiLU(),
            nn.Linear(self.intermediate_size, self.channels),
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(channels, 3 * channels, bias=True))

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h

class FinalLayer(nn.Module):
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(model_channels, 2 * model_channels, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class SimpleMLPAdaLN(nn.Module):
    def __init__(self, input_dim, cond_dim, dim=1536, layers=12, mlp_ratio=1.0,
                 # REPA参数 - 简化版本
                 enable_repa=False, repa_encoder_depth=6, dinov3_feature_dim=768):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.dim = dim
        self.layers = layers
        self.mlp_ratio = mlp_ratio
        
        # REPA参数 - 简化版本
        self.enable_repa = enable_repa
        self.repa_encoder_depth = repa_encoder_depth
        self.dinov3_feature_dim = dinov3_feature_dim
        self.time_embed = TimestepEmbedder(dim)
        self.cond_embed = nn.Linear(cond_dim, dim)
        self.input_proj = nn.Linear(input_dim, dim)

        res_blocks = []
        for _ in range(layers):
            res_blocks.append(ResBlock(dim, mlp_ratio))
        self.res_blocks = nn.ModuleList(res_blocks)

        # REPA投影头（如果启用）
        if enable_repa:
            # 固定投影头配置：hidden_dim -> 2048 -> dinov3_feature_dim
            self.repa_projector = build_mlp(dim, 2048, dinov3_feature_dim)
        else:
            self.repa_projector = None
        self.final_layer = FinalLayer(dim, input_dim)

        self.grad_checkpointing = False

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)
        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    
    def forward(self, x, t, c):
        """
        x.shape = (bsz, input_dim)
        t.shape = (bsz,)
        c.shape = (bsz, cond_dim)
        
        Returns:
            output: (bsz, input_dim) 预测输出
            repa_features: List[(bsz, feature_dim)] REPA特征（如果启用）
        """

        x = self.input_proj(x)
        t = self.time_embed(t)
        c = self.cond_embed(c)

        y = t + c

        repa_features = None
        for i, block in enumerate(self.res_blocks):
            if self.grad_checkpointing and self.training:
                x = checkpoint(block, x, y, use_reentrant=True)
            else:
                x = block(x, y)
            
            # 在指定深度提取REPA特征
            if self.enable_repa and (i + 1) == self.repa_encoder_depth:
                repa_features = self.repa_projector(x)  # (bsz, dinov3_feature_dim)

        output = self.final_layer(x, y)
        
        if self.enable_repa and self.training:
            return output, repa_features
        else:
            return output

def expand_t(t, x):
    """Function to reshape time t to broadcastable dimension of x
    Args:
        t: [bsz,], time vector
        x: [bsz,...], data point
    """
    dims = [1] * (len(x.size()) - 1)
    t = t.view(t.size(0), *dims)
    return t

def randn_tensor(shape, noise_repeat, device):
    """Generate random tensor with noise repeat"""
    #这里输入shape是真实样本数量(bsz*seq, dim)，bsz*seq是一整个图像 token，直接全部都随机
    return torch.randn(shape, device=device)
# 新增：时间重分布函数（偏向 t≈0 噪声端）
def time_shift_fn(t: torch.Tensor, shift: float = 1.0) -> torch.Tensor:
    # 等价于 t' = t / (t + (1 - t) * shift)，shift>=1时偏向小t
    return t / (t + (1.0 - t) * shift)

def time_shift_forward(t: torch.Tensor, shift: float = 1.0) -> torch.Tensor:
    # shift >= 1 时偏向大 t
    return 1.0 - t / (t + (1.0 - t) * shift)

def time_shift_fn_rae(t: torch.Tensor, shift: float = 1.0) -> torch.Tensor:
    # 等价于 t' = t / (t + (1 - t) * shift)，shift>=1时偏向小t
    return 1 - shift * t / (1 + (shift - 1) * t)

# 新增：DDT Final 层（宽浅解码头的最终线性输出）
class DDTFinalLayer(nn.Module):
    def __init__(self, model_channels: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(model_channels, 2 * model_channels, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
# 新增：带 DDT 宽浅解码分支的 MLP-AdaLN
class SimpleMLPDDT(nn.Module):
    def __init__(
        self,
        input_dim,
        cond_dim,
        enc_dim=1536,
        enc_layers=12,
        dec_dim=2048,
        dec_layers=2,
        mlp_ratio=1.0,
        enable_repa=False,
        repa_encoder_depth=6,
        dinov3_feature_dim=768
    ):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.enc_dim = enc_dim
        self.enc_layers = enc_layers
        self.dec_dim = dec_dim
        self.dec_layers = dec_layers
        self.mlp_ratio = mlp_ratio

        self.enable_repa = enable_repa
        self.repa_encoder_depth = repa_encoder_depth
        self.dinov3_feature_dim = dinov3_feature_dim
        # 编码侧
        self.time_embed = TimestepEmbedder(enc_dim)
        self.cond_embed = nn.Linear(cond_dim, enc_dim)
        self.input_proj = nn.Linear(input_dim, enc_dim)
        self.enc_blocks = nn.ModuleList([ResBlock(enc_dim, mlp_ratio) for _ in range(enc_layers)])

        # REPA 投影（可选）
        if enable_repa:
            self.repa_projector = build_mlp(enc_dim, 2048, dinov3_feature_dim)
        else:
            self.repa_projector = None

        # 解码侧（宽浅头）
        self.s_projector = nn.Linear(enc_dim, dec_dim) if enc_dim != dec_dim else nn.Identity()
        self.dec_blocks = nn.ModuleList([ResBlock(dec_dim, mlp_ratio) for _ in range(dec_layers)])
        self.final_layer = DDTFinalLayer(dec_dim, input_dim)
        self.grad_checkpointing = False
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)
        # Zero-out 编码侧 AdaLN
        for block in self.enc_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out 解码侧最终层
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
    def forward(self, x, t, c):
        # 编码侧
        x = self.input_proj(x)           # [B, enc_dim]
        t_enc = self.time_embed(t)       # [B, enc_dim]
        c_enc = self.cond_embed(c)       # [B, enc_dim]
        y = t_enc + c_enc

        repa_features = None
        for i, block in enumerate(self.enc_blocks):
            if self.grad_checkpointing and self.training:
                x = checkpoint(block, x, y, use_reentrant=True)
            else:
                x = block(x, y)
            if self.enable_repa and (i + 1) == self.repa_encoder_depth:
                repa_features = self.repa_projector(x)

        # 解码侧：将编码输出与时间条件融合，并投到更宽的解码维度
        s = torch.nn.functional.silu(t_enc + x)   # [B, enc_dim]
        s = self.s_projector(s)                   # [B, dec_dim]

        # 修复点：使用前一层输出作为下一层输入，避免丢弃 dec_blocks[0] 的贡献
        x_dec = s
        for block in self.dec_blocks:
            x_dec = block(x_dec, s)

        # 最终输出
        output = self.final_layer(x_dec, s)
        if self.enable_repa and self.training:
            return output, repa_features
        else:
            return output

class FlowMatchingHead(nn.Module):
    def __init__(self, input_dim, cond_dim, dim=1536, layers=12, mlp_ratio=1.0, sample_t="logit-normal", timeshift=None,
                 # REPA参数 - 简化版本
                 enable_repa=False, repa_encoder_depth=6, dinov3_feature_dim=768):
        super(FlowMatchingHead, self).__init__()
        self.input_dim = input_dim
        self.enable_repa = enable_repa
        self.timeshift = timeshift 
        self.sample_t = sample_t #auto: timeshift = math.sqrt(shift_dim(16*16*32) / shift_base(fix:4096)) sanavae:2  unilipvit:7.4833
        if self.sample_t == "auto":
            print(f"[FlowMatchingHead]: sample t use auto, your timeshift:{timeshift} must be math.sqrt(shift_dim(16*16*32) / shift_base(fix:4096)) sanavae:2  unilipvit:7.4833")
        print(f"[FlowMatchingHead]: timeshift: {timeshift}")
        self.net = SimpleMLPAdaLN(
            input_dim=input_dim, 
            cond_dim=cond_dim, 
            dim=dim, 
            layers=layers, 
            mlp_ratio=mlp_ratio,
            enable_repa=enable_repa,
            repa_encoder_depth=repa_encoder_depth,
            dinov3_feature_dim=dinov3_feature_dim
        )

    @property
    def dtype(self):
        return self.net.input_proj.weight.dtype

    @property
    def device(self):
        return self.net.input_proj.weight.device

    @property
    def trainable_params(self) -> float:
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return LargeInt(n_params)

    
    def forward(
        self,
        target: torch.Tensor,
        c: torch.Tensor,
        mask: torch.Tensor | None = None,
        keep_dim: bool = False,
    ):
        """
        target.shape = (bsz, input_dim)
        c.shape      = (bsz, cond_dim)
        mask.shape   = (bsz,)
        keep_dim     = 是否保留非 batch 维度 (默认 False)
        """
        noise = torch.randn_like(target)

        if self.sample_t == "logit-normal":
            # sample t from logit-normal distribution
            u = torch.normal(mean=0.0, std=1.0, size=(len(target),))
            t = (1 / (1 + torch.exp(-u))).to(target)

            # 应用时间偏置
            if self.timeshift is not None:
                t = time_shift_fn_rae(t, self.timeshift)
        elif self.sample_t == "auto":
            t = torch.rand((target.shape[0]), device=target.device)
            t = time_shift_fn_rae(t, self.timeshift)
        else:
            assert False, "Invalid sample_t, must [logit-normal, auto]"

        # linear interpolation between target and noise
        xt = expand_t(t, target) * target + (1 - expand_t(t, target)) * noise
        ut = target - noise

        if self.enable_repa:
            model_output, repa_features = self.net(xt, t, c)
        else:
            model_output = self.net(xt, t, c)
            repa_features = None
            
        loss = (model_output.float() - ut.float()) ** 2
        if not keep_dim:
            # 默认行为：对所有非 batch 维度做平均
            loss = torch.mean(loss, dim=list(range(1, len(loss.size()))))

        if mask is not None:
            # 按 mask 加权平均
            loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        else:
            loss = loss.mean() if not keep_dim else loss

        if self.enable_repa:
            return loss, repa_features
        else:
            return loss, None

            
    def get_score_from_velocity(self, velocity, x, t):
        """Wrapper function: transfrom velocity prediction model to score
        Args:
            velocity: [bsz, ...] shaped tensor; velocity model output
            x:        [bsz, ...] shaped tensor; x_t data point
            t:        [bsz,] time tensor
        """
        t = expand_t(t, x)
        alpha_t, d_alpha_t = t, 1
        sigma_t, d_sigma_t = 1 - t, -1
        mean = x
        reverse_alpha_ratio = alpha_t / d_alpha_t
        var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
        score = (reverse_alpha_ratio * velocity - mean) / var
        return score

    def get_velocity_from_cfg(self, velocity, cfg, cfg2, cfg_mult):
        if cfg_mult == 2:
            cond_v, uncond_v = torch.chunk(velocity, 2, dim=0)
            velocity = uncond_v + cfg * (cond_v - uncond_v)
        elif cfg_mult == 3:
            cond_v, uncond_v1, uncond_v2 = torch.chunk(velocity, 3, dim=0)
            velocity = uncond_v2 + cfg2 * (uncond_v1 - uncond_v2) + cfg * (cond_v - uncond_v1)
        return velocity

    # @smart_compile(options={"triton.cudagraphs": True}, fullgraph=True)
    @torch.no_grad()
    def sample(
        self,
        c: torch.Tensor,
        cfg: float = 1.0,
        cfg2: float = 1.0,
        timesteps_shift: float = 1.0,
        num_sampling_steps: int = 20,
        last_step_size: float = 0.04,
        noise_repeat: int = 1,
        progress: bool = False,
        **kwargs
    ):
        """c.shape = (bsz, cond_dim)"""
        cfg_mult = 1
        if cfg > 1.0:
            cfg_mult += 1
        if cfg2 > 1.0:
            cfg_mult += 1

        noise = randn_tensor((c.shape[0] // cfg_mult, self.input_dim), noise_repeat, self.device)

        mean_x = noise
        x = noise
        xs = []
        t0, t1 = 0, 1 - last_step_size
        timesteps = torch.linspace(t0, t1, num_sampling_steps)
        timestep_pairs = zip(timesteps[:-1], timesteps[1:])
        if progress:
            from tqdm.auto import tqdm
            # 计算总迭代次数（timesteps长度减1）
            total = len(timesteps) - 1 if hasattr(timesteps, '__len__') else None
            timestep_pairs = tqdm(timestep_pairs, total=total, desc="Processing flow matching")
            
        for ti, tj in timestep_pairs:
            dt = tj - ti

            combined = torch.cat([x] * cfg_mult, dim=0)
            velocity = self.net(combined.to(c.dtype), ti.expand(c.shape[0]).to(c), c)
            velocity = velocity.to(torch.float32)

            velocity = self.get_velocity_from_cfg(velocity, cfg, cfg2, cfg_mult)
            score = self.get_score_from_velocity(velocity, x, ti.expand(x.shape[0]).to(x))
            drift = velocity + (1 - expand_t(ti.expand(x.shape[0]).to(x), x)) * score
            w_cur = randn_tensor((c.shape[0] // cfg_mult, self.input_dim), noise_repeat, self.device)
            dw = w_cur * torch.sqrt(dt)

            mean_x = x + drift * dt
            x = mean_x + torch.sqrt(2 * (1 - expand_t(ti.expand(x.shape[0]).to(x), x))) * dw
            xs.append(x)

        combined = torch.cat([xs[-1]] * cfg_mult, dim=0)
        velocity = self.net(combined.to(c.dtype), timesteps[-1].expand(c.shape[0]).to(c), c)
        velocity = velocity.to(torch.float32)

        velocity = self.get_velocity_from_cfg(velocity, cfg, cfg2, cfg_mult)
        x = xs[-1] + velocity * last_step_size
        xs.append(x)

        if len(xs) != num_sampling_steps:
            raise ValueError(f"Samples ({len(xs)}) does not match the number of steps ({num_sampling_steps})")

        return xs[-1].to(c.dtype)
    # @smart_compile(options={"triton.cudagraphs": True}, fullgraph=True)
    @torch.no_grad()
    def sample_new(
        self,
        c: torch.Tensor,
        cfg: float = 1.0,
        cfg2: float = 1.0,
        timesteps_shift: float = 1.0,
        num_sampling_steps: int = 20,
        last_step_size: float = 0.04,
        noise_repeat: int = 1,
    ):
        z = c
        cfg_mult = 1
        if cfg > 1.0:
            cfg_mult += 1
        if cfg2 > 1.0:
            cfg_mult += 1

        bsz = z.shape[0]
        device, dtype = z.device, z.dtype

        noise = torch.randn(bsz // cfg_mult, self.input_dim, device=device, dtype=dtype)

        noise = torch.cat([noise] * cfg_mult, dim=0)

        mean_x = noise
        x = noise
        xs = []

        sigmas = torch.linspace(0, 1, num_sampling_steps + 1, device=device, dtype=dtype)[:-1]
        # 修改：采样 sigma 网格用与训练一致的 time_shift_fn
        if timesteps_shift is not None and timesteps_shift > 1.0:
            sigmas = time_shift_fn(sigmas, timesteps_shift)

        sigmas = torch.cat([sigmas, torch.ones(1, device=sigmas.device, dtype=dtype)])

        timesteps = sigmas
        for step, (ti, tj) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            dt = tj - ti
            w_cur = torch.randn(x.size(), device=device, dtype=dtype)
            dw = w_cur * torch.sqrt(dt)

            split_x = x[: len(x) // cfg_mult]
            combined = torch.cat([split_x] * cfg_mult, dim=0)
            t = torch.ones(x.size(0)).to(x) * ti
            model_output = self.net(combined, t, z)
            eps, rest = model_output[:, : self.input_dim], model_output[:, self.input_dim :]
            if cfg_mult == 2:
                cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
                half_eps = uncond_eps + cfg * (cond_eps - uncond_eps)
                # eps = torch.cat([half_eps, half_eps], dim=0)
                eps = torch.cat([half_eps, uncond_eps], dim=0)

            elif cfg_mult == 3:
                cond_eps, uncond_eps1, uncond_eps2 = torch.split(eps, len(eps) // 3, dim=0)
                third_eps = uncond_eps2 + cfg2 * (uncond_eps1 - uncond_eps2) + cfg * (cond_eps - uncond_eps1)
                eps = torch.cat([third_eps, third_eps, third_eps], dim=0)
            velocity = torch.cat([eps, rest], dim=1)
            score = self.get_score_from_velocity(velocity, x, t)
            drift = velocity + (1 - expand_t(t, x)) * score

            mean_x = x + drift * dt
            x = mean_x + torch.sqrt(2 * (1 - expand_t(t, x))) * dw
            xs.append(x)

        if len(xs) != num_sampling_steps:
            raise ValueError(f"Samples ({len(xs)}) does not match the number of steps ({num_sampling_steps})")
        return xs[-1][:1]
