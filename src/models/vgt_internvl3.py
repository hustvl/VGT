"""
VGT-InternVL3: Visual Generation Transformer with InternVL3 backbone
Inference-only implementation for multimodal image generation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from torch.nn.modules.module import T
from torch.cuda.amp import autocast
from transformers.cache_utils import DynamicCache
from einops import rearrange

from xtuner.registry import BUILDER
from mmengine.model import BaseModel
from mmengine.logging import print_log
from mmengine.dist import is_main_process
from mmengine.runner.checkpoint import load_checkpoint

from src.models.vgtae_internvl3 import VGTAE_InternVL3
from src.models.generation_scheduler import get_generation_scheduler
from src.utils import pad_input_ids

def layer_norm_2d(input: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Apply 2D layer normalization.
    
    Args:
        input: Input tensor with shape (batch_size, channels, height, width)
        eps: Small value for numerical stability
        
    Returns:
        Normalized tensor with same shape as input
    """
    _input = input.permute(0, 2, 3, 1)
    _input = F.layer_norm(_input, _input.size()[-1:], None, None, eps)
    _input = _input.permute(0, 3, 1, 2)
    return _input


def layer_norm(input: torch.Tensor, normalized_shape: torch.Size, eps: float = 1e-6) -> torch.Tensor:
    """
    Apply layer normalization.
    
    Args:
        input: Input tensor
        normalized_shape: Shape of normalized axes
        eps: Small value for numerical stability
        
    Returns:
        Normalized tensor
    """
    return F.layer_norm(input, normalized_shape, None, None, eps)


class VGT_InternVL3(BaseModel):
    """
    VGT-InternVL3: Visual Generation with InternVL3 backbone.
    """
    
    def __init__(
        self,
        lmm: Dict,
        tokenizer: Dict,
        prompt_template: Dict,
        image_head: Dict,
        vgt_ae: Dict,
        adaptive_metaquery: Dict,
        max_length: int = 2048,
        latent_patch_size: int = 1,
        **kwargs
    ):
        """
        Initialize VGT-InternVL3 model.
        
        Args:
            lmm: Language model configuration
            tokenizer: Tokenizer configuration
            prompt_template: Prompt template configuration
            image_head: Image head configuration
            vgt_ae: VAE wrapper configuration
            adaptive_metaquery: Adaptive metaquery configuration
            max_length: Maximum sequence length
            latent_patch_size: Patch size for latent representation
        """
        super().__init__()
        self.max_length = max_length
        self.latent_patch_size = latent_patch_size

        # Initialize MLLM (InternVL3)
        self.lmm = BUILDER.build(lmm)
        self.lmm.requires_grad_(False)
        
        self.vgt_ae = VGTAE_InternVL3(**vgt_ae)
        
        # Adaptive MetaQuery system
        adaptive_metaquery.update(
            mllm_embed_dim=self.llm.config.hidden_size
        )
        self.adaptive_metaquery = BUILDER.build(adaptive_metaquery)
        
        token_dim = self.vgt_ae.latent_dim * self.latent_patch_size**2
        
        # Image input projector
        self.image_in_projector = nn.Linear(token_dim, self.llm.config.hidden_size)
        self.image_in_projector.weight.data.normal_(mean=0.0, std=0.02)
        self.image_in_projector.bias.data.zero_()
        
        # Image output projector
        self.image_out_projector = nn.Linear(
            self.llm.config.hidden_size, 
            self.llm.config.hidden_size
        )
        self.image_out_projector.weight.data.normal_(mean=0.0, std=0.02)
        self.image_out_projector.bias.data.zero_()
        
        # Flow Matching Head
        if image_head is not None:
            image_head.update(
                input_dim=token_dim, 
                cond_dim=self.llm.config.hidden_size,
            )
            self.image_head = BUILDER.build(image_head)
        else:
            raise ValueError("image_head config is required")

        # Tokenizer and prompt related
        if tokenizer is not None:
            self.tokenizer = BUILDER.build(tokenizer)
        else:
            raise ValueError("tokenizer config is required")
            
        if prompt_template is not None:
            self.prompt_template = prompt_template
            self.image_token_id = self.tokenizer.convert_tokens_to_ids(
                prompt_template['IMG_CONTEXT_TOKEN']
            )
        else:
            raise ValueError("prompt_template config is required")
        
        self.to(dtype=self.dtype)
        
    @property
    def llm(self):
        """Get language model."""
        return self.lmm.language_model

    @property
    def device(self) -> torch.device:
        """Get model device."""
        return self.llm.device

    @property
    def dtype(self) -> torch.dtype:
        """Get model data type."""
        return self.llm.dtype

    def train(self: T, mode: bool = True) -> T:
        """
        Set training mode.
        
        Args:
            mode: Whether to set training mode
            
        Returns:
            Model instance
        """
        super().train(mode=mode)
        self.vgt_ae.train(mode=False)
        return self

    @torch.no_grad()
    def pixels_to_latents(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode pixels to latents.
        
        Args:
            x: Input image tensor
            
        Returns:
            Encoded latent tensor
        """
        z = self.vgt_ae.encode(x)
        assert len(z.shape) == 4, f"Invalid shape: {z.shape}"
        return z

    @torch.no_grad()
    def latents_to_pixels(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latents to pixels.
        
        Args:
            z: Input latent tensor
            
        Returns:
            Decoded image tensor
        """
        x_rec = self.vgt_ae.decode(z)
        assert len(x_rec.shape) == 4, f"Invalid shape: {x_rec.shape}"
        return x_rec

    def prepare_forward_input(
        self,
        queries: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare forward input for the model.
        
        Args:
            queries: Query embeddings
            inputs_embeds: Input embeddings
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Dictionary of prepared inputs
        """
        # Get batch size
        batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        b, l, _ = queries.shape
        assert l > 0
        
        attention_mask = attention_mask.to(device=self.device, dtype=torch.bool)
        attention_mask = torch.cat([
            attention_mask, attention_mask.new_ones(b, l)
        ], dim=1)
        position_ids = torch.cumsum(attention_mask, dim=1) - 1
        position_ids[position_ids < 0] = 0

        # Prepare context
        if inputs_embeds is None:
            input_ids = input_ids.to(self.device)
            inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([inputs_embeds, queries], dim=1)
        inputs_embeds = inputs_embeds.to(dtype=self.dtype)

        inputs = dict(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids
        )

        return inputs

    def patchify(self, img: torch.Tensor, patch_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Convert image to patches.
        
        Args:
            img: Input image tensor with shape (batch_size, channels, height, width)
            patch_size: Size of each patch
            
        Returns:
            Tuple of (patched tensor, (height_patches, width_patches))
        """
        bsz, c, h, w = img.shape
        p = patch_size
        h_, w_ = h // p, w // p

        img = img.reshape(bsz, c, h_, p, w_, p)
        img = torch.einsum("nchpwq->nhwcpq", img)
        x = img.reshape(bsz, h_ * w_, c * p**2)
        return x, (h_, w_)

    def unpatchify(
        self, 
        x: torch.Tensor, 
        patch_size: int, 
        h: Optional[int] = None, 
        w: Optional[int] = None
    ) -> torch.Tensor:
        """
        Convert patches back to image.
        
        Args:
            x: Patched tensor with shape (batch_size, num_patches, patch_dim)
            patch_size: Size of each patch
            h: Height in patches (optional)
            w: Width in patches (optional)
            
        Returns:
            Reconstructed image tensor
        """
        bsz = x.shape[0]
        p = patch_size
        c = self.vgt_ae.latent_dim
        if h is None and w is None:
            h_ = w_ = int(x.shape[1] ** 0.5)
        else:
            h_, w_ = h, w
        assert h_ * w_ == x.shape[1], f"Invalid sequence length {x.shape[1]}."

        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum("nhwcpq->nchpwq", x)
        img = x.reshape(bsz, c, h_ * p, w_ * p)
        return img

    @torch.no_grad()
    def prepare_text_conditions(
        self, 
        prompt: str, 
        cfg_prompt: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare text conditions for inference.
        
        Args:
            prompt: Input prompt text
            cfg_prompt: CFG prompt text (optional)
            
        Returns:
            Dictionary containing input_ids and attention_mask
        """
        if cfg_prompt is None:
            cfg_prompt = self.prompt_template['CFG']
        else:
            cfg_prompt = self.prompt_template['GENERATION'].format(input=cfg_prompt.strip())
        prompt = self.prompt_template['GENERATION'].format(input=prompt.strip())
        
        all_prompts = [
            self.prompt_template['INSTRUCTION'].format(input=prompt),
            self.prompt_template['INSTRUCTION'].format(input=cfg_prompt),
        ]

        input_ids = [
            self.tokenizer.encode(p, add_special_tokens=True, return_tensors='pt')[0]
            for p in all_prompts
        ]

        input_ids, attention_mask = pad_input_ids(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
            padding_mode="right"
        )

        return dict(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device)
        )

    @torch.no_grad()
    def prepare_batch_text_conditions(
        self, 
        prompts: List[str], 
        cfg_prompts: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare batch text conditions for multiple prompts.
        
        Args:
            prompts: List of text prompts
            cfg_prompts: List of CFG prompts (optional), if None uses default CFG
            
        Returns:
            Dictionary containing batched input_ids and attention_mask
        """
        if not isinstance(prompts, list):
            prompts = [prompts]
            
        if cfg_prompts is None:
            cfg_prompts = [self.prompt_template['CFG']] * len(prompts)
        elif not isinstance(cfg_prompts, list):
            cfg_prompts = [cfg_prompts] * len(prompts)
            
        all_prompts = []
        all_cfg_prompts = []
        
        for prompt, cfg_prompt in zip(prompts, cfg_prompts):
            # Process prompts
            if cfg_prompt == self.prompt_template['CFG']:
                cfg_prompt = self.prompt_template['CFG']
            else:
                cfg_prompt = self.prompt_template['GENERATION'].format(input=cfg_prompt.strip())
            prompt = self.prompt_template['GENERATION'].format(input=prompt.strip())

            all_prompts.append(
                self.prompt_template['INSTRUCTION'].format(input=prompt)
            )
            all_cfg_prompts.append(
                self.prompt_template['INSTRUCTION'].format(input=cfg_prompt)
            )

        # Encode prompts and CFG prompts
        prompt_input_ids = [
            self.tokenizer.encode(p, add_special_tokens=True, return_tensors='pt')[0]
            for p in all_prompts
        ]
        cfg_prompt_input_ids = [
            self.tokenizer.encode(p, add_special_tokens=True, return_tensors='pt')[0]
            for p in all_cfg_prompts
        ]
        
        # Use right padding to maintain consistency
        prompt_input_ids, prompt_attention_mask = pad_input_ids(
            prompt_input_ids + cfg_prompt_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
            padding_mode="right"
        )

        return dict(
            input_ids=prompt_input_ids.to(self.device),
            attention_mask=prompt_attention_mask.to(self.device),
        )
    
    def interleave_queries_latents(
        self, 
        input_querys: torch.Tensor, 
        latents: torch.Tensor
    ) -> torch.Tensor:
        """
        Interleave query embeddings with latent tokens.
        
        Args:
            input_querys: Query embeddings with shape [B, L, D]
            latents: Latent tokens with shape [B, L, D]
            
        Returns:
            Interleaved tensor with shape [B, 2*L, D]
        """
        B, L, D = input_querys.shape

        # Reshape to [B, L, 1, D] for both tensors
        q = input_querys.unsqueeze(2)          # [B, L, 1, D]
        l = latents.unsqueeze(2)               # [B, L, 1, D]

        # Concatenate along dim=2 → [B, L, 2, D]
        interleaved = torch.cat([q, l], dim=2)

        # Flatten dims 1+2 → [B, 2*L, D]
        interleaved = interleaved.view(B, 2*L, D)
        return interleaved

    def generate_embedding(self, B: int) -> torch.Tensor:
        """
        Generate embedding for image start token.
        
        Args:
            B: Batch size
            
        Returns:
            Expanded embedding tensor
        """
        generate_id = self.tokenizer.encode(
            self.prompt_template['IMG_START_TOKEN'], 
            add_special_tokens=True, 
            return_tensors='pt'
        )[0].to(device=self.device)
        generate_embedding = self.llm.get_input_embeddings()(generate_id)  # [1, dim]
        return generate_embedding.expand(B, -1, -1)

    def forward(
        self, 
        data: Dict, 
        data_samples: Optional[Dict] = None, 
        mode: str = 'loss'
    ) -> Dict:
        pass

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cfg_scale: float = 4.5,
        num_steps: int = 30,
        height: int = 256,
        width: int = 256,
        progress_bar: bool = True,
        cfg_schedule: str = "linear",
        scheduler_type: str = 'random',
        scheduler_seed: Optional[int] = None,
        acc_ratio: int = 1,  # Parallel acceleration token number, 1 means next token predict
        **kwargs
    ) -> torch.Tensor:
        """
        Args:
            input_ids: Input token IDs
            inputs_embeds: Input embeddings
            attention_mask: Attention mask
            cfg_scale: CFG scale for text guidance
            num_steps: Number of diffusion sampling steps
            height: Output image height
            width: Output image width
            progress_bar: Whether to show progress bar
            cfg_schedule: CFG schedule ("linear" or "constant")
            scheduler_type: Type of generation scheduler
            scheduler_seed: Random seed for scheduler
            **kwargs: Additional arguments for sampling
            
        Returns:
            Generated image tensor with shape (batch_size, 3, height, width)
        """
        # ========== 1. Prepare inputs ==========
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        
        bsz_with_cfg = attention_mask.shape[0]
        if cfg_scale > 1.0:
            assert bsz_with_cfg % 2 == 0
            bsz = bsz_with_cfg // 2
        else:
            bsz = bsz_with_cfg
        
        # Calculate dimensions
        vae_h = height // self.vgt_ae.stride // self.latent_patch_size
        vae_w = width // self.vgt_ae.stride // self.latent_patch_size
        total_tokens = vae_h * vae_w
        multi_mode = acc_ratio > 1
        total_step = max(1, total_tokens // acc_ratio)
        
        # ========== 2. Initialize generation scheduler ==========
        scheduler = get_generation_scheduler(
            scheduler_type=scheduler_type,
            num_steps=total_step,
            total_tokens=total_tokens,
            seed=scheduler_seed
        )
        
        generation_schedule = scheduler.schedule()  # [[step1_tokens], [step2_tokens], ...]
        num_groups = len(generation_schedule)

        text_len = inputs_embeds.shape[1]
        
        # ========== 3. Initialize DynamicCache ==========
        past_key_values = DynamicCache()
        
        # Store generation results
        generated_latents_list = []
        
        if progress_bar:
            from tqdm.auto import tqdm
            group_iterator = tqdm(range(1, num_groups), desc="Generating with KV cache")
        else:
            group_iterator = range(1, num_groups)
        
        # Get meta queries and positional embeddings
        meta_queries, pos_embed = self.adaptive_metaquery(target_h=vae_h, target_w=vae_w)
        meta_queries = meta_queries + pos_embed
        meta_queries = meta_queries.to(device=self.device, dtype=self.dtype)
        pos_embed = pos_embed.to(device=self.device, dtype=self.dtype)

        # ========== 4. First group: full forward pass ==========
        first_step_tokens = generation_schedule[0]
        current_queries = meta_queries[:, first_step_tokens].expand(
            bsz_with_cfg, -1, -1
        ).clone().to(device=self.device, dtype=self.dtype)
            
        mllm_inputs = self.prepare_forward_input(
            queries=torch.cat([self.generate_embedding(bsz_with_cfg), current_queries], dim=1),
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )
        
        output = self.llm.model(
            **mllm_inputs,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True
        )
        
        # Cache is automatically updated
        past_key_values = output.past_key_values
        cur_output = output.last_hidden_state[:, -current_queries.shape[1]:, :]
        cur_conditions = self.image_out_projector(cur_output)
        cur_conditions = cur_conditions.view(-1, cur_conditions.shape[-1])

        # Sample first group
        sampled_latents_flat = self.image_head.sample(
            c=cur_conditions,
            cfg=cfg_scale,
            cfg2=1.0,
            num_sampling_steps=num_steps,
            progress=False,
            **kwargs
        )

        sampled_latents_flat = layer_norm(sampled_latents_flat, sampled_latents_flat.size()[1:])
        sampled_latents = sampled_latents_flat.reshape(bsz, len(first_step_tokens), -1)
        generated_latents_list.append(sampled_latents)
        
        # Prepare interleaved sequence for next round
        current_queries = []
        
        if multi_mode:
            cur_in = self.interleave_queries_latents(
                input_querys=meta_queries[:, first_step_tokens].expand(bsz_with_cfg, -1, -1),
                latents=self.image_in_projector(sampled_latents).repeat(bsz_with_cfg // bsz, 1, 1) + 
                        pos_embed[:, first_step_tokens].expand(bsz_with_cfg, -1, -1)
            )
        else:
            cur_in = self.image_in_projector(sampled_latents).repeat(bsz_with_cfg // bsz, 1, 1) + \
                     pos_embed[:, first_step_tokens].expand(bsz_with_cfg, -1, -1)
        current_queries.append(cur_in)
        
        # ========== 5. Generate group by group ==========
        for group_idx in group_iterator:
            # Use scheduler-specified token order
            current_step_tokens = generation_schedule[group_idx]
            current_group_size = len(current_step_tokens)
            input_querys = meta_queries[:, current_step_tokens].expand(bsz_with_cfg, -1, -1).clone()
            
            current_queries.append(input_querys)

            current_queries = torch.cat(current_queries, dim=1)
            current_pos_ids = torch.tensor(
                [text_len + len(generated_latents_list) * current_queries.shape[1] + i 
                 for i in range(current_queries.shape[1])],
                dtype=torch.long,
                device=self.device
            ).unsqueeze(0).expand(bsz_with_cfg, -1)
            
            # KV cache strategy: update cache + compute current query
            output = self.llm.model(
                inputs_embeds=current_queries,
                position_ids=current_pos_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
            )
            
            past_key_values = output.past_key_values
            
            # Extract output and prepare for sampling
            cur_output = output.last_hidden_state[:, -current_group_size:, :]
            cur_conditions = self.image_out_projector(cur_output)
            cur_conditions = cur_conditions.view(-1, cur_conditions.shape[-1])
            
            # Apply CFG schedule
            if cfg_schedule == "linear":
                cfg_iter = 1 + (cfg_scale - 1) * (num_groups - len(generated_latents_list)) / num_groups
            elif cfg_schedule == "constant":
                cfg_iter = cfg_scale

            # Sample current group
            sampled_latents_flat = self.image_head.sample(
                c=cur_conditions,
                cfg=cfg_iter,
                cfg2=1.0,
                num_sampling_steps=num_steps,
                progress=False,
                **kwargs
            )

            if progress_bar:
                group_iterator.set_description(
                    f"Group {group_idx}/{num_groups-1} | CFG={cfg_iter:.2f}"
                )
            
            sampled_latents_flat = layer_norm(sampled_latents_flat, sampled_latents_flat.size()[1:])
            sampled_latents = sampled_latents_flat.reshape(bsz, current_group_size, -1)
            generated_latents_list.append(sampled_latents)
            
            # Prepare interleaved sequence for next round
            current_queries = []
            
            # Add positional embeddings to generated latents
            current_generates = self.image_in_projector(sampled_latents).repeat(bsz_with_cfg // bsz, 1, 1) + \
                                pos_embed[:, current_step_tokens].expand(bsz_with_cfg, -1, -1)

            if multi_mode:
                cur_in = self.interleave_queries_latents(
                    input_querys=meta_queries[:, current_step_tokens].expand(bsz_with_cfg, -1, -1),
                    latents=current_generates
                )
            else:
                cur_in = current_generates
            current_queries.append(cur_in)

        # ========== 6. Merge and decode ==========
        # Rearrange generated latents according to original order
        final_latents_seq = torch.cat(generated_latents_list, dim=1)
        
        # Create inverse mapping to map schedule order back to original order
        original_order = torch.zeros(total_tokens, dtype=torch.long, device=self.device)
        current_idx = 0
        for step_tokens in generation_schedule:
            for token_idx in step_tokens:
                original_order[token_idx] = current_idx
                current_idx += 1
        
        # Rearrange latents to original order
        generated_latents_seq = final_latents_seq[:, original_order, :]
            
        generated_latents = self.unpatchify(
            generated_latents_seq,
            patch_size=self.latent_patch_size,
            h=vae_h,
            w=vae_w
        )
        
        generated_images = self.latents_to_pixels(z=generated_latents)
        
        # Adjust dimensions if needed
        cur_height, cur_width = generated_images.shape[-2:]
        if cur_height != height or cur_width != width:
            generated_images = F.interpolate(
                generated_images,
                size=(height, width),
                mode='bilinear',
                align_corners=False
            )
        
        return generated_images