# Copyright 2024 The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")

import gc
import logging
import math
import os
import shutil
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Tuple, Optional
from collections import defaultdict

import diffusers
import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    set_seed,
)
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXTransformer3DModel,
)
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import export_to_video
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from huggingface_hub import create_repo, upload_folder
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch.cuda
from torch.cuda.amp import autocast, GradScaler
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from args import get_args
from dataset import VideoInpaintingDataset, BucketSampler
from utils import (
    get_gradient_norm,
    get_optimizer,
    prepare_rotary_positional_embeddings,
    print_memory,
    reset_memory,
    unwrap_model,
)

logger = get_logger(__name__)

def pad_to_multiple(x: torch.Tensor, multiple: int = 64, max_dim: int = 2048) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Pad tensor to multiple with size safety check."""
    h, w = x.shape[-2:]
    print(f"Input dimensions: h={h}, w={w}")
    
    # Check if input dimensions exceed maximum
    if h > max_dim or w > max_dim:
        raise ValueError(f"Input dimensions ({h}, {w}) exceed maximum safe size {max_dim}")
    
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    print(f"Padding amounts: pad_h={pad_h}, pad_w={pad_w}")
    print(f"Final dimensions will be: h={h+pad_h}, w={w+pad_w}")
    
    # Check if padded dimensions exceed or equal maximum (we want strictly less than max_dim)
    if h + pad_h >= max_dim or w + pad_w >= max_dim:
        raise ValueError(f"Padded dimensions ({h+pad_h}, {w+pad_w}) must be strictly less than maximum size {max_dim}")
    
    # Pad tensor
    x_padded = F.pad(x, (0, pad_w, 0, pad_h))
    return x_padded, (pad_h, pad_w)

def unpad(x: torch.Tensor, pad_sizes: Tuple[int, int]) -> torch.Tensor:
    """Remove padding from tensor."""
    pad_h, pad_w = pad_sizes
    if pad_h > 0:
        x = x[..., :-pad_h, :]
    if pad_w > 0:
        x = x[..., :-pad_w]
    return x

def temporal_smooth(frames: torch.Tensor, window_size: int = 5) -> torch.Tensor:
    """Apply temporal smoothing at chunk boundaries.
    
    Args:
        frames: Input frames tensor of shape [B, T, C, H, W]
        window_size: Size of the smoothing window
    
    Returns:
        Smoothed frames tensor of same shape
    """
    if frames.shape[1] <= window_size:
        return frames
    
    smoothed = frames.clone()
    half_window = window_size // 2
    
    # Create weights for smooth transition
    weights = torch.linspace(0, 1, window_size, device=frames.device)
    weights = weights.view(1, -1, 1, 1, 1)
    
    # Apply smoothing in the middle region
    mid_start = frames.shape[1] // 2 - half_window
    mid_end = mid_start + window_size
    
    # Get the frames before and after transition
    pre_transition = frames[:, mid_start-1:mid_start-1+window_size]
    post_transition = frames[:, mid_start:mid_start+window_size]
    
    # Blend between pre and post transition frames
    smoothed[:, mid_start:mid_end] = (
        pre_transition * (1 - weights) +
        post_transition * weights
    )
    
    return smoothed

def compute_metrics(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    """Compute metrics with memory efficiency.
    
    Args:
        pred: Predicted frames [B, T, C, H, W]
        gt: Ground truth frames [B, T, C, H, W]
        mask: Binary mask [B, T, 1, H, W]
    
    Returns:
        Dict of metrics including PSNR, SSIM, and temporal consistency
    """
    metrics = {}
    
    # Move to CPU for metric computation
    with torch.no_grad():
        pred_cpu = pred.detach().cpu()
        gt_cpu = gt.detach().cpu()
        mask_cpu = mask.detach().cpu()
        
        # Initialize metric computers on CPU
        psnr = PeakSignalNoiseRatio()
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0)  # Specify data range
        
        # Compute metrics for masked regions
        masked_pred = pred_cpu * mask_cpu
        masked_gt = gt_cpu * mask_cpu
        
        # Reshape for 2D metrics (combine batch and time dimensions)
        B, T, C, H, W = pred_cpu.shape
        masked_pred_2d = masked_pred.view(B*T, C, H, W)
        masked_gt_2d = masked_gt.view(B*T, C, H, W)
        
        metrics['masked_psnr'] = psnr(masked_pred_2d, masked_gt_2d)
        metrics['masked_ssim'] = ssim(masked_pred_2d, masked_gt_2d)
        
        # Temporal consistency (compute on GPU if memory allows)
        if pred.device.type == "cuda" and torch.cuda.memory_allocated() < torch.cuda.max_memory_allocated() * 0.8:
            device = pred.device
            pred_diff = (pred[:, 1:] - pred[:, :-1]).abs().mean()
            gt_diff = (gt[:, 1:] - gt[:, :-1]).abs().mean()
            metrics['temporal_consistency'] = 1.0 - (pred_diff - gt_diff).abs().item()
        else:
            pred_diff = (pred_cpu[:, 1:] - pred_cpu[:, :-1]).abs().mean()
            gt_diff = (gt_cpu[:, 1:] - gt_cpu[:, :-1]).abs().mean()
            metrics['temporal_consistency'] = 1.0 - (pred_diff - gt_diff).abs().item()
    
    return metrics

class CogVideoXInpaintingPipeline:
    def __init__(
        self,
        vae,
        transformer,
        scheduler,
        window_size: int = 32,
        overlap: int = 8,
        vae_precision: str = "fp16",
        max_resolution: int = 2048,
    ):
        super().__init__()
        self.vae = vae
        self.transformer = transformer
        self.scheduler = scheduler
        self.window_size = window_size
        self.overlap = overlap
        self.vae_precision = vae_precision
        self.max_resolution = max_resolution
        
        # Set model configurations
        if hasattr(transformer, 'config'):
            transformer.config.use_memory_efficient_attention = True
            transformer.config.attention_mode = "xformers"
            transformer.config.gradient_checkpointing_steps = 2
        
        # Set scheduler configurations
        scheduler.config.prediction_type = "epsilon"
        scheduler.config.num_train_timesteps = 1000
        scheduler.config.beta_schedule = "scaled_linear"
    
    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode frames with specified precision."""
        dtype = torch.float16 if self.vae_precision == "fp16" else torch.float32
        with torch.cuda.amp.autocast(enabled=self.vae_precision == "fp16"):
            latents = self.vae.encode(frames).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        return latents.to(dtype)
    
    def _process_window(
        self,
        rgb_frames: torch.Tensor,
        mask_frames: torch.Tensor,
        timestep: torch.Tensor,
        start_idx: int,
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Process window with memory management."""
        try:
            # Handle edge case for last window
            window_size = min(self.window_size, rgb_frames.shape[1] - start_idx)
            if window_size < self.window_size:
                pad_size = self.window_size - window_size
                rgb_frames = F.pad(rgb_frames, (0, 0, 0, 0, 0, 0, 0, pad_size))
                mask_frames = F.pad(mask_frames, (0, 0, 0, 0, 0, 0, 0, pad_size))
            
            window_rgb = rgb_frames[:, start_idx:start_idx + self.window_size]
            window_mask = mask_frames[:, start_idx:start_idx + self.window_size]
            
            # Pad resolution
            window_rgb, rgb_pad = pad_to_multiple(window_rgb, max_dim=self.max_resolution)
            window_mask, _ = pad_to_multiple(window_mask, max_dim=self.max_resolution)
            
            # Process with appropriate precision
            latents = self.encode_frames(window_rgb)
            mask = F.interpolate(window_mask, size=latents.shape[-2:])
            
            # Model forward pass
            latent_input = torch.cat([latents, mask], dim=2)
            noise_pred = self.transformer(
                sample=latent_input,
                timestep=timestep,
                return_dict=False,
            )[0]
            
            torch.cuda.empty_cache()
            return noise_pred, rgb_pad
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                logger.error("OOM error in window processing, attempting recovery...")
            raise
        except Exception as e:
            logger.error(f"Error processing window at index {start_idx}: {e}")
            raise

    @torch.no_grad()
    def __call__(
        self,
        rgb_frames: torch.Tensor,
        mask_frames: torch.Tensor,
        num_inference_steps: int = 50,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        self.scheduler.set_timesteps(num_inference_steps)
        device = self.transformer.device
        
        # Initialize output tensor
        output_frames = torch.zeros_like(rgb_frames)
        blend_weights = torch.zeros((1, rgb_frames.shape[1], 1, 1, 1), device=device)
        
        try:
            for t in self.scheduler.timesteps:
                # Process each window
                for start_idx in range(0, rgb_frames.shape[1] - self.window_size + 1, 
                                     self.window_size - self.overlap):
                    noise_pred, pad = self._process_window(rgb_frames, mask_frames, t, start_idx)
                    
                    # Create blending weights for this window
                    window_weights = torch.ones((1, self.window_size, 1, 1, 1), device=device)
                    if start_idx > 0:  # Blend start
                        window_weights[:, :self.overlap] = torch.linspace(0, 1, self.overlap, device=device).view(1, -1, 1, 1, 1)
                    if start_idx + self.window_size < rgb_frames.shape[1]:  # Blend end
                        window_weights[:, -self.overlap:] = torch.linspace(1, 0, self.overlap, device=device).view(1, -1, 1, 1, 1)
                    
                    # Remove padding from prediction
                    noise_pred = unpad(noise_pred, pad)
                    
                    # Accumulate predictions and weights
                    output_frames[:, start_idx:start_idx + self.window_size] += noise_pred * window_weights
                    blend_weights[:, start_idx:start_idx + self.window_size] += window_weights
                    
                    torch.cuda.empty_cache()  # Clear cache after each window
                
                # Apply temporal smoothing
                output_frames = temporal_smooth(output_frames)
            
            # Normalize by blend weights
            output_frames = output_frames / (blend_weights + 1e-8)
            
            return output_frames
            
        except Exception as e:
            logger.error(f"Error in pipeline inference: {e}")
            raise

def collate_fn(examples):
    rgb = torch.stack([example["rgb"] for example in examples])
    mask = torch.stack([example["mask"] for example in examples])
    gt = torch.stack([example["gt"] for example in examples])
    
    return {
        "rgb": rgb,
        "mask": mask,
        "gt": gt,
    }

def log_validation(
    accelerator,
    pipeline,
    args,
    epoch,
    validation_data=None,
):
    logger.info("Running validation...")
    
    pipeline.transformer.eval()
    pipeline.vae.eval()
    
    if validation_data is None:
        # Use first batch from training set
        validation_data = next(iter(accelerator.get_eval_dataloader()))
    
    rgb = validation_data["rgb"]
    mask = validation_data["mask"]
    
    # Generate video
    videos = pipeline(
        rgb_frames=rgb,
        mask_frames=mask,
        num_inference_steps=args.num_inference_steps,
    )
    
    # Log videos
    if accelerator.is_main_process:
        for idx, video in enumerate(videos):
            # Save video
            video_path = os.path.join(args.output_dir, f"validation_epoch_{epoch}_sample_{idx}.mp4")
            export_to_video(video, video_path, fps=args.fps)
            
            if args.tracker_project_name is not None:
                wandb.log({
                    f"validation_video_{idx}": wandb.Video(video_path, fps=args.fps, format="mp4"),
                    "epoch": epoch,
                })
    
    pipeline.transformer.train()
    pipeline.vae.train()

def main(args):
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=logging_dir,
    )
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=True),
            InitProcessGroupKwargs(timeout=timedelta(hours=4)),
        ],
    )
    
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    
    # Load models
    vae = AutoencoderKLCogVideoX.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
    )
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
    )
    
    # Modify transformer input channels to accept mask
    transformer.config.in_channels += 1  # Add channel for mask
    old_conv = transformer.conv_in
    transformer.conv_in = torch.nn.Conv3d(
        transformer.config.in_channels,
        transformer.conv_in.out_channels,
        kernel_size=transformer.conv_in.kernel_size,
        stride=transformer.conv_in.stride,
        padding=transformer.conv_in.padding,
    )
    # Initialize new conv with weights from old conv for RGB channels
    with torch.no_grad():
        transformer.conv_in.weight[:, :3] = old_conv.weight
        transformer.conv_in.weight[:, 3:] = 0  # Initialize mask channels to 0
        transformer.conv_in.bias = torch.nn.Parameter(old_conv.bias.clone())
    
    # Freeze VAE
    vae.requires_grad_(False)
    
    # Create pipeline
    pipeline = CogVideoXInpaintingPipeline(
        vae=vae,
        transformer=transformer,
        scheduler=CogVideoXDPMScheduler.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="scheduler",
        ),
        window_size=args.window_size,
        overlap=args.overlap,
        vae_precision=args.vae_precision,
        max_resolution=args.max_resolution,
    )
    
    # Enable memory efficient attention
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            transformer.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
    
    # Dataset and DataLoaders
    train_dataset = VideoInpaintingDataset(
        data_root=args.data_root,
        max_num_frames=args.max_num_frames,
        height=720,
        width=1280,
        random_flip_h=args.random_flip_h,
        random_flip_v=args.random_flip_v,
        noise_range=args.noise_range,
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    
    # Optimizer
    if args.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            transformer.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    else:
        optimizer = torch.optim.AdamW(
            transformer.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    
    # Scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    
    # Prepare everything with accelerator
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )
    
    # Initialize gradient scaler for mixed precision
    scaler = GradScaler(enabled=args.mixed_precision == "fp16")
    
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    
    global_step = 0
    first_epoch = 0
    
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="Steps",
    )
    
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        epoch_metrics = defaultdict(float)
        
        for step, batch in enumerate(train_dataloader):
            try:
                with accelerator.accumulate(transformer):
                    total_frames = batch["rgb"].shape[1]
                    chunk_losses = []
                    chunk_metrics = defaultdict(list)
                    
                    for start_idx in range(0, total_frames - args.chunk_size + 1, 
                                         args.chunk_size - args.overlap):
                        # Process chunk with mixed precision
                        with autocast(enabled=args.mixed_precision == "fp16"):
                            # Get and pad chunk
                            chunk_rgb = batch["rgb"][:, start_idx:start_idx + args.chunk_size]
                            chunk_mask = batch["mask"][:, start_idx:start_idx + args.chunk_size]
                            chunk_gt = batch["gt"][:, start_idx:start_idx + args.chunk_size]
                            
                            chunk_rgb, rgb_pad = pad_to_multiple(chunk_rgb, max_dim=args.max_resolution)
                            chunk_mask, _ = pad_to_multiple(chunk_mask, max_dim=args.max_resolution)
                            chunk_gt, _ = pad_to_multiple(chunk_gt, max_dim=args.max_resolution)
                            
                            # Process with VAE
                            rgb_latents = pipeline.encode_frames(chunk_rgb)
                            gt_latents = pipeline.encode_frames(chunk_gt)
                            
                            # Prepare mask and noise
                            mask = F.interpolate(chunk_mask, size=rgb_latents.shape[-2:])
                            noise = torch.randn_like(gt_latents)
                            timesteps = torch.randint(
                                0, pipeline.scheduler.config.num_train_timesteps,
                                (chunk_rgb.shape[0],), device=chunk_rgb.device
                            )
                            
                            # Model forward pass
                            noisy_latents = pipeline.scheduler.add_noise(gt_latents, noise, timesteps)
                            model_input = torch.cat([noisy_latents, mask], dim=2)
                            noise_pred = transformer(
                                sample=model_input,
                                timestep=timesteps,
                                return_dict=False,
                            )[0]
                            
                            # Remove padding
                            noise_pred = unpad(noise_pred, (rgb_pad[0]//8, rgb_pad[1]//8))
                            noise = unpad(noise, (rgb_pad[0]//8, rgb_pad[1]//8))
                            
                            # Calculate loss
                            loss = compute_loss(noise_pred, noise, mask)
                        
                        # Scale loss and backward pass
                        scaler.scale(loss).backward()
                        chunk_losses.append(loss)
                        
                        # Compute metrics
                        if step % args.validation_steps == 0:
                            with torch.no_grad():
                                metrics = compute_metrics(noise_pred, noise, mask)
                                for k, v in metrics.items():
                                    chunk_metrics[k].append(v)
                        
                        torch.cuda.empty_cache()
                    
                    # Average losses and update
                    if accelerator.sync_gradients:
                        scaler.unscale_(optimizer)
                        accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                    # Update metrics
                    if chunk_metrics:
                        for k, v in chunk_metrics.items():
                            epoch_metrics[k] += torch.stack(v).mean().item()
                
            except Exception as e:
                logger.error(f"Error in training step {step}, epoch {epoch}: {e}")
                continue
            
            if global_step % args.validation_steps == 0:
                if accelerator.is_main_process:
                    # Log metrics
                    for k, v in epoch_metrics.items():
                        wandb.log({f"train/{k}": v / step}, step=global_step)
                    
                    pipeline.transformer = accelerator.unwrap_model(transformer)
                    log_validation(
                        accelerator=accelerator,
                        pipeline=pipeline,
                        args=args,
                        epoch=epoch,
                    )
            
            progress_bar.update(1)
            global_step += 1
    
    accelerator.end_training()

if __name__ == "__main__":
    args = get_args()
    args.window_size = getattr(args, 'window_size', 32)
    args.overlap = getattr(args, 'overlap', 8)
    args.chunk_size = getattr(args, 'chunk_size', 32)
    args.use_8bit_adam = getattr(args, 'use_8bit_adam', False)
    args.use_flash_attention = getattr(args, 'use_flash_attention', False)
    args.vae_precision = getattr(args, 'vae_precision', "fp16")
    args.max_resolution = getattr(args, 'max_resolution', 2048)
    main(args)