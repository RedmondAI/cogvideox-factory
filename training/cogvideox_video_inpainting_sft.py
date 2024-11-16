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

def compute_loss(model_pred: torch.Tensor, noise: torch.Tensor, mask: torch.Tensor, latents: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute the loss for video inpainting training.
    
    Args:
        model_pred: Model prediction tensor [B, T, C, H, W]
        noise: Target noise tensor [B, T, C, H, W]
        mask: Binary mask tensor [B, T, 1, H, W]
        latents: Optional original latents for perceptual loss [B, T, C, H, W]
    
    Returns:
        Loss value as a scalar tensor
    """
    # Compute MSE loss in masked regions
    mse_loss = F.mse_loss(model_pred * mask, noise * mask, reduction='none')
    
    # Average over all dimensions except batch
    mse_loss = mse_loss.mean(dim=[1, 2, 3, 4])
    
    # Add temporal consistency loss if sequence length > 1
    if model_pred.shape[1] > 1:
        # Compute temporal gradients
        pred_grad = model_pred[:, 1:] - model_pred[:, :-1]
        noise_grad = noise[:, 1:] - noise[:, :-1]
        mask_grad = mask[:, 1:]  # Mask for gradient regions
        
        # Temporal consistency loss in masked regions
        temp_loss = F.mse_loss(
            pred_grad * mask_grad,
            noise_grad * mask_grad,
            reduction='none'
        ).mean(dim=[1, 2, 3, 4])
        
        # Optional perceptual loss
        if latents is not None and hasattr(F, 'cosine_similarity'):
            pred_features = model_pred.flatten(2)
            latent_features = latents.flatten(2)
            perceptual_loss = (1 - F.cosine_similarity(pred_features, latent_features, dim=2)).mean()
            
            # Combine losses with weights
            loss = mse_loss.mean() + 0.1 * temp_loss.mean() + 0.01 * perceptual_loss
        else:
            # Just MSE and temporal loss
            loss = mse_loss.mean() + 0.1 * temp_loss.mean()
    else:
        # Single frame case - just MSE loss
        loss = mse_loss.mean()
    
    return loss

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
            latents = self.encode_frames(window_rgb)  # Shape: [B, T, 4, H/8, W/8]
            mask = F.interpolate(window_mask, size=latents.shape[-2:], mode='nearest')  # Shape: [B, T, 1, H/8, W/8]
            
            # Model forward pass
            latent_input = torch.cat([latents, mask], dim=2)  # Shape: [B, T, 5, H/8, W/8]
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
        # Use first batch from validation set
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

def train_loop(
    config,
    model,
    noise_scheduler,
    optimizer,
    train_dataloader,
    val_dataloader,
    lr_scheduler,
    accelerator,
    start_global_step,
    num_update_steps_per_epoch,
    num_train_epochs,
    gradient_accumulation_steps,
    checkpoints_total_limit,
):
    # Initialize progress bar
    progress_bar = tqdm(
        range(start_global_step, num_update_steps_per_epoch * num_train_epochs),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")
    global_step = start_global_step
    
    # Training loop
    for epoch in range(num_train_epochs):
        model.train()
        train_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if start_global_step and global_step < start_global_step:
                global_step += 1
                continue
                
            with accelerator.accumulate(model):
                # Get input tensors
                clean_frames = batch["rgb"]
                mask = batch["mask"]
                
                # Sample noise and add to frames
                noise = torch.randn_like(clean_frames)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (clean_frames.shape[0],), 
                    device=clean_frames.device
                )
                noisy_frames = noise_scheduler.add_noise(clean_frames, noise, timesteps)
                
                # Get model prediction
                noise_pred = model(noisy_frames, mask, timesteps)
                
                # Compute loss
                loss = compute_loss(noise_pred, noise, mask, clean_frames)
                
                # Backprop
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Checks if we should save checkpoint
            if accelerator.sync_gradients:
                if global_step % config.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(config.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        
                        if checkpoints_total_limit is not None:
                            checkpoints = os.listdir(config.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            
                            # Remove old checkpoints
                            if len(checkpoints) > checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - checkpoints_total_limit
                                removing_checkpoints = checkpoints[0:num_to_remove]
                                
                                for removing_checkpoint in removing_checkpoints:
                                    removing_path = os.path.join(config.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_path)
                
                # Validation loop
                if global_step % config.validation_steps == 0:
                    model.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        for val_step, val_batch in enumerate(val_dataloader):
                            clean_frames = val_batch["rgb"] 
                            mask = val_batch["mask"]
                            
                            noise = torch.randn_like(clean_frames)
                            timesteps = torch.randint(
                                0, noise_scheduler.config.num_train_timesteps, (clean_frames.shape[0],),
                                device=clean_frames.device
                            )
                            noisy_frames = noise_scheduler.add_noise(clean_frames, noise, timesteps)
                            
                            noise_pred = model(noisy_frames, mask, timesteps)
                            val_loss += compute_loss(noise_pred, noise, mask, clean_frames).item()
                    
                    val_loss /= len(val_dataloader)
                    accelerator.log(
                        {
                            "val_loss": val_loss,
                            "train_loss": loss.detach().item(),
                            "step": global_step,
                        },
                        step=global_step,
                    )
                    model.train()
            
            progress_bar.update(1)
            global_step += 1
            
            # Log metrics
            accelerator.log(
                {
                    "train_loss": loss.detach().item(),
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    "epoch": epoch,
                    "step": global_step,
                },
                step=global_step,
            )
    
    # Save final model
    if accelerator.is_main_process:
        save_path = os.path.join(config.output_dir, "final_model")
        accelerator.save_state(save_path)

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
        torch_dtype=torch.bfloat16,  # Match model precision
    )
    vae_latent_channels = vae.config.latent_channels
    logger.info(f"VAE latent channels: {vae_latent_channels}")
    if vae_latent_channels != 16:
        raise ValueError(f"Expected 16 latent channels for CogVideoX-5b VAE, got {vae_latent_channels}")
    
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
        torch_dtype=torch.bfloat16,  # Match model precision
    )
    
    # Modify transformer input channels to accept mask
    old_proj = transformer.patch_embed.proj
    original_in_channels = old_proj.weight.size(1)
    if original_in_channels != vae_latent_channels:
        logger.warning(
            f"Transformer input channels ({original_in_channels}) don't match VAE latent channels ({vae_latent_channels}). "
            "This may cause issues with the model."
        )
    
    new_proj = torch.nn.Conv2d(
        vae_latent_channels + 1,  # Add mask channel to latent channels
        old_proj.out_channels,
        kernel_size=old_proj.kernel_size,
        stride=old_proj.stride,
        padding=old_proj.padding,
    ).to(dtype=torch.bfloat16)
    
    # Initialize new weights
    with torch.no_grad():
        # Copy latent channel weights
        new_proj.weight[:, :vae_latent_channels] = old_proj.weight[:, :vae_latent_channels]
        # Initialize new mask channel to 0
        new_proj.weight[:, vae_latent_channels:] = 0
        new_proj.bias = torch.nn.Parameter(old_proj.bias.clone())
    
    transformer.patch_embed.proj = new_proj
    
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
    
    # Configure scheduler for better inpainting
    pipeline.scheduler.config.prediction_type = "v_prediction"  # Better for inpainting
    pipeline.scheduler.config.num_train_timesteps = 1000
    pipeline.scheduler.config.beta_schedule = "scaled_linear"
    pipeline.scheduler.config.steps_offset = 1
    pipeline.scheduler.config.clip_sample = False
    
    # Enable memory optimizations
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            transformer.enable_xformers_memory_efficient_attention()
        else:
            logger.warning("xformers not available, falling back to standard attention")
    
    if args.gradient_checkpointing:
        transformer.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    
    # Enable CPU offloading if specified
    if args.use_cpu_offload and accelerator.is_main_process:
        pipeline.enable_model_cpu_offload()
        logger.info("Model CPU offloading enabled")
    
    # Enable slicing and tiling for memory efficiency
    if args.enable_slicing:
        pipeline.vae.enable_slicing()
        logger.info("VAE slicing enabled")
    
    if args.enable_tiling:
        pipeline.vae.enable_tiling()
        logger.info("VAE tiling enabled")
    
    # Dataset and DataLoaders creation
    train_dataset = VideoInpaintingDataset(
        data_root=args.data_root,
        split='train',
        max_num_frames=args.max_num_frames,
        height=720,
        width=1280,
        random_flip_h=args.random_flip_h,
        random_flip_v=args.random_flip_v,
        noise_range=args.noise_range,
    )
    
    val_dataset = VideoInpaintingDataset(
        data_root=args.data_root,
        split='val',
        max_num_frames=args.max_num_frames,
        height=720,
        width=1280,
        random_flip_h=0.0,  # No augmentation for validation
        random_flip_v=0.0,
        noise_range=None,
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
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
                        with torch.amp.autocast('cuda', enabled=accelerator.mixed_precision == "fp16"):
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
                            
                            # Get encoder hidden states
                            encoder_hidden_states = torch.randn(
                                chunk_rgb.shape[0], 
                                args.chunk_size, 
                                transformer.config.hidden_size,  # Match hidden_size
                                device=accelerator.device,
                            )
                            
                            # Forward through transformer
                            model_pred = transformer(
                                sample=torch.cat([gt_latents, mask], dim=2),
                                encoder_hidden_states=encoder_hidden_states,
                                timestep=timesteps,
                            )
                            
                            # Remove padding
                            model_pred = unpad(model_pred, (rgb_pad[0]//8, rgb_pad[1]//8))
                            noise = unpad(noise, (rgb_pad[0]//8, rgb_pad[1]//8))
                            
                            # Calculate loss
                            loss = compute_loss(model_pred, noise, mask, chunk_gt)
                        
                        # Scale loss and backward pass
                        scaler.scale(loss).backward()
                        chunk_losses.append(loss)
                        
                        # Compute metrics
                        if step % args.validation_steps == 0:
                            with torch.no_grad():
                                metrics = compute_metrics(model_pred, noise, mask)
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
                        validation_data=next(iter(accelerator.get_eval_dataloader())),
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