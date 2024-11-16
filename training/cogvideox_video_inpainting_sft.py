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
from typing import Any, Dict, Tuple, Optional, Union, List
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
        vae: AutoencoderKLCogVideoX,
        transformer: CogVideoXTransformer3DModel,
        scheduler: CogVideoXDPMScheduler,
        text_encoder=None,
        tokenizer=None,
    ):
        self.vae = vae
        self.transformer = transformer
        self.scheduler = scheduler
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        
        # Set default device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16
        
        # Move models to device
        self.vae = self.vae.to(self.device, dtype=self.dtype)
        self.transformer = self.transformer.to(self.device, dtype=self.dtype)
        
        # Configure scheduler
        self.scheduler.config.prediction_type = "v_prediction"
        self.scheduler.config.rescale_betas_zero_snr = True
        self.scheduler.config.snr_shift_scale = 1.0
        
        # Get model dimensions
        self.model_height = transformer.config.sample_height
        self.model_width = transformer.config.sample_width
        self.model_frames = transformer.config.sample_frames
    
    def _calculate_scaling(self, height: int, width: int, num_frames: int):
        """Calculate scaling factors between input and model dimensions."""
        # Initialize accelerate state for logging
        from accelerate.state import PartialState
        _ = PartialState()
        
        # Calculate spatial scaling
        spatial_scale = min(
            height / self.model_height,
            width / self.model_width
        )
        
        # Calculate temporal scaling
        temporal_scale = num_frames / self.model_frames
        
        logger.info("Dimension scaling factors:")
        logger.info(f"  Spatial: {spatial_scale:.2f}")
        logger.info(f"  Temporal: {temporal_scale:.2f}")
        
        return spatial_scale, temporal_scale
    
    def validate_dimensions(self, video: torch.Tensor, mask: torch.Tensor):
        """Validate input dimensions and scaling."""
        if video.ndim != 5:
            raise ValueError(f"Video must be 5D [B,C,T,H,W], got shape {video.shape}")
        if mask.ndim != 5:
            raise ValueError(f"Mask must be 5D [B,1,T,H,W], got shape {mask.shape}")
        
        B, C, T, H, W = video.shape
        B_m, C_m, T_m, H_m, W_m = mask.shape
        
        # Check batch size
        if B != B_m:
            raise ValueError(f"Video batch size {B} != mask batch size {B_m}")
        
        # Check channels
        if C != 3:
            raise ValueError(f"Video must have 3 channels, got {C}")
        if C_m != 1:
            raise ValueError(f"Mask must have 1 channel, got {C_m}")
        
        # Check temporal dimension
        if T != T_m:
            raise ValueError(f"Video frames {T} != mask frames {T_m}")
        
        # Check spatial dimensions
        if H != H_m or W != W_m:
            raise ValueError(f"Video size ({H}x{W}) != mask size ({H_m}x{W_m})")
        
        # Calculate and validate scaling
        spatial_scale, temporal_scale = self._calculate_scaling(H, W, T)
        
        # Log memory requirements
        bytes_per_element = 2 if self.dtype == torch.float16 else 4
        video_memory = B * C * T * H * W * bytes_per_element / (1024**3)  # GB
        latent_memory = B * 16 * (T//2) * (H//8) * (W//8) * bytes_per_element / (1024**3)  # GB
        transformer_memory = 10.8  # GB (from analysis)
        total_memory = video_memory + latent_memory + transformer_memory
        
        logger.info("Memory requirements:")
        logger.info(f"  Video: {video_memory:.2f}GB")
        logger.info(f"  Latents: {latent_memory:.2f}GB")
        logger.info(f"  Transformer: {transformer_memory:.2f}GB")
        logger.info(f"  Total: {total_memory:.2f}GB")
        
        return spatial_scale, temporal_scale
    
    @torch.no_grad()
    def encode(self, x: torch.Tensor, chunk_size: Optional[int] = None, overlap: int = 32) -> torch.Tensor:
        """Encode input video to latent space.
        
        Args:
            x: Input tensor of shape [B, C, T, H, W]
            chunk_size: Optional size for chunked processing of large inputs
            overlap: Overlap size between chunks for smooth blending
            
        Returns:
            Latent tensor of shape [B, C, T//2, H//8, W//8]
        """
        B, C, T, H, W = x.shape
        
        # Process in chunks if input is large
        if chunk_size is not None and W > chunk_size:
            # Calculate effective chunk size with overlap
            effective_chunk = chunk_size - 2 * overlap
            num_chunks = math.ceil(W / effective_chunk)
            chunks = []
            weights = []  # For overlap blending
            chunk_starts = []  # Track start positions
            
            for i in range(num_chunks):
                # Clear cache before processing chunk
                torch.cuda.empty_cache()
                
                # Calculate chunk boundaries with overlap
                start_w = max(0, i * effective_chunk - overlap)
                end_w = min((i + 1) * effective_chunk + overlap, W)
                chunk_starts.append(start_w)
                
                # Process chunk
                chunk_x = x[..., start_w:end_w]
                chunk_latents = self.vae.encode(chunk_x).latent_dist.sample()
                chunk_latents = chunk_latents * self.vae.config.scaling_factor
                
                # Create blending weights
                weight = torch.ones_like(chunk_latents)
                if i > 0:  # Left overlap
                    left_size = overlap // 8
                    weight[..., :left_size] = torch.linspace(0, 1, left_size, device=weight.device).view(1, 1, 1, 1, -1)
                if i < num_chunks - 1:  # Right overlap
                    right_size = overlap // 8
                    weight[..., -right_size:] = torch.linspace(1, 0, right_size, device=weight.device).view(1, 1, 1, 1, -1)
                
                chunks.append(chunk_latents)
                weights.append(weight)
                
                # Clear cache after processing chunk
                torch.cuda.empty_cache()
            
            # Blend chunks with weights
            final_latents = torch.zeros(B, 16, T//2, H//8, W//8, device=x.device, dtype=torch.float16)
            weight_sum = torch.zeros_like(final_latents)
            
            for chunk, weight, start_w in zip(chunks, weights, chunk_starts):
                start_idx = start_w // 8  # Convert to latent space
                chunk_width = chunk.shape[-1]
                final_latents[..., start_idx:start_idx + chunk_width] += chunk * weight
                weight_sum[..., start_idx:start_idx + chunk_width] += weight
            
            # Normalize by weight sum
            final_latents = final_latents / (weight_sum + 1e-8)
            return final_latents
        
        # Process normally if input is small
        latents = self.vae.encode(x).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        return latents
    
    @torch.no_grad()
    def decode(self, latents: torch.Tensor, chunk_size: Optional[int] = None, overlap: int = 32) -> torch.Tensor:
        """Decode latents to video space with temporal expansion.
        
        Args:
            latents: Latent tensor of shape [B, C, T, H, W]
            chunk_size: Optional size for chunked processing of large inputs
            overlap: Overlap size between chunks for smooth blending
            
        Returns:
            Video tensor of shape [B, C, 8, H*8, W*8]  # Fixed 8 frames output
        """
        B, C, T, H, W = latents.shape
        
        # Process in chunks if input is large
        if chunk_size is not None and W * 8 > chunk_size:  # Compare in pixel space
            # Calculate effective chunk size with overlap
            effective_chunk = (chunk_size - 2 * overlap) // 8  # Work in latent space
            num_chunks = math.ceil(W / effective_chunk)
            chunks = []
            weights = []  # For overlap blending
            chunk_starts = []  # Track start positions
            
            for i in range(num_chunks):
                # Clear cache before processing chunk
                torch.cuda.empty_cache()
                
                # Calculate chunk boundaries with overlap
                start_w = max(0, i * effective_chunk - overlap // 8)
                end_w = min((i + 1) * effective_chunk + overlap // 8, W)
                chunk_starts.append(start_w)
                
                # Process chunk
                chunk_latents = latents[..., start_w:end_w]
                chunk_latents = 1 / self.vae.config.scaling_factor * chunk_latents
                chunk_video = self.vae.decode(chunk_latents).sample
                
                # Create blending weights
                weight = torch.ones_like(chunk_video)
                if i > 0:  # Left overlap
                    left_size = overlap
                    weight[..., :left_size] = torch.linspace(0, 1, left_size, device=weight.device).view(1, 1, 1, 1, -1)
                if i < num_chunks - 1:  # Right overlap
                    right_size = overlap
                    weight[..., -right_size:] = torch.linspace(1, 0, right_size, device=weight.device).view(1, 1, 1, 1, -1)
                
                chunks.append(chunk_video)
                weights.append(weight)
                
                # Clear cache after processing chunk
                torch.cuda.empty_cache()
            
            # Blend chunks with weights
            final_video = torch.zeros(B, C, 8, H*8, W*8, device=latents.device, dtype=torch.float16)
            weight_sum = torch.zeros_like(final_video)
            
            for chunk, weight, start_w in zip(chunks, weights, chunk_starts):
                start_idx = start_w * 8  # Convert to pixel space
                chunk_width = chunk.shape[-1]
                final_video[..., start_idx:start_idx + chunk_width] += chunk * weight
                weight_sum[..., start_idx:start_idx + chunk_width] += weight
            
            # Normalize by weight sum
            final_video = final_video / (weight_sum + 1e-8)
            return final_video
        
        # Process normally if input is small
        latents = 1 / self.vae.config.scaling_factor * latents
        video = self.vae.decode(latents).sample
        return video
    
    def prepare_latents(
        self,
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Prepare random latents accounting for temporal compression."""
        latents_shape = (batch_size, self.transformer.config.in_channels, num_frames//2, height//8, width//8)
        latents = torch.randn(latents_shape, generator=generator, device=self.device, dtype=self.dtype)
        latents = latents * self.scheduler.init_noise_sigma
        return latents
    
    def prepare_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Prepare mask for latent space.
        
        Args:
            mask: Binary mask of shape [B, 1, T, H, W]
            
        Returns:
            Processed mask of shape [B, 1, T//2, H//8, W//8]
        """
        mask = F.interpolate(mask, size=(mask.shape[2]//2, mask.shape[3]//8, mask.shape[4]//8), mode='nearest')
        return mask
    
    def prepare_encoder_hidden_states(
        self,
        prompt: Union[str, List[str]],
        batch_size: int = 1,
    ) -> torch.Tensor:
        """Prepare text conditioning.
        
        Args:
            prompt: Text prompt or list of prompts
            batch_size: Batch size
            
        Returns:
            Conditioning tensor of shape [B, 1, 4096]
        """
        if self.text_encoder is None:
            # Return random conditioning if no text encoder
            return torch.randn(batch_size, 1, self.transformer.config.text_embed_dim, device=self.device, dtype=self.dtype)
            
        if isinstance(prompt, str):
            prompt = [prompt] * batch_size
            
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.device)
        
        encoder_hidden_states = self.text_encoder(text_input_ids)[0]
        # Take first token only
        encoder_hidden_states = encoder_hidden_states[:, :1]
        return encoder_hidden_states
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        video: torch.Tensor,
        mask: torch.Tensor,
        num_inference_steps: int = 50,
        generator: Optional[torch.Generator] = None,
        eta: float = 0.0,
        **kwargs,
    ):
        """Run inpainting pipeline.
        
        Args:
            prompt: Conditioning text
            video: Input video [B, C, T, H, W]
            mask: Binary mask [B, 1, T, H, W]
            num_inference_steps: Number of denoising steps
            generator: Random number generator
            eta: Eta parameter for variance during sampling
            
        Returns:
            Generated video completing the masked regions
        """
        batch_size = video.shape[0]
        
        # Validate inputs and calculate scaling
        spatial_scale, temporal_scale = self.validate_dimensions(video, mask)
        
        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps
        
        # Prepare inputs
        latents = self.encode(video)
        mask = self.prepare_mask(mask)
        encoder_hidden_states = self.prepare_encoder_hidden_states(prompt, batch_size)
        
        # Initialize noise
        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps[0])
        
        # Prepare for transformer
        latent_model_input = noisy_latents.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
        
        # Denoise
        for i, t in enumerate(timesteps[:-1]):
            t_back = timesteps[i + 1]
            
            # Predict noise
            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=t,
                encoder_hidden_states=encoder_hidden_states,
            ).sample
            
            # Scheduler step
            latent_output = self.scheduler.step(
                model_output=noise_pred,
                timestep=t,
                timestep_back=t_back,
                sample=latent_model_input,
                old_pred_original_sample=latents.permute(0, 2, 1, 3, 4),
            )[0]
            
            # Apply mask
            latent_model_input = (
                latent_output * mask + latent_model_input * (1 - mask)
            )
        
        # Final decoding
        latent_model_input = latent_model_input.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        video = self.decode(latent_model_input)
        
        return video

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
        prompt="",
        video=rgb,
        mask=mask,
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
                noise_pred = model(
                    hidden_states=noisy_frames,
                    timestep=timesteps,
                    encoder_hidden_states=None,  # No text conditioning during training
                ).sample
                
                # Compute loss
                loss = compute_loss(noise_pred, noise, mask, noisy_frames)
                
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
                            
                            noise_pred = model(
                                hidden_states=noisy_frames,
                                timestep=timesteps,
                                encoder_hidden_states=None,  # No text conditioning during training
                            ).sample
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

def get_default_config():
    """Get default training configuration."""
    return {
        # Model configuration
        "model_height": 60,     # Native model height
        "model_width": 90,      # Native model width
        "model_frames": 49,     # Native model frame count
        "in_channels": 16,
        "out_channels": 16,
        "attention_heads": 48,
        "attention_head_dim": 64,
        "num_layers": 42,
        "patch_size": 2,
        "text_embed_dim": 4096,
        "time_embed_dim": 512,
        "temporal_compression": 4,
        
        # Input configuration
        "input_height": 720,    # Input video height
        "input_width": 1280,    # Input video width
        "input_frames": 100,    # Input frame count
        
        # Training configuration
        "learning_rate": 1e-4,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_weight_decay": 1e-2,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1.0,
        
        # Scheduler configuration
        "num_train_timesteps": 1000,
        "beta_start": 0.00085,
        "beta_end": 0.012,
        "beta_schedule": "scaled_linear",
        "prediction_type": "v_prediction",
        "set_alpha_to_one": True,
        "steps_offset": 0,
        "rescale_betas_zero_snr": True,
        "snr_shift_scale": 1.0,
        "timestep_spacing": "trailing",
        
        # Memory optimization
        "gradient_checkpointing": True,
        "use_8bit_adam": True,
        "enable_xformers_memory_efficient_attention": True,
        
        # Mixed precision
        "mixed_precision": "fp16",
        "dtype": torch.float16,
        
        # Dataset configuration
        "train_batch_size": 1,  # Due to high memory usage (10.8GB per sample)
        "eval_batch_size": 1,
        "dataloader_num_workers": 4,
        "num_inference_steps": 50,
    }

def create_pipeline(args):
    """Create and configure the inpainting pipeline."""
    config = get_default_config()
    config.update(vars(args))
    
    # Load models
    vae = AutoencoderKLCogVideoX.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=config["dtype"]
    )
    
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=config["dtype"]
    )
    
    scheduler = CogVideoXDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    
    # Update scheduler config
    scheduler.config.prediction_type = config["prediction_type"]
    scheduler.config.beta_schedule = config["beta_schedule"]
    scheduler.config.beta_start = config["beta_start"]
    scheduler.config.beta_end = config["beta_end"]
    scheduler.config.set_alpha_to_one = config["set_alpha_to_one"]
    scheduler.config.steps_offset = config["steps_offset"]
    scheduler.config.rescale_betas_zero_snr = config["rescale_betas_zero_snr"]
    scheduler.config.snr_shift_scale = config["snr_shift_scale"]
    scheduler.config.timestep_spacing = config["timestep_spacing"]
    
    # Create pipeline
    pipeline = CogVideoXInpaintingPipeline(
        vae=vae,
        transformer=transformer,
        scheduler=scheduler,
    )
    
    # Enable memory optimizations
    if config["gradient_checkpointing"]:
        transformer.enable_gradient_checkpointing()
    
    if config["enable_xformers_memory_efficient_attention"]:
        transformer.enable_xformers_memory_efficient_attention()
    
    return pipeline, config

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
    
    pipeline, config = create_pipeline(args)
    
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
            pipeline.transformer.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    else:
        optimizer = torch.optim.AdamW(
            pipeline.transformer.parameters(),
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
    pipeline.transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        pipeline.transformer, optimizer, train_dataloader, lr_scheduler
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
        pipeline.transformer.train()
        epoch_metrics = defaultdict(float)
        
        for step, batch in enumerate(train_dataloader):
            try:
                with accelerator.accumulate(pipeline.transformer):
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
                            rgb_latents = pipeline.encode(chunk_rgb)
                            gt_latents = pipeline.encode(chunk_gt)
                            
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
                                pipeline.transformer.config.hidden_size,  # Match hidden_size
                                device=accelerator.device,
                            )
                            
                            # Forward through transformer
                            model_pred = pipeline.transformer(
                                hidden_states=torch.cat([gt_latents, mask], dim=2),
                                encoder_hidden_states=encoder_hidden_states,
                                timestep=timesteps,
                            ).sample
                            
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
                        accelerator.clip_grad_norm_(pipeline.transformer.parameters(), args.max_grad_norm)
                    
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
                    
                    pipeline.transformer = accelerator.unwrap_model(pipeline.transformer)
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