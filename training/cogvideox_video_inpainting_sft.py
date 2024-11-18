# Import components
from training.components import (
    pad_to_multiple,
    unpad,
    temporal_smooth,
    compute_metrics,
    compute_loss,
    compute_loss_v_pred,
    compute_loss_v_pred_with_snr,
    handle_vae_temporal_output,
    CogVideoXInpaintingPipeline as BasePipeline
)

# Keep existing imports
import math
import gc
import logging
import os
import shutil
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, Union, List
from collections import defaultdict
import traceback

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

from utils import (
    get_gradient_norm,
    get_optimizer,
    prepare_rotary_positional_embeddings,
    print_memory,
    reset_memory,
    unwrap_model,
)

from training.dataset import VideoInpaintingDataset

import logging
import sys
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

def unwrap_model(model):
    """Unwrap a model from its distributed wrapper."""
    if hasattr(model, "module"):
        return model.module
    return model

class CogVideoXInpaintingPipeline(BasePipeline):
    def __init__(
        self,
        vae: AutoencoderKLCogVideoX,
        transformer: CogVideoXTransformer3DModel,
        scheduler: CogVideoXDPMScheduler,
        text_encoder=None,
        tokenizer=None,
        args=None,
    ):
        super().__init__(vae, transformer, scheduler, text_encoder, tokenizer)
        
        # Set processing parameters
        self.chunk_size = getattr(args, 'chunk_size', 64) if args else 64
        self.overlap = getattr(args, 'overlap', 8) if args else 8
        self.max_resolution = getattr(args, 'max_resolution', 2048) if args else 2048
        
        # Store original configs
        self._transformer_config = transformer.config
        self._vae_config = vae.config
    
    @property
    def transformer_config(self):
        """Get transformer config."""
        return self._transformer_config
    
    @property
    def vae_config(self):
        """Get VAE config."""
        return self._vae_config
    
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
    
    def validate_inputs(self, batch):
        """Validate input tensors."""
        try:
            # Check for required keys
            required_keys = ["rgb", "mask", "gt"]
            for key in required_keys:
                if key not in batch:
                    raise ValueError(f"Missing required key '{key}' in batch")
            
            # Get shapes
            rgb_shape = batch["rgb"].shape
            mask_shape = batch["mask"].shape
            gt_shape = batch["gt"].shape
            
            logger.info(f"=== Input Validation ===")
            logger.info(f"RGB shape: {rgb_shape}")
            logger.info(f"Mask shape: {mask_shape}")
            logger.info(f"GT shape: {gt_shape}")
            
            # Check dimensions
            if len(rgb_shape) != 5:
                raise ValueError(f"RGB tensor should have 5 dimensions [B,C,T,H,W], got {len(rgb_shape)}")
            if len(mask_shape) != 5:
                raise ValueError(f"Mask tensor should have 5 dimensions [B,C,T,H,W], got {len(mask_shape)}")
            if len(gt_shape) != 5:
                raise ValueError(f"GT tensor should have 5 dimensions [B,C,T,H,W], got {len(gt_shape)}")
            
            # Check batch size consistency
            if not (rgb_shape[0] == mask_shape[0] == gt_shape[0]):
                raise ValueError(f"Inconsistent batch sizes: RGB={rgb_shape[0]}, Mask={mask_shape[0]}, GT={gt_shape[0]}")
            
            # Check temporal dimension consistency
            if not (rgb_shape[2] == mask_shape[2] == gt_shape[2]):
                raise ValueError(f"Inconsistent temporal dimensions: RGB={rgb_shape[2]}, Mask={mask_shape[2]}, GT={gt_shape[2]}")
            
            # Check spatial dimensions
            if not (rgb_shape[3:] == mask_shape[3:] == gt_shape[3:]):
                raise ValueError(f"Inconsistent spatial dimensions: RGB={rgb_shape[3:]}, Mask={mask_shape[3:]}, GT={gt_shape[3:]}")
            
            # Check value ranges
            rgb_min, rgb_max = batch["rgb"].min(), batch["rgb"].max()
            mask_min, mask_max = batch["mask"].min(), batch["mask"].max()
            gt_min, gt_max = batch["gt"].min(), batch["gt"].max()
            
            logger.info(f"=== Value Ranges ===")
            logger.info(f"RGB range: [{rgb_min}, {rgb_max}]")
            logger.info(f"Mask range: [{mask_min}, {mask_max}]")
            logger.info(f"GT range: [{gt_min}, {gt_max}]")
            
            # Check for NaN/Inf values
            if torch.isnan(batch["rgb"]).any():
                raise ValueError("NaN values detected in RGB tensor")
            if torch.isnan(batch["mask"]).any():
                raise ValueError("NaN values detected in Mask tensor")
            if torch.isnan(batch["gt"]).any():
                raise ValueError("NaN values detected in GT tensor")
            
            if torch.isinf(batch["rgb"]).any():
                raise ValueError("Inf values detected in RGB tensor")
            if torch.isinf(batch["mask"]).any():
                raise ValueError("Inf values detected in Mask tensor")
            if torch.isinf(batch["gt"]).any():
                raise ValueError("Inf values detected in GT tensor")
            
            # Check device consistency
            if not (batch["rgb"].device == batch["mask"].device == batch["gt"].device):
                raise ValueError("Tensors must be on the same device")
            
            # Check dtype consistency
            if not (batch["rgb"].dtype == batch["mask"].dtype == batch["gt"].dtype):
                raise ValueError("Tensors must have the same dtype")
            
            logger.info("Input validation successful")
            return True
            
        except Exception as e:
            logger.error(f"Input validation failed: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            raise
    
    @torch.no_grad()
    def encode(self, x: torch.Tensor, chunk_size: Optional[int] = None, overlap: Optional[int] = None) -> torch.Tensor:
        """Encode video to latent space with temporal compression."""
        try:
            # Get dimensions and calculate temporal compression
            B, C, T, H, W = x.shape
            temporal_ratio = 4  # VAE's fixed temporal compression ratio
            expected_temporal_frames = T // temporal_ratio
            chunk_size = chunk_size or self.chunk_size
            overlap = overlap or self.overlap
            
            # Process in chunks if input is large
            if chunk_size is not None and W > chunk_size:
                # Calculate effective chunk size with overlap
                effective_chunk = chunk_size - 2 * overlap
                num_chunks = math.ceil(W / effective_chunk)
                chunks = []
                
                # Initialize final latents tensor with correct temporal dimension
                final_latents = torch.zeros(
                    B, self.transformer_config.in_channels, 
                    expected_temporal_frames, H//8, W//8,
                    device=x.device, dtype=torch.float16
                )
                weight_sum = torch.zeros_like(final_latents)
                
                for i in range(num_chunks):
                    # Clear cache before processing chunk
                    torch.cuda.empty_cache()
                    
                    # Calculate chunk boundaries with overlap
                    start_w = max(0, i * effective_chunk - overlap)
                    end_w = min((i + 1) * effective_chunk + overlap, W)
                    
                    # Process chunk
                    chunk = x[..., start_w:end_w]
                    chunk_latents = self.vae.encode(chunk).latent_dist.sample()
                    
                    # Verify temporal dimension
                    if chunk_latents.shape[2] != expected_temporal_frames:
                        logger.warning(f"Chunk temporal dim {chunk_latents.shape[2]} != expected {expected_temporal_frames}")
                        # Take center frames if needed
                        if chunk_latents.shape[2] > expected_temporal_frames:
                            start_idx = (chunk_latents.shape[2] - expected_temporal_frames) // 2
                            chunk_latents = chunk_latents[:, :, start_idx:start_idx + expected_temporal_frames]
                        else:
                            # Pad if chunk has fewer frames
                            pad_size = expected_temporal_frames - chunk_latents.shape[2]
                            chunk_latents = F.pad(chunk_latents, (0, 0, 0, 0, 0, pad_size))
                    
                    # Create blending weights
                    _, _, _, _, chunk_w = chunk_latents.shape
                    weight = torch.ones_like(chunk_latents)
                    
                    if i > 0 and overlap > 0:  # Left overlap
                        left_size = min(overlap // 8, chunk_w)  # Convert to latent space
                        if left_size > 0:
                            weight[..., :left_size] *= torch.linspace(0, 1, left_size, device=weight.device).view(1, 1, 1, 1, -1)
                    
                    if i < num_chunks - 1 and overlap > 0:  # Right overlap
                        right_size = min(overlap // 8, chunk_w)  # Convert to latent space
                        if right_size > 0:
                            weight[..., -right_size:] *= torch.linspace(1, 0, right_size, device=weight.device).view(1, 1, 1, 1, -1)
                    
                    # Add weighted chunk to final latents
                    start_idx = start_w // 8  # Convert to latent space
                    chunk_width = chunk_latents.shape[-1]
                    final_latents[..., start_idx:start_idx + chunk_width] += chunk_latents * weight
                    weight_sum[..., start_idx:start_idx + chunk_width] += weight
                    
                    # Clear chunk from memory
                    del chunk_latents
                    del weight
                    torch.cuda.empty_cache()
                
                # Normalize by weight sum
                final_latents = final_latents / (weight_sum + 1e-8)
                return final_latents
            
            # Process normally if input is small
            try:
                latents = self.vae.encode(x).latent_dist.sample()
                # Verify temporal dimension
                if latents.shape[2] != expected_temporal_frames:
                    logger.warning(f"VAE output temporal dim {latents.shape[2]} != expected {expected_temporal_frames}")
                    # Take center frames if needed
                    if latents.shape[2] > expected_temporal_frames:
                        start_idx = (latents.shape[2] - expected_temporal_frames) // 2
                        latents = latents[:, :, start_idx:start_idx + expected_temporal_frames]
                    else:
                        # Pad if we have fewer frames
                        pad_size = expected_temporal_frames - latents.shape[2]
                        latents = F.pad(latents, (0, 0, 0, 0, 0, pad_size))
                return latents
            
            except Exception as e:
                logger.error(f"Error in VAE encoding: {str(e)}")
                logger.error(f"VAE input shapes - RGB: {x.shape}")
                raise
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.warning("Out of memory in encode, trying with smaller chunks")
                torch.cuda.empty_cache()
                gc.collect()
                # Retry with smaller chunks
                chunk_size = chunk_size // 2 if chunk_size else self.chunk_size // 2
                return self.encode(x, chunk_size=chunk_size, overlap=overlap)
            raise
    
    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to pixel space."""
        try:
            # VAE decoding (fixed 8-frame output)
            decoded = self.vae.decode(latents).sample
            
            # Get input temporal dimension
            input_frames = latents.shape[2] * 4  # T//4 → T
            
            # Handle fixed 8-frame output from VAE
            if decoded.shape[2] != input_frames:
                # Take center frames if output is expanded
                start_idx = (decoded.shape[2] - input_frames) // 2
                decoded = decoded[:, :, start_idx:start_idx + input_frames]
            
            return decoded
            
        except Exception as e:
            logger.error(f"Error in VAE decoding: {str(e)}")
            logger.error(f"Latent input shape: {latents.shape}")
            raise
    
    def prepare_latents(
        self,
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Prepare random latents accounting for temporal and spatial compression."""
        # Get compression ratios from model configs
        temporal_ratio = self.transformer_config.temporal_compression_ratio
        vae_spatial_ratio = 8   # VAE's spatial compression ratio
        
        latents_shape = (
            batch_size, 
            self.transformer_config.in_channels, 
            num_frames//temporal_ratio,  # Temporal compression from transformer config
            height//vae_spatial_ratio,   # VAE spatial compression (8x)
            width//vae_spatial_ratio     # VAE spatial compression (8x)
        )
        latents = torch.randn(latents_shape, generator=generator, device=self.device, dtype=self.dtype)
        latents = latents * self.scheduler.init_noise_sigma
        return latents
    
    def prepare_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Prepare mask for latent space.
        
        Args:
            mask: Binary mask of shape [B, 1, T, H, W]
            
        Returns:
            Processed mask of shape [B, 1, T//temporal_ratio, H//8, W//8]
            where temporal_ratio is from transformer config and 8 is VAE's spatial compression
        """
        # Use transformer's temporal compression ratio and VAE's spatial ratio
        temporal_ratio = self.transformer_config.temporal_compression_ratio
        vae_spatial_ratio = 8   # VAE's spatial compression ratio
        
        mask = F.interpolate(mask, size=(
            mask.shape[2]//temporal_ratio,  # Temporal compression from transformer config
            mask.shape[3]//vae_spatial_ratio,   # VAE spatial compression (8x)
            mask.shape[4]//vae_spatial_ratio    # VAE spatial compression (8x)
        ), mode='nearest')
        
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
            return torch.randn(batch_size, 1, self.transformer_config.text_embed_dim, device=self.device, dtype=self.dtype)
            
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

    def check_boundary_continuity(self, video: torch.Tensor, boundary: int, window_size: int) -> Tuple[float, float]:
        """Check continuity at a chunk boundary.
        
        Args:
            video: Video tensor of shape [B, C, T, H, W]
            boundary: Boundary position in pixels (at center of chunk overlap)
            window_size: Size of window to check around boundary (in pixels)
            
        Returns:
            Tuple of (mean_diff, max_diff) at the boundary
        """
        # Ensure window is within bounds and has valid size
        left_start = max(0, boundary - window_size)
        left_end = boundary
        right_start = boundary
        right_end = min(video.shape[-1], boundary + window_size)
        
        # Ensure we have at least 1 pixel on each side
        if left_start >= left_end or right_start >= right_end:
            return 0.0, 0.0  # Skip if window is invalid
        
        # Get values around boundary
        left_vals = video[..., left_start:left_end].float()
        right_vals = video[..., right_start:right_end].float()
        
        # Reshape tensors to combine all dimensions except the last
        # Combine batch, channel, time, and height dimensions
        left_flat = left_vals.reshape(-1, left_vals.shape[-1])
        right_flat = right_vals.reshape(-1, right_vals.shape[-1])
        
        # Calculate mean values along the seam
        left_mean = left_flat.mean(dim=0)  # Average over all dimensions except width
        right_mean = right_flat.mean(dim=0)
        mean_diff = (right_mean - left_mean).abs().mean().item()
        
        # Calculate max difference along the seam
        diff_flat = (right_flat - left_flat).abs()
        max_diff = diff_flat.mean(dim=0).max().item()  # Max of the mean differences
        
        return mean_diff, max_diff

    def training_step(self, batch):
        """Training step for video inpainting."""
        try:
            # Clear memory before training step
            torch.cuda.empty_cache()
            gc.collect()
            
            # Validate input tensors
            self.validate_inputs(batch)
            
            total_frames = batch["rgb"].shape[2]  # [B, C, T, H, W]
            chunk_losses = []
            
            # Get compression ratios from model configs
            temporal_ratio = self.transformer_config.temporal_compression_ratio  # Usually 4
            spatial_ratio = 8  # VAE's fixed spatial compression ratio
            
            logger.info(f"=== Training Step Configuration ===")
            logger.info(f"Total frames: {total_frames}")
            logger.info(f"Temporal compression ratio: {temporal_ratio}")
            logger.info(f"Spatial compression ratio: {spatial_ratio}")
            logger.info(f"Initial chunk size: {self.chunk_size}")
            logger.info(f"Initial overlap: {self.overlap}")
            logger.info(f"Batch shape: {batch['rgb'].shape}")
            
            # Calculate minimum chunk size based on model requirements
            min_chunk_size = temporal_ratio * 2  # Need at least 2 frames after compression
            
            # Ensure chunk size is valid for temporal compression
            if self.chunk_size < min_chunk_size:
                logger.warning(f"Chunk size {self.chunk_size} is too small for temporal ratio {temporal_ratio}, increasing to {min_chunk_size}")
                self.chunk_size = min_chunk_size
            
            # Ensure chunk size is divisible by temporal ratio
            if self.chunk_size % temporal_ratio != 0:
                old_size = self.chunk_size
                self.chunk_size = ((self.chunk_size + temporal_ratio - 1) // temporal_ratio) * temporal_ratio
                logger.info(f"Adjusted chunk size from {old_size} to {self.chunk_size} to be divisible by temporal ratio {temporal_ratio}")
            
            # Calculate number of chunks with overlap
            if total_frames <= self.chunk_size:
                logger.info(f"Total frames {total_frames} <= chunk size {self.chunk_size}, processing as single chunk")
                num_chunks = 1
                effective_chunk_size = total_frames
                self.overlap = 0  # No overlap needed for single chunk
            else:
                # Ensure overlap is divisible by temporal ratio
                if self.overlap % temporal_ratio != 0:
                    old_overlap = self.overlap
                    self.overlap = (self.overlap // temporal_ratio) * temporal_ratio
                    logger.info(f"Adjusted overlap from {old_overlap} to {self.overlap} to be divisible by temporal ratio {temporal_ratio}")
                
                # Calculate stride between chunks
                stride = self.chunk_size - self.overlap
                if stride <= 0:
                    raise ValueError(f"Invalid stride: chunk_size={self.chunk_size} - overlap={self.overlap} <= 0")
                
                # Calculate number of chunks needed
                num_chunks = max(1, (total_frames - self.overlap + stride - 1) // stride)
            
            logger.info(f"=== Chunk Configuration ===")
            logger.info(f"Final chunk size: {self.chunk_size}")
            logger.info(f"Final overlap: {self.overlap}")
            logger.info(f"Stride between chunks: {self.chunk_size - self.overlap}")
            logger.info(f"Number of chunks: {num_chunks}")
            
            for chunk_idx in range(num_chunks):
                try:
                    # Calculate chunk boundaries
                    start_idx = chunk_idx * (self.chunk_size - self.overlap)
                    end_idx = min(start_idx + self.chunk_size, total_frames)
                    
                    logger.info(f"\n=== Processing Chunk {chunk_idx+1}/{num_chunks} ===")
                    logger.info(f"Start index: {start_idx}")
                    logger.info(f"End index: {end_idx}")
                    logger.info(f"Chunk frames: {end_idx - start_idx}")
                    
                    # Ensure chunk has enough frames for temporal compression
                    if (end_idx - start_idx) < min_chunk_size:
                        logger.warning(f"Skipping chunk {chunk_idx} - insufficient frames for temporal compression (got {end_idx - start_idx}, need {min_chunk_size})")
                        continue
                    
                    # Ensure chunk size is divisible by temporal ratio
                    chunk_frames = end_idx - start_idx
                    if chunk_frames % temporal_ratio != 0:
                        pad_frames = ((chunk_frames + temporal_ratio - 1) // temporal_ratio) * temporal_ratio - chunk_frames
                        logger.info(f"Padding chunk with {pad_frames} frames for temporal compression")
                        end_idx = min(end_idx + pad_frames, total_frames)
                        chunk_frames = end_idx - start_idx
                        logger.info(f"New chunk frames after padding: {chunk_frames}")
                    
                    # Get chunk data
                    chunk_rgb = batch["rgb"][:, :, start_idx:end_idx]
                    chunk_mask = batch["mask"][:, :, start_idx:end_idx]
                    chunk_gt = batch["gt"][:, :, start_idx:end_idx]
                    
                    logger.info(f"=== Chunk Tensor Shapes ===")
                    logger.info(f"RGB shape: {chunk_rgb.shape}")
                    logger.info(f"Mask shape: {chunk_mask.shape}")
                    logger.info(f"GT shape: {chunk_gt.shape}")
                    
                    # Pad spatial dimensions for VAE
                    chunk_rgb, rgb_pad = pad_to_multiple(chunk_rgb, multiple=spatial_ratio * 8, max_dim=self.max_resolution)
                    chunk_mask, _ = pad_to_multiple(chunk_mask, multiple=spatial_ratio * 8, max_dim=self.max_resolution)
                    chunk_gt, _ = pad_to_multiple(chunk_gt, multiple=spatial_ratio * 8, max_dim=self.max_resolution)
                    
                    logger.info(f"=== After Spatial Padding ===")
                    logger.info(f"RGB shape: {chunk_rgb.shape}")
                    logger.info(f"Mask shape: {chunk_mask.shape}")
                    logger.info(f"GT shape: {chunk_gt.shape}")
                    logger.info(f"Padding amounts: {rgb_pad}")
                    
                    # Process with VAE
                    try:
                        rgb_latents = self.encode(chunk_rgb)
                        gt_latents = self.encode(chunk_gt)
                        logger.info(f"=== VAE Encoding ===")
                        logger.info(f"RGB latents shape: {rgb_latents.shape}")
                        logger.info(f"GT latents shape: {gt_latents.shape}")
                    except Exception as e:
                        logger.error(f"Error in VAE encoding: {str(e)}")
                        logger.error(f"VAE input shapes - RGB: {chunk_rgb.shape}, GT: {chunk_gt.shape}")
                        raise
                    
                    # Sample timesteps
                    timesteps = torch.randint(
                        0, self.scheduler.config.num_train_timesteps,
                        (chunk_rgb.shape[0],), device=chunk_rgb.device
                    )
                    
                    # Get encoder hidden states
                    encoder_hidden_states = torch.zeros(
                        chunk_rgb.shape[0], 
                        chunk_rgb.shape[2] // temporal_ratio,  # Compress temporal dimension
                        self.transformer_config.hidden_size,
                        device=chunk_rgb.device,
                        dtype=chunk_rgb.dtype
                    )
                    
                    logger.info(f"=== Model Inputs ===")
                    logger.info(f"Timesteps: {timesteps}")
                    logger.info(f"Hidden states shape: {encoder_hidden_states.shape}")
                    
                    # Add noise and get prediction
                    try:
                        noise = torch.randn_like(rgb_latents)
                        noisy_latents = self.scheduler.add_noise(rgb_latents, noise, timesteps)
                        model_pred = self.transformer(
                            noisy_latents,
                            timesteps,
                            encoder_hidden_states=encoder_hidden_states,
                        ).sample
                        
                        logger.info(f"=== Model Prediction ===")
                        logger.info(f"Noise shape: {noise.shape}")
                        logger.info(f"Noisy latents shape: {noisy_latents.shape}")
                        logger.info(f"Model prediction shape: {model_pred.shape}")
                    except Exception as e:
                        logger.error(f"Error in transformer forward pass: {str(e)}")
                        logger.error(f"Input shapes - Noisy latents: {noisy_latents.shape}, Hidden states: {encoder_hidden_states.shape}")
                        raise
                    
                    # Compute loss
                    try:
                        chunk_loss = compute_loss_v_pred_with_snr(
                            model_pred, noise, timesteps, self.scheduler,
                            mask=chunk_mask, noisy_frames=noisy_latents
                        )
                        chunk_losses.append(chunk_loss)
                        logger.info(f"Chunk {chunk_idx+1} loss: {chunk_loss.item()}")
                    except Exception as e:
                        logger.error(f"Error computing loss: {str(e)}")
                        logger.error(f"Loss inputs - Pred: {model_pred.shape}, Noise: {noise.shape}, Mask: {chunk_mask.shape}")
                        raise
                    
                except Exception as e:
                    logger.error(f"Error processing chunk {chunk_idx}: {str(e)}")
                    logger.error(f"Stack trace: {traceback.format_exc()}")
                    continue
            
            if not chunk_losses:
                logger.error("No chunks were successfully processed")
                raise RuntimeError("No chunks were successfully processed")
            
            # Average losses across chunks
            loss = torch.stack(chunk_losses).mean()
            logger.info(f"=== Final Loss ===")
            logger.info(f"Number of successful chunks: {len(chunk_losses)}")
            logger.info(f"Average loss: {loss.item()}")
            
            return loss
            
        except Exception as e:
            logger.error(f"Error in training step: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
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

def gelu_approximate(x):
    """Approximate GELU activation function."""
    return x * 0.5 * (1.0 + torch.tanh(0.7978845608028654 * x * (1 + 0.044715 * x * x)))

def handle_vae_temporal_output(decoded, target_frames):
    """Handle potential temporal expansion from VAE.
    
    Args:
        decoded: Tensor from VAE decoder with shape [B, C, T, H, W]
        target_frames: Number of frames expected in output
        
    Returns:
        Tensor with target number of frames [B, C, target_frames, H, W]
    """
    if decoded.shape[2] != target_frames:
        # Take center frames if output is expanded
        start_idx = (decoded.shape[2] - target_frames) // 2
        decoded = decoded[:, :, start_idx:start_idx + target_frames]
        
    assert decoded.shape[2] == target_frames, \
        f"VAE output frames {decoded.shape[2]} doesn't match target frames {target_frames}"
    
    return decoded

def apply_rotary_pos_emb(x, cos, sin, position_ids):
    """Apply rotary position embeddings to input tensor."""
    # Rotary embeddings
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    
    # Apply rotation
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed

def rotate_half(x):
    """Rotate half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def create_layer_norm(hidden_dim, model_config, device, dtype):
    """Create layer normalization with correct configuration."""
    return nn.LayerNorm(
        hidden_dim,
        eps=model_config.norm_eps,
        elementwise_affine=model_config.norm_elementwise_affine,
        device=device,
        dtype=dtype
    )

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
    
    # Get model dtype and config
    model_dtype = next(model.parameters()).dtype
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing = True
    assert model.is_gradient_checkpointing, "Gradient checkpointing should be enabled"
    
    # Verify model configuration
    assert model.config.use_rotary_positional_embeddings, "Rotary embeddings should be enabled"
    assert not model.config.use_learned_positional_embeddings, "Learned embeddings should be disabled"
    assert model.config.activation_fn == "gelu-approximate", "Unexpected activation function"
    assert model.config.timestep_activation_fn == "silu", "Unexpected timestep activation function"
    assert model.config.norm_elementwise_affine, "Layer norm should use elementwise affine"
    assert model.config.norm_eps == 1e-5, f"Unexpected norm epsilon: {model.config.norm_eps}"
    
    # VAE has fixed 8-frame output and 8x spatial downsampling
    vae_spatial_ratio = 8
    target_frames = model.config.sample_frames  # Use model's native frame count
    
    # Create rotary embedding cache
    max_position_embeddings = 512
    base = 10000
    inv_freq = 1.0 / (base ** (torch.arange(0, model.config.attention_head_dim, 2).float().to(model.device) / model.config.attention_head_dim))
    
    # Create layer norm for hidden states
    hidden_norm = create_layer_norm(
        model.patch_embed.proj.out_channels,  # Use output channels from patch embedding
        model.config,
        model.device,
        model_dtype
    )
    
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
                # Get input tensors and ensure correct dtype
                clean_frames = batch["rgb"].to(dtype=model_dtype)  # [B, C, T, H, W]
                mask = batch["mask"].to(dtype=model_dtype)
                
                # Validate input dimensions
                B, C, T, H, W = clean_frames.shape
                assert H % vae_spatial_ratio == 0 and W % vae_spatial_ratio == 0, \
                    f"Input dimensions ({H}, {W}) must be divisible by VAE ratio {vae_spatial_ratio}"
                
                # Handle spatial downsampling
                clean_frames = F.interpolate(
                    clean_frames,
                    size=(T, H//vae_spatial_ratio, W//vae_spatial_ratio),
                    mode='bilinear',
                    align_corners=False
                )
                
                # Test VAE encoding/decoding and handle temporal expansion
                latent = vae.encode(clean_frames).latent_dist.sample()
                decoded = vae.decode(latent).sample
                
                # Handle fixed 8-frame VAE output
                decoded = handle_vae_temporal_output(decoded, T)
                
                # Verify no NaN values from VAE
                assert not torch.isnan(latent).any(), "VAE latent contains NaN values"
                assert not torch.isnan(decoded).any(), "VAE output contains NaN values"
                
                # Convert to [B, T, C, H, W] format for transformer
                clean_frames = clean_frames.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
                
                # Create encoder hidden states
                encoder_hidden_states = torch.zeros(
                    clean_frames.shape[0], 1, model.config.text_embed_dim,
                    device=clean_frames.device,
                    dtype=model_dtype
                )
                
                # Get model prediction
                noise_pred = model(
                    hidden_states=clean_frames,
                    timestep=torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (clean_frames.shape[0],), 
                        device=clean_frames.device
                    ),
                    encoder_hidden_states=encoder_hidden_states,
                ).sample
                
                # Verify no NaN values from activations or normalization
                if torch.isnan(noise_pred).any():
                    raise ValueError("Model output contains NaN values - possible activation or normalization issue")
                
                # Verify shape consistency
                if noise_pred.shape[:3] != clean_frames.shape[:3]:
                    raise ValueError(f"Model output shape {noise_pred.shape} doesn't match input shape {clean_frames.shape} in batch, temporal, or channel dimensions")
                
                # Verify spatial dimensions after patch embedding
                H_in, W_in = clean_frames.shape[3:]
                H_out = H_in - (H_in % model.config.patch_size)  # Round down to nearest multiple of patch_size
                W_out = W_in - (W_in % model.config.patch_size)  # Round down to nearest multiple of patch_size
                
                if noise_pred.shape[3:] != (H_out, W_out):
                    raise ValueError(f"Expected spatial dimensions ({H_out}, {W_out}), got {noise_pred.shape[3:]}")
                
                # Compute loss with SNR rescaling
                loss = compute_loss_v_pred_with_snr(
                    noise_pred, torch.randn_like(clean_frames), 
                    torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (clean_frames.shape[0],), 
                        device=clean_frames.device
                    ), noise_scheduler,
                    mask=mask, noisy_frames=clean_frames
                )
                
                # Verify loss is valid
                if torch.isnan(loss).any():
                    raise ValueError("Loss contains NaN values - possible activation or normalization issue")
                
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
                            val_loss += compute_loss_v_pred_with_snr(noise_pred, noise, timesteps, noise_scheduler, mask=mask, noisy_frames=clean_frames).item()
                    
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
        torch_dtype=torch.bfloat16 if "5b" in args.pretrained_model_name_or_path.lower() else torch.float16
    )
    
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=torch.float16 if args.mixed_precision == "fp16" else (torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32)
    )
    
    scheduler = CogVideoXDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    
    # Update scheduler config
    scheduler.config.prediction_type = "v_prediction"
    scheduler.config.rescale_betas_zero_snr = True
    scheduler.config.snr_shift_scale = 1.0
    
    # Enable memory optimizations
    if args.gradient_checkpointing:
        transformer.gradient_checkpointing = True
    
    if args.enable_xformers_memory_efficient_attention:
        transformer.enable_xformers_memory_efficient_attention()
    
    # Create pipeline with unwrapped models
    pipeline = CogVideoXInpaintingPipeline(
        vae=vae,
        transformer=transformer,
        scheduler=scheduler,
        args=args,
    )
    
    # Prepare models for distributed training
    vae, transformer = accelerator.prepare(vae, transformer)
    
    # Update pipeline with wrapped models
    pipeline.vae = vae
    pipeline.transformer = transformer
    
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
    
    # Load models
    vae = AutoencoderKLCogVideoX.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.bfloat16 if "5b" in args.pretrained_model_name_or_path.lower() else torch.float16
    )
    
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=torch.float16 if args.mixed_precision == "fp16" else (torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32)
    )
    
    scheduler = CogVideoXDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    
    # Update scheduler config
    scheduler.config.prediction_type = "v_prediction"
    scheduler.config.rescale_betas_zero_snr = True
    scheduler.config.snr_shift_scale = 1.0
    
    # Enable memory optimizations
    if args.gradient_checkpointing:
        transformer.gradient_checkpointing = True
    
    if args.enable_xformers_memory_efficient_attention:
        transformer.enable_xformers_memory_efficient_attention()
    
    # Create pipeline
    pipeline = CogVideoXInpaintingPipeline(
        vae=vae,
        transformer=transformer,
        scheduler=scheduler,
        args=args,
    )
    
    # Dataset and DataLoaders creation
    train_dataset = VideoInpaintingDataset(
        data_root=args.data_root,
        video_dir=args.video_dir,
        mask_dir=args.mask_dir,
        gt_dir=args.gt_dir,
        image_size=args.image_size,
        num_frames=args.max_num_frames,
        center_crop=True,
        normalize=True
    )
    
    val_dataset = VideoInpaintingDataset(
        data_root=args.data_root,
        video_dir=args.video_dir,
        mask_dir=args.mask_dir,
        gt_dir=args.gt_dir,
        image_size=args.image_size,
        num_frames=args.max_num_frames,
        center_crop=True,
        normalize=True
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
    
    # Calculate training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_warmup_steps = args.lr_warmup_steps * args.gradient_accumulation_steps
    args.num_training_steps = args.max_train_steps * args.gradient_accumulation_steps
    
    # Optimizer
    if args.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            transformer.parameters(),
            lr=args.learning_rate,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            eps=args.epsilon,
        )
    else:
        optimizer = torch.optim.AdamW(
            transformer.parameters(),
            lr=args.learning_rate,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            eps=args.epsilon,
        )
    
    # Prepare for distributed training
    pipeline, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        pipeline, optimizer, train_dataloader, val_dataloader
    )
    
    # Get scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_training_steps,
    )
    
    # Store original configs
    pipeline._transformer_config = transformer.config
    pipeline._vae_config = vae.config
    
    # Initialize training state
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Number of warmup steps = {args.num_warmup_steps}")
    
    global_step = 0
    first_epoch = 0
    
    # Get validation data
    try:
        validation_data = next(iter(val_dataloader))
    except:
        validation_data = None
        logger.warning("No validation data available")
    
    # Training loop
    for epoch in range(first_epoch, args.num_train_epochs):
        pipeline.transformer.train()
        train_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            # Convert inputs to correct dtype
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(dtype=torch.float32 if not args.mixed_precision else (torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16))
            
            with accelerator.accumulate(pipeline.transformer):
                loss = pipeline.training_step(batch)
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(pipeline.transformer.parameters(), args.max_grad_norm)
                    
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Logging
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                train_loss += loss.detach().item()
                
                if global_step % args.logging_steps == 0:
                    accelerator.log(
                        {
                            "train_loss": train_loss / args.logging_steps,
                            "learning_rate": lr_scheduler.get_last_lr()[0],
                            "epoch": epoch,
                            "global_step": global_step,
                        },
                        step=global_step,
                    )
                    train_loss = 0.0
                
                if global_step % args.validation_steps == 0:
                    if validation_data is not None:
                        log_validation(accelerator, pipeline, args, epoch, validation_data)
                
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        
                if global_step >= args.max_train_steps:
                    break
    
    # Save final model
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, "final_model")
        accelerator.save_state(save_path)

if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = str(Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    from args import get_args
    args = get_args()
    args.window_size = getattr(args, 'window_size', 32)
    args.overlap = getattr(args, 'overlap', 8)
    args.chunk_size = getattr(args, 'chunk_size', 32)
    args.use_8bit_adam = getattr(args, 'use_8bit_adam', False)
    args.use_flash_attention = getattr(args, 'use_flash_attention', False)
    args.vae_precision = getattr(args, 'vae_precision', "fp16")
    args.max_resolution = getattr(args, 'max_resolution', 2048)
    main(args)