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
from transformers import CLIPTextModelWithProjection

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
    """
    Unwrap a model from its distributed wrapper.
    
    Args:
        model: The model to unwrap
        
    Returns:
        The unwrapped model
    """
    if hasattr(model, "module"):
        return model.module
    return model

class CogVideoXInpaintingPipeline:
    """Pipeline for training CogVideoX on video inpainting."""
    
    def __init__(
        self,
        vae,
        transformer,
        scheduler,
        device=None,
        dtype=None
    ):
        """Initialize the pipeline."""
        self.vae = vae
        self.transformer = transformer
        self.noise_scheduler = scheduler
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Configure precision based on model type
        model_name = getattr(transformer.config, "_name_or_path", "").lower()
        self.dtype = dtype or (torch.bfloat16 if "5b" in model_name else torch.float16)
        self.weight_dtype = self.dtype
        
        # Move models to device and dtype
        self.vae = self.vae.to(self.device, dtype=self.dtype)
        self.transformer = self.transformer.to(self.device, dtype=self.dtype)
        
        # Ensure transformer's parameters are in correct dtype
        for param in self.transformer.parameters():
            param.data = param.data.to(dtype=self.dtype)
        
        # Freeze VAE
        self.vae.requires_grad_(False)
        self.vae.eval()
        
        # Set processing parameters
        self.chunk_size = 32
        self.overlap = 4
        self.max_resolution = 512
        
        # Set memory optimization
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.8,expandable_segments:True"
        
        # Enable gradient checkpointing for transformer
        if hasattr(self.transformer, 'gradient_checkpointing'):
            self.transformer.gradient_checkpointing = True
        
        # Enable memory efficient attention
        if hasattr(self.transformer, 'set_use_memory_efficient_attention_xformers'):
            self.transformer.set_use_memory_efficient_attention_xformers(True)
        
        # Enable VAE optimizations
        if hasattr(self.vae, 'enable_slicing'):
            self.vae.enable_slicing()
        if hasattr(self.vae, 'enable_tiling'):
            self.vae.enable_tiling()
    
    @property
    def transformer_config(self):
        """Get transformer config."""
        return self.transformer.config
    
    @property
    def vae_config(self):
        """Get VAE config."""
        return self.vae.config
    
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
            
            # Check mask channel dimension
            if mask_shape[1] != 1:
                raise ValueError(f"Mask should have 1 channel, got {mask_shape[1]}")
            
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
            
            # Validate mask values are binary (0 or 1)
            if not torch.all(torch.logical_or(batch["mask"] == 0, batch["mask"] == 1)):
                raise ValueError("Mask values must be binary (0 or 1)")
            
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
            
            # Validate spatial dimensions are divisible by 8 (VAE downsampling factor)
            if rgb_shape[3] % 8 != 0 or rgb_shape[4] % 8 != 0:
                raise ValueError(f"Spatial dimensions must be divisible by 8, got {rgb_shape[3:]} (H, W)")
            
            logger.info("Input validation successful")
            return True
            
        except Exception as e:
            logger.error(f"Input validation failed: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            raise
    
    @torch.no_grad()
    def encode(self, frames, return_dict=True):
        """
        Encode input frames to latent space without text conditioning.
        
        Args:
            frames: Input video frames tensor [B, C, T, H, W]
            return_dict: Whether to return a dictionary
            
        Returns:
            Encoded latent representation
        """
        batch_size = frames.shape[0]
        
        try:
            # Process frames in temporal chunks for memory efficiency
            latents = []
            for chunk_start in range(0, frames.shape[2], self.chunk_size):
                chunk_end = min(chunk_start + self.chunk_size, frames.shape[2])
                chunk = frames[:, :, chunk_start:chunk_end]
                
                # Encode chunk
                with torch.no_grad():
                    latent_dist = self.vae.encode(chunk)
                    if isinstance(latent_dist, DiagonalGaussianDistribution):
                        chunk_latents = latent_dist.sample()
                    else:
                        chunk_latents = latent_dist
                    
                latents.append(chunk_latents)
            
            # Concatenate chunks
            latents = torch.cat(latents, dim=2)
            
            # Scale latents
            latents = latents * self.vae.config.scaling_factor
            
            if return_dict:
                return {"latents": latents}
            return latents
            
        except Exception as e:
            logger.error(f"Error in encode method: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to pixel space."""
        try:
            # VAE decoding (fixed 8-frame output)
            decoded = self.vae.decode(latents).sample
            
            # Get input temporal dimension
            input_frames = latents.shape[2] * self.transformer.config.temporal_compression_ratio
            
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
        temporal_ratio = self.transformer.config.temporal_compression_ratio
        vae_spatial_ratio = 8   # VAE's spatial compression ratio
        
        latents_shape = (
            batch_size, 
            self.transformer.config.in_channels, 
            num_frames//temporal_ratio,  # Temporal compression from transformer config
            height//vae_spatial_ratio,   # VAE spatial compression (8x)
            width//vae_spatial_ratio     # VAE spatial compression (8x)
        )
        latents = torch.randn(latents_shape, generator=generator, device=self.device, dtype=self.dtype)
        latents = latents * self.noise_scheduler.init_noise_sigma
        return latents
    
    def prepare_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Prepare mask for latent space.
        
        Args:
            mask: Binary mask of shape [B, 1, T, H, W]
            
        Returns:
            Processed mask of shape [B, 1, T//temporal_ratio, H//8, W//8]
            where temporal_ratio is from transformer config and 8 is VAE's spatial compression
        """
        # Validate input mask is binary
        if not torch.all(torch.logical_or(mask == 0, mask == 1)):
            raise ValueError("Input mask must contain only binary values (0 or 1)")
        
        # Use transformer's temporal compression ratio and VAE's spatial ratio
        temporal_ratio = self.transformer.config.temporal_compression_ratio
        vae_spatial_ratio = 8   # VAE's spatial compression ratio
        
        mask = mask.permute(0, 1, 2, 3, 4)
        # Interpolate spatial dimensions only by reshaping to combine batch and time dims
        mask = F.interpolate(
            mask.reshape(-1, mask.shape[1], *mask.shape[-2:]),  # Combine batch and time dims
            size=(16, 16),
            mode="nearest"  # Use nearest neighbor to preserve binary values
        ).reshape(mask.shape[0], mask.shape[1], mask.shape[2], 16, 16)  # Restore original shape
        
        # Ensure mask remains binary after interpolation
        mask = (mask > 0.5).float()
        
        # Validate output mask is binary
        if not torch.all(torch.logical_or(mask == 0, mask == 1)):
            raise ValueError("Mask contains non-binary values after processing")
        
        return mask
    
    def prepare_encoder_hidden_states(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        """
        Prepare zero tensors for conditioning the transformer.
        
        Args:
            batch_size: Number of samples in the batch
            device: Device to put the tensors on
            dtype: Data type of the tensors
        
        Returns:
            Zero tensor of shape (batch_size, 1, 4096) - matches CogVideoX-5b hidden size
        """
        hidden_size = 4096  # Fixed size for CogVideoX-5b
        encoder_hidden_states = torch.zeros(
            (batch_size, 1, hidden_size),
            device=device,
            dtype=dtype
        )
        
        # Validate shape and type
        assert encoder_hidden_states.shape == (batch_size, 1, hidden_size), \
            f"Incorrect hidden states shape: {encoder_hidden_states.shape}"
        assert encoder_hidden_states.dtype == dtype, \
            f"Incorrect hidden states dtype: {encoder_hidden_states.dtype}"
        
        return encoder_hidden_states
    
    @torch.no_grad()
    def __call__(
        self,
        frames,
        mask,
        generator=None,
        num_inference_steps=50,
        eta=0.0,
        guidance_scale=1.0,
        output_type="pil",
        return_dict=True,
    ):
        """
        Perform inpainting on video frames.
        
        Args:
            frames: Input video frames tensor [B, C, T, H, W]
            mask: Binary mask tensor [B, 1, T, H, W]
            generator: Random number generator
            num_inference_steps: Number of denoising steps
            eta: Weight for noise in each step
            guidance_scale: Scale for classifier-free guidance
            output_type: Output format ('pil' or 'pt')
            return_dict: Whether to return a dictionary
            
        Returns:
            Generated inpainted video frames
        """
        # Initialize
        device = self.device
        do_classifier_free_guidance = guidance_scale > 1.0
        
        # Prepare latent variables
        latents = self.encode(frames)["latents"]
        
        # Set timesteps
        self.noise_scheduler.set_timesteps(num_inference_steps)
        timesteps = self.noise_scheduler.timesteps
        
        # Add noise to latents
        noise = torch.randn(latents.shape, generator=generator, device=device)
        latents = self.noise_scheduler.add_noise(latents, noise, timesteps[0])
        
        # Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        
        # Denoising loop
        for i, t in enumerate(timesteps):
            # Expand latents for guidance
            latent_model_input = latents
            if do_classifier_free_guidance:
                latent_model_input = torch.cat([latent_model_input] * 2)
            
            # Predict noise residual
            noise_pred = self.transformer(
                hidden_states=latent_model_input,  
                timestep=t.to(dtype=latent_model_input.dtype),
                encoder_hidden_states=None,  
            ).sample  
            
            # Perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Compute previous noisy sample
            latents = self.noise_scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
            
            # Apply mask
            if mask is not None:
                init_latents = self.encode(frames)["latents"]
                latents = (1 - mask) * init_latents + mask * latents
        
        # Decode latents
        video_frames = self.decode(latents)
        
        # Convert to output format
        if output_type == "pil":
            video_frames = self.numpy_to_pil(video_frames)
        
        if not return_dict:
            return video_frames
        
        return {"videos": video_frames}
    
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
        """Execute a training step on a batch of inputs."""
        # Get clean frames and mask
        clean_frames = batch["rgb"]
        mask = batch["mask"]
        
        # Validate inputs
        self.validate_inputs(batch)
        
        # Get batch size and dimensions
        batch_size = clean_frames.shape[0]
        num_frames = clean_frames.shape[2]
        
        # Sample timesteps
        timesteps = self.noise_scheduler.sample_timesteps(batch_size, device=clean_frames.device)
        timesteps = timesteps.to(device=self.device, dtype=self.dtype)  # Ensure timesteps match model dtype
        
        # Get latents
        clean_latents = self.encode(clean_frames).latents
        clean_latents = clean_latents.to(dtype=self.dtype)  # Ensure latents match model dtype
        
        # Add noise to latents
        noise = torch.randn_like(clean_latents, device=clean_latents.device, dtype=self.dtype)  # Generate noise in correct dtype
        noisy_latents = self.noise_scheduler.add_noise(clean_latents, noise, timesteps)
        
        # Process in smaller chunks with gradient disabled for VAE
        frames = batch["rgb"].to(self.weight_dtype)  # [B, C, T, H, W]
        mask = batch["mask"].to(self.weight_dtype)   # [B, 1, T, H, W]
        mask = (mask > 0.5).to(self.weight_dtype)
        gt = batch["gt"].to(self.weight_dtype)       # [B, C, T, H, W]
        
        chunk_size = 1  
        latents_list = []
        
        # Split frames into chunks
        for i in range(0, frames.shape[0], chunk_size):
            chunk = frames[i:i+chunk_size]
            torch.cuda.empty_cache()
            
            # Move chunk to CPU after processing if needed
            with torch.no_grad():
                latents_chunk = self.vae.encode(chunk).latent_dist.sample()
                latents_chunk = latents_chunk * self.vae.config.scaling_factor
                latents_list.append(latents_chunk.cpu() if i + chunk_size < frames.shape[0] else latents_chunk)
            del chunk
            torch.cuda.empty_cache()
        
        # Concatenate all chunks
        if len(latents_list) > 1:
            latents = torch.cat([chunk.cuda() if chunk.device.type == 'cpu' else chunk for chunk in latents_list], dim=0)
        else:
            latents = latents_list[0]
        del latents_list
        torch.cuda.empty_cache()
        
        # Get noise
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (frames.shape[0],), device=latents.device)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Free up memory
        del frames
        torch.cuda.empty_cache()
        
        # Prepare mask for latent space
        # Downsample mask to match latent dimensions
        B, _, T, H, W = mask.shape
        latent_h = H // 8  # VAE spatial reduction
        latent_w = W // 8
        latent_t = noisy_latents.shape[2]  # Use temporal dimension from noisy_latents
        
        # First ensure mask is binary
        mask = (mask > 0.5).float()
        
        # Use nearest neighbor interpolation to preserve binary values
        mask_latent = F.interpolate(
            mask.view(B, 1, T, H, W),
            size=(latent_t, latent_h, latent_w),
            mode='nearest'
        )
        
        # Re-binarize just to be safe
        mask_latent = (mask_latent > 0.5).float()
        
        # Convert to [B, T, C, H, W] format for transformer
        noisy_frames = noisy_latents.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
        
        # Print shapes for debugging
        print(f"noisy_frames shape: {noisy_frames.shape}")
        print(f"mask_latent shape before permute: {mask_latent.shape}")
        
        # Expand mask to match noisy_frames channel dimension
        C = noisy_frames.shape[2]  # Get number of channels from noisy_frames
        mask_latent = mask_latent.permute(0, 2, 1, 3, 4)  # [B, T, 1, H, W]
        
        print(f"mask_latent shape after permute: {mask_latent.shape}")
        print(f"Expanding mask to {C} channels")
        
        mask_latent = mask_latent.expand(-1, -1, C, -1, -1)  # Expand to match channels
        
        print(f"mask_latent shape after expand: {mask_latent.shape}")
        print(f"Attempting to concatenate along dim 2 (channel dim)")
        
        # Concatenate mask with noisy frames along channel dimension
        transformer_input = torch.cat([noisy_frames, mask_latent], dim=2)
        
        # Create dummy encoder hidden states (no text conditioning)
        batch_size = transformer_input.shape[0]
        device = transformer_input.device
        dtype = transformer_input.dtype
        encoder_hidden_states = torch.zeros(batch_size, 1, self.transformer.config.text_embed_dim, device=device, dtype=dtype)
        
        # Predict noise
        noise_pred = self.transformer(
            hidden_states=transformer_input,
            timestep=timesteps.to(dtype=transformer_input.dtype),
            encoder_hidden_states=encoder_hidden_states,
        ).sample
        
        # Convert predictions back to scheduler format [B, C, T, H, W]
        noise_pred_scheduler = noise_pred.permute(0, 2, 1, 3, 4)
        
        # Compute loss
        loss = F.mse_loss(noise_pred_scheduler.float(), noise.float(), reduction="mean")
        
        # Free up memory
        del noise_pred, noisy_latents, noise, transformer_input
        torch.cuda.empty_cache()
        
        return loss
    
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
        model.patch_embed.proj.out_channels,  
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
                clean_frames = batch["rgb"].to(device=model.device, dtype=model_dtype)  
                mask = batch["mask"].to(device=model.device, dtype=model_dtype)
                mask = (mask > 0.5).to(model_dtype)
                
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
                
                # Convert to [B, T, C, H, W] format for transformer
                clean_frames = clean_frames.permute(0, 2, 1, 3, 4)  
                
                # Create dummy encoder hidden states (4096 is CogVideoX-5b hidden size)
                encoder_hidden_states = torch.zeros((B, 1, 4096), device=model.device, dtype=model_dtype)
                
                # Sample timesteps and ensure correct dtype
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (B,),
                    device=model.device
                ).to(dtype=model_dtype)
                
                # Add noise
                noise = torch.randn_like(clean_frames)
                noisy_frames = noise_scheduler.add_noise(clean_frames, noise, timesteps)
                
                # Get model prediction
                image_rotary_emb = (
                    prepare_rotary_positional_embeddings(
                        height=clean_frames.shape[3] * 8,  
                        width=clean_frames.shape[4] * 8,
                        num_frames=clean_frames.shape[2],
                        vae_scale_factor_spatial=8,  
                        patch_size=model.config.patch_size,
                        attention_head_dim=model.config.attention_head_dim,
                        device=model.device,
                    )
                    if model.config.use_rotary_positional_embeddings
                    else None
                )
                
                # Create dummy text embeddings for the patch embedding layer
                batch_size = noisy_frames.shape[0]
                dummy_text_embeds = torch.zeros((batch_size, 1, 4096), device=model.device, dtype=model_dtype)
                
                model_output = model(
                    hidden_states=noisy_frames,
                    timestep=timesteps,  # No need to cast timesteps again since we did it earlier
                    encoder_hidden_states=dummy_text_embeds,  
                    image_rotary_emb=image_rotary_emb,
                    return_dict=True
                ).sample  
                noise_pred = model_output
                
                # Verify shape consistency
                assert noise_pred.shape == noisy_frames.shape, \
                    f"Model output shape {noise_pred.shape} doesn't match input shape {noisy_frames.shape}"
                
                # Compute loss with SNR rescaling
                loss = compute_loss_v_pred_with_snr(
                    noise_pred, noise, timesteps, noise_scheduler,
                    mask=mask.permute(0, 2, 1, 3, 4),  
                    noisy_frames=clean_frames
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
                            mask = (mask > 0.5).to(model_dtype)
                            
                            noise = torch.randn_like(clean_frames)
                            timesteps = torch.randint(
                                0, noise_scheduler.config.num_train_timesteps, (clean_frames.shape[0],),
                                device=clean_frames.device
                            )
                            noisy_frames = noise_scheduler.add_noise(clean_frames, noise, timesteps)
                            
                            image_rotary_emb = (
                                prepare_rotary_positional_embeddings(
                                    height=clean_frames.shape[3] * 8,  
                                    width=clean_frames.shape[4] * 8,
                                    num_frames=clean_frames.shape[2],
                                    vae_scale_factor_spatial=8,  
                                    patch_size=model.config.patch_size,
                                    attention_head_dim=model.config.attention_head_dim,
                                    device=clean_frames.device,
                                )
                                if model.config.use_rotary_positional_embeddings
                                else None
                            )
                            
                            # Create dummy text embeddings for the patch embedding layer
                            batch_size = noisy_frames.shape[0]
                            device = noisy_frames.device
                            dtype = noisy_frames.dtype
                            dummy_text_embeds = torch.zeros((batch_size, 1, 4096), device=device, dtype=dtype)
                            
                            model_output = model(
                                hidden_states=noisy_frames,
                                timestep=timesteps.to(dtype=noisy_frames.dtype),
                                encoder_hidden_states=dummy_text_embeds,  
                                image_rotary_emb=image_rotary_emb,
                                return_dict=True
                            ).sample  
                            noise_pred = model_output
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

def train_one_epoch(
    args,
    accelerator,
    vae,
    transformer,
    optimizer,
    scheduler,
    train_dataloader,
    weight_dtype,
    epoch,
):
    transformer.train()
    vae.eval()  
    
    # Enable gradient checkpointing for transformer
    if hasattr(transformer, "enable_gradient_checkpointing"):
        transformer.enable_gradient_checkpointing()
        if hasattr(transformer, 'set_use_memory_efficient_attention_xformers'):
            transformer.set_use_memory_efficient_attention_xformers(True)

    # Enable memory efficient attention if available
    if hasattr(transformer, "set_use_memory_efficient_attention_xformers"):
        transformer.set_use_memory_efficient_attention_xformers(True)
    
    progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
    progress_bar.set_description(f"Epoch {epoch}")
    
    for step, batch in enumerate(train_dataloader):
        # Reset memory before processing each batch
        gc.collect()
        torch.cuda.empty_cache()
        
        with accelerator.accumulate(transformer):
            with torch.cuda.amp.autocast(device_type='cuda', enabled=True, dtype=weight_dtype):
                # Process in smaller chunks with gradient disabled for VAE
                frames = batch["frames"].to(weight_dtype)
                chunk_size = 1  
                latents_list = []
                
                # Split frames into chunks
                for i in range(0, frames.shape[0], chunk_size):
                    chunk = frames[i:i+chunk_size]
                    torch.cuda.empty_cache()
                    
                    # Move chunk to CPU after processing if needed
                    with torch.no_grad():
                        latents_chunk = vae.encode(chunk).latent_dist.sample()
                        latents_chunk = latents_chunk * vae.config.scaling_factor
                        latents_list.append(latents_chunk.cpu() if i + chunk_size < frames.shape[0] else latents_chunk)
                    del chunk
                    torch.cuda.empty_cache()
                
                # Concatenate all chunks
                if len(latents_list) > 1:
                    latents = torch.cat([chunk.cuda() if chunk.device.type == 'cpu' else chunk for chunk in latents_list], dim=0)
                else:
                    latents = latents_list[0]
                del latents_list
                torch.cuda.empty_cache()
                
                # Get noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, args.noise_scheduler.config.num_train_timesteps, (frames.shape[0],), device=latents.device).to(dtype=latents.dtype)  # Cast timesteps to model dtype immediately
                noisy_latents = args.noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Free up memory
                del frames, latents
                torch.cuda.empty_cache()
                
                # Create dummy encoder hidden states
                batch_size = noisy_latents.shape[0]
                device = noisy_latents.device
                dtype = noisy_latents.dtype
                encoder_hidden_states = torch.zeros(batch_size, 1, transformer.config.text_embed_dim, device=device, dtype=dtype)
                
                # Convert to [B, T, C, H, W] format for transformer
                noisy_frames = noisy_latents.permute(0, 2, 1, 3, 4)
                
                # Create position IDs for rotary embeddings
                position_ids = torch.arange(noisy_frames.shape[1], device=device)
                
                # Predict noise
                noise_pred = transformer(
                    hidden_states=noisy_frames,
                    timestep=timesteps.to(dtype=noisy_frames.dtype),
                    encoder_hidden_states=encoder_hidden_states,
                    position_ids=position_ids,
                ).sample
                
                # Convert predictions back to scheduler format [B, C, T, H, W]
                noise_pred_scheduler = noise_pred.permute(0, 2, 1, 3, 4)
                
                # Compute loss
                loss = F.mse_loss(noise_pred_scheduler.float(), noise.float(), reduction="mean")
            
            # Backprop and optimize
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Free up more memory
            del model_output, noisy_latents, noise
            torch.cuda.empty_cache()
        
        progress_bar.update(1)
        logs = {"loss": loss.detach().item(), "lr": optimizer.param_groups[0]["lr"]}
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=epoch * len(train_dataloader) + step)
        
        # Monitor memory usage
        if accelerator.is_local_main_process and step % 100 == 0:
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            max_memory_allocated = torch.cuda.max_memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"\nStep {step} Memory Stats:")
            print(f"  Current memory allocated: {memory_allocated:.2f}GB")
            print(f"  Max memory allocated: {max_memory_allocated:.2f}GB")
            print(f"  Memory reserved: {memory_reserved:.2f}GB")
            
            # Reset peak memory stats periodically
            torch.cuda.reset_peak_memory_stats()
    
    progress_bar.close()

def main(args):
    """Main training function."""
    # Initialize accelerator
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )

    # Load models
    noise_scheduler = CogVideoXDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )

    vae = AutoencoderKLCogVideoX.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.float16 if args.vae_precision == "fp16" else torch.bfloat16,
    )

    transformer = CogVideoXTransformer3DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16,
    )

    # Create pipeline
    pipeline = CogVideoXInpaintingPipeline(
        vae=vae,
        transformer=transformer,
        scheduler=noise_scheduler,
        device=accelerator.device,
        dtype=torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16 if args.mixed_precision == "fp16" else torch.float32
    )

    # Enable memory optimizations
    if args.enable_slicing:
        vae.enable_slicing()
    if args.enable_tiling:
        vae.enable_tiling()

    # Freeze VAE and put in eval mode
    vae.requires_grad_(False)
    vae.eval()

    # Enable gradient checkpointing for transformer
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        if hasattr(transformer, 'set_use_memory_efficient_attention_xformers'):
            transformer.set_use_memory_efficient_attention_xformers(True)

    # Move models to device and set dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    transformer.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # Create optimizer
    optimizer = torch.optim.AdamW(
        transformer.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        eps=args.epsilon,
    )

    # Dataset and DataLoaders creation
    train_dataset = VideoInpaintingDataset(
        data_root=args.data_root,
        video_dir=args.video_dir,
        mask_dir=args.mask_dir,
        gt_dir=args.gt_dir,
        num_frames=args.num_frames,
        frame_stride=1,  
        image_size=args.image_size,
        center_crop=True,
        normalize=True
    )

    val_dataset = VideoInpaintingDataset(
        data_root=args.data_root,
        video_dir=args.video_dir,
        mask_dir=args.mask_dir,
        gt_dir=args.gt_dir,
        num_frames=args.num_frames,
        frame_stride=1,  
        image_size=args.image_size,
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

    # Calculate max training steps
    args.max_train_steps = args.num_train_epochs * len(train_dataloader) // args.gradient_accumulation_steps

    # Create learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Calculate training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.num_warmup_steps = args.lr_warmup_steps * args.gradient_accumulation_steps
    args.num_training_steps = args.max_train_steps * args.gradient_accumulation_steps
    
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
    import argparse
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = str(Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    parser = argparse.ArgumentParser(description="Train CogVideoX for video inpainting")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--video_dir", type=str, default="RGB_480")
    parser.add_argument("--mask_dir", type=str, default="MASK_480")
    parser.add_argument("--gt_dir", type=str, default="GT_480")
    parser.add_argument("--image_size", type=int, default=480)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=1000)
    parser.add_argument("--checkpointing_steps", type=int, default=2000)
    parser.add_argument("--validation_steps", type=int, default=500)
    parser.add_argument("--mixed_precision", type=str, choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--vae_precision", type=str, choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--window_size", type=int, default=16)
    parser.add_argument("--overlap", type=int, default=4)
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--random_flip_h", type=float, default=0.5)
    parser.add_argument("--random_flip_v", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--deepspeed_config", type=str, default=None)
    parser.add_argument("--ignore_text_encoder", action="store_true", help="Run in text-free mode without text encoder")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--enable_slicing", action="store_true", help="Whether to use VAE slicing for saving memory.")
    parser.add_argument("--enable_tiling", action="store_true", help="Whether to use VAE tiling for saving memory.")
    args = parser.parse_args()
    args.window_size = getattr(args, 'window_size', 32)
    args.overlap = getattr(args, 'overlap', 8)
    args.chunk_size = getattr(args, 'chunk_size', 32)
    args.use_8bit_adam = getattr(args, 'use_8bit_adam', False)
    args.use_flash_attention = getattr(args, 'use_flash_attention', False)
    args.vae_precision = getattr(args, 'vae_precision', "fp16")
    args.max_resolution = getattr(args, 'max_resolution', 640)  
    main(args)