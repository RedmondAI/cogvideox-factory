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

class CogVideoXInpaintingPipeline(BasePipeline):
    def __init__(
        self,
        vae: AutoencoderKLCogVideoX,
        transformer: CogVideoXTransformer3DModel,
        scheduler: CogVideoXDPMScheduler,
        args=None,
    ):
        # Initialize without text encoder and tokenizer since they're not needed for inpainting
        super().__init__(vae, transformer, scheduler)
        
        # Set processing parameters
        self.chunk_size = getattr(args, 'chunk_size', 32) if args else 32
        self.overlap = getattr(args, 'overlap', 4) if args else 4
        self.max_resolution = getattr(args, 'max_resolution', 512) if args else 512
        
        # Set memory optimization
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.8,expandable_segments:True"
        
        # Store model references
        self.transformer = transformer
        self.vae = vae
        
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
        
        # Configure precision based on model type
        model_name = getattr(transformer.config, "_name_or_path", "").lower()
        self.dtype = torch.bfloat16 if "5b" in model_name else torch.float16
    
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
        """Encode video to latent space with temporal compression and memory optimizations."""
        try:
            # Get dimensions and calculate temporal compression
            B, C, T, H, W = x.shape
            temporal_ratio = self.transformer.config.temporal_compression_ratio
            expected_temporal_frames = T // temporal_ratio
            chunk_size = chunk_size or self.chunk_size
            overlap = overlap or self.overlap
            
            # Process in smaller temporal chunks to save memory
            temporal_chunk_size = min(8, T)  # Process 8 frames at a time
            num_temporal_chunks = (T + temporal_chunk_size - 1) // temporal_chunk_size
            temporal_latents = []
            
            # Enable memory optimizations
            if hasattr(self.vae, 'enable_slicing'):
                self.vae.enable_slicing()
            if hasattr(self.vae, 'enable_tiling'):
                self.vae.enable_tiling()
            
            # Move VAE to CPU temporarily if needed
            vae_device = self.vae.device
            if torch.cuda.memory_allocated() > 0.8 * torch.cuda.get_device_properties(0).total_memory:
                self.vae = self.vae.cpu()
                torch.cuda.empty_cache()
            
            for t_idx in range(num_temporal_chunks):
                t_start = t_idx * temporal_chunk_size
                t_end = min(t_start + temporal_chunk_size, T)
                x_chunk = x[:, :, t_start:t_end]
                
                # Process spatial chunks if input is large
                if H * W > self.max_resolution * self.max_resolution:
                    effective_chunk = chunk_size - 2 * overlap
                    num_chunks_h = math.ceil(H / effective_chunk)
                    num_chunks_w = math.ceil(W / effective_chunk)
                    spatial_chunks = []
                    
                    for h_idx in range(num_chunks_h):
                        h_start = h_idx * effective_chunk
                        h_end = min(h_start + chunk_size, H)
                        h_start = max(0, h_end - chunk_size)
                        
                        for w_idx in range(num_chunks_w):
                            w_start = w_idx * effective_chunk
                            w_end = min(w_start + chunk_size, W)
                            w_start = max(0, w_end - chunk_size)
                            
                            # Extract chunk and move to CPU if needed
                            chunk = x_chunk[..., h_start:h_end, w_start:w_end]
                            if self.vae.device == torch.device('cpu'):
                                chunk = chunk.cpu()
                            
                            # Clear cache before encoding
                            torch.cuda.empty_cache()
                            
                            # Encode chunk with mixed precision
                            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                                with torch.no_grad():
                                    chunk_latents = self.vae.encode(chunk).latent_dist.sample()
                                    chunk_latents = chunk_latents * self.vae.config.scaling_factor
                            
                            # Move back to original device
                            if self.vae.device == torch.device('cpu'):
                                chunk_latents = chunk_latents.to(vae_device)
                            
                            # Remove overlap if not first or last chunk
                            if h_idx > 0:
                                chunk_latents = chunk_latents[..., overlap//8:, :]
                            if h_idx < num_chunks_h - 1:
                                chunk_latents = chunk_latents[..., :-(overlap//8), :]
                            if w_idx > 0:
                                chunk_latents = chunk_latents[..., overlap//8:]
                            if w_idx < num_chunks_w - 1:
                                chunk_latents = chunk_latents[..., :-(overlap//8)]
                            
                            spatial_chunks.append(chunk_latents)
                            
                            # Clear chunk from memory
                            del chunk, chunk_latents
                            torch.cuda.empty_cache()
                    
                    # Reconstruct from spatial chunks
                    rows = []
                    chunks_per_row = num_chunks_w
                    for i in range(0, len(spatial_chunks), chunks_per_row):
                        row = torch.cat(spatial_chunks[i:i + chunks_per_row], dim=-1)
                        rows.append(row)
                    chunk_latents = torch.cat(rows, dim=-2)
                    
                    del spatial_chunks, rows
                    torch.cuda.empty_cache()
                else:
                    # Encode whole temporal chunk
                    if self.vae.device == torch.device('cpu'):
                        x_chunk = x_chunk.cpu()
                    
                    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                        with torch.no_grad():
                            chunk_latents = self.vae.encode(x_chunk).latent_dist.sample()
                            chunk_latents = chunk_latents * self.vae.config.scaling_factor
                    
                    if self.vae.device == torch.device('cpu'):
                        chunk_latents = chunk_latents.to(vae_device)
                
                # Handle temporal compression
                target_frames = (t_end - t_start) // temporal_ratio
                if chunk_latents.shape[2] > target_frames:
                    start_idx = (chunk_latents.shape[2] - target_frames) // 2
                    chunk_latents = chunk_latents[:, :, start_idx:start_idx + target_frames]
                
                temporal_latents.append(chunk_latents)
                
                # Clear chunk from memory
                del chunk_latents, x_chunk
                torch.cuda.empty_cache()
            
            # Move VAE back to original device
            if self.vae.device == torch.device('cpu'):
                self.vae = self.vae.to(vae_device)
            
            # Concatenate temporal chunks
            latents = torch.cat(temporal_latents, dim=2)
            del temporal_latents
            torch.cuda.empty_cache()
            
            return latents
            
        except Exception as e:
            logger.error(f"Error in encode: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            raise e
    
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
        temporal_ratio = self.transformer.config.temporal_compression_ratio
        vae_spatial_ratio = 8   # VAE's spatial compression ratio
        
        mask = F.interpolate(mask, size=(
            mask.shape[2]//temporal_ratio,  # Temporal compression from transformer config
            mask.shape[3]//vae_spatial_ratio,   # VAE spatial compression (8x)
            mask.shape[4]//vae_spatial_ratio    # VAE spatial compression (8x)
        ), mode='nearest')
        
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
            prompt: Unused for inpainting
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
        encoder_hidden_states = self.prepare_encoder_hidden_states(batch_size, video.device, video.dtype)
        
        # Initialize noise
        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps[0])
        
        # Prepare for transformer
        latent_model_input = noisy_latents.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
        
        # Get image rotary embeddings
        image_rotary_emb = (
            prepare_rotary_positional_embeddings(
                height=video_latents.shape[3] * 8,  # Scale back to pixel space
                width=video_latents.shape[4] * 8,
                num_frames=video_latents.shape[2],
                vae_scale_factor_spatial=8,  # VAE spatial scaling factor
                patch_size=self.transformer.config.patch_size,
                attention_head_dim=self.transformer.config.attention_head_dim,
                device=video.device,
            )
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )
        
        # Denoise
        for i, t in enumerate(timesteps[:-1]):
            t_back = timesteps[i + 1]
            
            # Predict noise
            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=t,
                encoder_hidden_states=encoder_hidden_states,
                image_rotary_emb=image_rotary_emb,
                return_dict=False,
            )[0]
            
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
        """
        Perform a single training step.
        
        Args:
            batch: Dictionary containing:
                - rgb: Input video frames [B, C, T, H, W]
                - mask: Binary mask [B, 1, T, H, W]
                
        Returns:
            loss: Training loss for this batch
        """
        # Clear cache before processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Move models to correct device and dtype if needed
        device = batch["rgb"].device
        dtype = batch["rgb"].dtype
        
        # Get input tensors with memory-efficient casting
        video = batch["rgb"].to(device=device, dtype=dtype)  # [B, C, T, H, W]
        mask = batch["mask"].to(device=device, dtype=dtype)  # [B, 1, T, H, W]
        
        # Create encoder hidden states (zero conditioning for inpainting)
        encoder_hidden_states = torch.zeros(
            (video.shape[0], 1, self.transformer.config.text_embed_dim),
            device=device,
            dtype=dtype
        )
        
        # Convert video to latent space with memory-efficient encoding
        with torch.amp.autocast(device_type='cuda', enabled=True, dtype=dtype):
            # Encode input video
            video_latents = self.vae.encode(video).latent_dist.sample()
            video_latents = video_latents * self.vae.config.scaling_factor
            
            # Convert mask to latent space
            mask_latents = F.interpolate(
                mask,
                size=(
                    mask.shape[2],  # Keep temporal dimension
                    video_latents.shape[3],  # Downscale spatial dims
                    video_latents.shape[4]
                ),
                mode="nearest"
            )
            
            # Sample timesteps
            timesteps = torch.randint(
                0, self.scheduler.config.num_train_timesteps,
                (video_latents.shape[0],), device=device
            )
            
            # Add noise
            noise = torch.randn_like(video_latents)
            noisy_latents = self.scheduler.add_noise(video_latents, noise, timesteps)
            
            # Convert to [B, T, C, H, W] format for transformer
            noisy_latents = noisy_latents.permute(0, 2, 1, 3, 4)
            mask_latents = mask_latents.permute(0, 2, 1, 3, 4)
            noise = noise.permute(0, 2, 1, 3, 4)
            
            # Get image rotary embeddings
            image_rotary_emb = (
                prepare_rotary_positional_embeddings(
                    height=video_latents.shape[3] * 8,  # Scale back to pixel space
                    width=video_latents.shape[4] * 8,
                    num_frames=video_latents.shape[2],
                    vae_scale_factor_spatial=8,  # VAE spatial scaling factor
                    patch_size=self.transformer.config.patch_size,
                    attention_head_dim=self.transformer.config.attention_head_dim,
                    device=device,
                )
                if self.transformer.config.use_rotary_positional_embeddings
                else None
            )
            
            # Predict noise residual
            noise_pred = self.transformer(
                hidden_states=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states,
                image_rotary_emb=image_rotary_emb,
                return_dict=False,
            )[0]
            
            # Convert back to [B, C, T, H, W] format for loss computation
            noise_pred = noise_pred.permute(0, 2, 1, 3, 4)
            noise = noise.permute(0, 2, 1, 3, 4)
            noisy_latents = noisy_latents.permute(0, 2, 1, 3, 4)
            mask_latents = mask_latents.permute(0, 2, 1, 3, 4)
            
            # Compute loss with SNR rescaling
            loss = compute_loss_v_pred_with_snr(
                noise_pred, noise, timesteps, self.scheduler,
                mask=mask_latents,
                noisy_frames=noisy_latents
            )
        
        # Clear intermediate tensors
        del noise, noisy_latents, video_latents, mask_latents
        if torch.cuda.is_available():
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
                
                # Convert to [B, T, C, H, W] format for transformer
                clean_frames = clean_frames.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
                
                # Create encoder hidden states with validation
                encoder_hidden_states = model.prepare_encoder_hidden_states(
                    batch_size=B,
                    device=clean_frames.device,
                    dtype=model_dtype
                )
                
                # Sample timesteps
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (B,),
                    device=clean_frames.device
                )
                
                # Add noise
                noise = torch.randn_like(clean_frames)
                noisy_frames = noise_scheduler.add_noise(clean_frames, noise, timesteps)
                
                # Get model prediction
                image_rotary_emb = (
                    prepare_rotary_positional_embeddings(
                        height=clean_frames.shape[3] * 8,  # Scale back to pixel space
                        width=clean_frames.shape[4] * 8,
                        num_frames=clean_frames.shape[2],
                        vae_scale_factor_spatial=8,  # VAE spatial scaling factor
                        patch_size=model.config.patch_size,
                        attention_head_dim=model.config.attention_head_dim,
                        device=clean_frames.device,
                    )
                    if model.config.use_rotary_positional_embeddings
                    else None
                )
                
                noise_pred = model(
                    hidden_states=noisy_frames,  # Already in [B, T, C, H, W] format
                    timestep=timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    image_rotary_emb=image_rotary_emb,
                ).sample
                
                # Verify shape consistency
                assert noise_pred.shape == noisy_frames.shape, \
                    f"Model output shape {noise_pred.shape} doesn't match input shape {noisy_frames.shape}"
                
                # Compute loss with SNR rescaling
                loss = compute_loss_v_pred_with_snr(
                    noise_pred, noise, timesteps, noise_scheduler,
                    mask=mask.permute(0, 2, 1, 3, 4),  # Match permuted frames
                    noisy_frames=noisy_frames
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
                            
                            image_rotary_emb = (
                                prepare_rotary_positional_embeddings(
                                    height=clean_frames.shape[3] * 8,  # Scale back to pixel space
                                    width=clean_frames.shape[4] * 8,
                                    num_frames=clean_frames.shape[2],
                                    vae_scale_factor_spatial=8,  # VAE spatial scaling factor
                                    patch_size=model.config.patch_size,
                                    attention_head_dim=model.config.attention_head_dim,
                                    device=clean_frames.device,
                                )
                                if model.config.use_rotary_positional_embeddings
                                else None
                            )
                            
                            noise_pred = model(
                                hidden_states=noisy_frames,
                                timestep=timesteps,
                                encoder_hidden_states=None,  # No text conditioning during training
                                image_rotary_emb=image_rotary_emb,
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
    vae.eval()  # Ensure VAE is in eval mode
    
    # Enable gradient checkpointing for memory efficiency
    if hasattr(transformer, "enable_gradient_checkpointing"):
        transformer.enable_gradient_checkpointing()
    elif hasattr(transformer, "gradient_checkpointing"):
        transformer.gradient_checkpointing = True
    
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
                chunk_size = 1  # Process one frame at a time
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
                timesteps = torch.randint(0, args.noise_scheduler.config.num_train_timesteps, (frames.shape[0],), device=latents.device)
                noisy_latents = args.noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Free up memory
                del frames, latents
                torch.cuda.empty_cache()
                
                # Predict noise
                image_rotary_emb = (
                    prepare_rotary_positional_embeddings(
                        height=frames.shape[3] * 8,  # Scale back to pixel space
                        width=frames.shape[4] * 8,
                        num_frames=frames.shape[2],
                        vae_scale_factor_spatial=8,  # VAE spatial scaling factor
                        patch_size=transformer.config.patch_size,
                        attention_head_dim=transformer.config.attention_head_dim,
                        device=frames.device,
                    )
                    if transformer.config.use_rotary_positional_embeddings
                    else None
                )
                
                noise_pred = transformer(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=batch["prompt_embeds"],
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
                )[0]
                
                # Compute loss
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            
            # Backprop and optimize
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Free up more memory
            del noise_pred, noisy_latents, noise
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
    
    # Load models with memory optimizations
    vae = AutoencoderKLCogVideoX.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.bfloat16 if "5b" in args.pretrained_model_name_or_path.lower() else torch.float16,
        low_cpu_mem_usage=True
    )
    vae.requires_grad_(False)  # Freeze VAE
    vae.eval()  # Ensure VAE is in eval mode
    
    # Set CUDA memory allocation config
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.8,expandable_segments:True"
    
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=torch.float16 if args.mixed_precision == "fp16" else (torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32),
        use_memory_efficient_attention=True,  # Enable memory efficient attention
        use_gradient_checkpointing=True,  # Enable gradient checkpointing
        low_cpu_mem_usage=True
    )
    
    # Additional memory optimizations
    if hasattr(transformer, "enable_gradient_checkpointing"):
        transformer.enable_gradient_checkpointing()
    elif hasattr(transformer, "gradient_checkpointing"):
        transformer.gradient_checkpointing = True
        
    if hasattr(transformer, "set_use_memory_efficient_attention_xformers"):
        transformer.set_use_memory_efficient_attention_xformers(True)
    
    scheduler = CogVideoXDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    
    # Update scheduler config
    scheduler.config.prediction_type = "v_prediction"
    scheduler.config.rescale_betas_zero_snr = True
    scheduler.config.snr_shift_scale = 1.0
    
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
    args.max_resolution = getattr(args, 'max_resolution', 640)  # Maximum resolution before chunking
    main(args)