"""Tests for training components."""

import torch
import pytest
from diffusers import (
    CogVideoXTransformer3DModel,
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler
)
from diffusers.utils.torch_utils import randn_tensor
import torch.nn.functional as F
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from training.components import (
    pad_to_multiple,
    unpad,
    temporal_smooth,
    compute_metrics,
    compute_loss,
    compute_loss_v_pred,
    compute_loss_v_pred_with_snr,
    CogVideoXInpaintingPipeline,
    handle_vae_temporal_output
)

device = "cuda" if torch.cuda.is_available() else "cpu"

def compute_loss_v_pred_with_snr(noise_pred, noise, timesteps, scheduler, mask=None, noisy_frames=None):
    """Compute v-prediction loss with SNR scaling."""
    # Get scheduler parameters and ensure float16 precision
    alphas_cumprod = scheduler.alphas_cumprod.to(device=noise_pred.device, dtype=noise_pred.dtype)
    alpha_prod_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1, 1)
    sigma_t = torch.sqrt(1 - alpha_prod_t)
    
    # Ensure all tensors are in float16
    noise = noise.to(dtype=noise_pred.dtype)
    if noisy_frames is not None:
        noisy_frames = noisy_frames.to(dtype=noise_pred.dtype)
    if mask is not None:
        mask = mask.to(dtype=noise_pred.dtype)
    
    # Adjust noise and noisy_frames to match model output spatial dimensions
    if noise.shape != noise_pred.shape:
        H_out, W_out = noise_pred.shape[3:]
        noise = torch.nn.functional.interpolate(
            noise.reshape(-1, *noise.shape[2:]), 
            size=(H_out, W_out), 
            mode='bilinear'
        ).reshape(*noise.shape[:2], *noise_pred.shape[2:])
    
    if noisy_frames is not None and noisy_frames.shape != noise_pred.shape:
        H_out, W_out = noise_pred.shape[3:]
        noisy_frames = torch.nn.functional.interpolate(
            noisy_frames.reshape(-1, *noisy_frames.shape[2:]), 
            size=(H_out, W_out), 
            mode='bilinear'
        ).reshape(*noisy_frames.shape[:2], *noise_pred.shape[2:])
    
    # Compute target
    v_target = noise * alpha_prod_t.sqrt() - sigma_t * noisy_frames if noisy_frames is not None else noise
    
    # Apply mask if provided
    if mask is not None:
        # Adjust mask to match model output spatial dimensions
        if mask.shape != noise_pred.shape:
            H_out, W_out = noise_pred.shape[3:]
            mask = torch.nn.functional.interpolate(
                mask.reshape(-1, *mask.shape[2:]), 
                size=(H_out, W_out), 
                mode='nearest'
            ).reshape(*mask.shape[:2], *noise_pred.shape[2:])
        masked_pred = noise_pred * mask
        masked_target = v_target * mask
        return F.mse_loss(masked_pred, masked_target)
    
    return F.mse_loss(noise_pred, v_target)

def test_loss_temporal():
    """Test temporal loss computation."""
    # Create dummy predictions and targets with model dimensions
    B = 2
    C = 16  # Model latent channels
    T = 49 // 4  # Frames after temporal compression
    H = 60 // 8  # Height in latent space
    W = 90 // 8  # Width in latent space
    
    pred = torch.randn(B, C, T, H, W, device=device)
    target = torch.randn(B, C, T, H, W, device=device)
    
    # Test temporal difference loss
    temp_diff_pred = pred[:, :, 1:] - pred[:, :, :-1]
    temp_diff_target = target[:, :, 1:] - target[:, :, :-1]
    
    temp_loss = torch.nn.functional.mse_loss(temp_diff_pred, temp_diff_target)
    assert not torch.isnan(temp_loss).any(), "Temporal loss contains NaN values"
    
    print("Temporal loss test passed!")

def test_padding_functions():
    """Test padding and unpadding functions."""
    # Test input
    x = torch.randn(1, 3, 16, 100, 100, device=device)
    
    # Test padding
    padded, pad_sizes = pad_to_multiple(x, multiple=64, max_dim=2048)
    pad_h, pad_w = pad_sizes
    
    # Verify padding
    assert padded.shape[-2] % 64 == 0, "Height not padded to multiple"
    assert padded.shape[-1] % 64 == 0, "Width not padded to multiple"
    assert padded.shape[-2] == 128, f"Expected padded height 128, got {padded.shape[-2]}"
    assert padded.shape[-1] == 128, f"Expected padded width 128, got {padded.shape[-1]}"
    
    # Test unpadding
    unpadded = unpad(padded, pad_sizes)
    assert torch.allclose(unpadded, x), "Unpadding did not restore original tensor"
    
    print("Padding functions test passed!")

def test_vae_temporal_handling():
    """Test VAE temporal output handling."""
    # Create dummy VAE output with extra temporal frames
    B, C, T, H, W = 1, 3, 8, 32, 32
    decoded = torch.randn(B, C, T, H, W, device=device)
    target_frames = 5
    
    # Test temporal handling
    processed = handle_vae_temporal_output(decoded, target_frames)
    
    # Verify output
    assert processed.shape[2] == target_frames, \
        f"Expected {target_frames} frames, got {processed.shape[2]}"
    assert processed.shape[:2] == decoded.shape[:2], "Batch or channel dimension changed"
    assert processed.shape[3:] == decoded.shape[3:], "Spatial dimensions changed"
    
    print("VAE temporal handling test passed!")

def test_metrics_computation():
    """Test metrics computation."""
    # Create test inputs
    B, T, C, H, W = 1, 4, 3, 32, 32
    pred = torch.rand(B, T, C, H, W, device=device)
    gt = torch.rand(B, T, C, H, W, device=device)
    mask = torch.ones(B, T, 1, H, W, device=device)
    mask[:, :, :, H//4:3*H//4, W//4:3*W//4] = 0  # Create hole in middle
    
    # Compute metrics
    metrics = compute_metrics(pred, gt, mask)
    
    # Verify metrics
    assert 'masked_psnr' in metrics, "PSNR not computed"
    assert 'masked_ssim' in metrics, "SSIM not computed"
    assert 'temporal_consistency' in metrics, "Temporal consistency not computed"
    assert all(not torch.isnan(torch.tensor(v)) for v in metrics.values()), \
        "Metrics contain NaN values"
    
    print("Metrics computation test passed!")

def test_loss_functions():
    """Test all loss functions."""
    # Create test inputs
    B, T, C, H, W = 2, 4, 16, 32, 32
    model_pred = torch.randn(B, T, C, H, W, device=device)
    noise = torch.randn(B, T, C, H, W, device=device)
    mask = torch.ones(B, T, 1, H, W, device=device)
    mask[:, :, :, H//4:3*H//4, W//4:3*W//4] = 0
    latents = torch.randn(B, T, C, H, W, device=device)
    
    # Test main loss function
    loss = compute_loss(model_pred, noise, mask, latents)
    assert not torch.isnan(loss).any(), "Main loss contains NaN values"
    
    # Test v-prediction loss
    scheduler = CogVideoXDPMScheduler.from_pretrained(
        "THUDM/CogVideoX-5b",
        subfolder="scheduler"
    )
    timesteps = torch.randint(0, 1000, (B,), device=device)
    
    loss_v = compute_loss_v_pred_with_snr(
        model_pred, noise, timesteps, scheduler,
        mask=mask, noisy_frames=latents
    )
    assert not torch.isnan(loss_v).any(), "V-prediction loss contains NaN values"
    
    print("Loss functions test passed!")

def test_training_components():
    """Test training loop components."""
    # Create models and optimizer
    model = CogVideoXTransformer3DModel.from_pretrained(
        "THUDM/CogVideoX-5b",
        subfolder="transformer",
        torch_dtype=torch.float16
    ).to(device)
    
    scheduler = CogVideoXDPMScheduler.from_pretrained(
        "THUDM/CogVideoX-5b",
        subfolder="scheduler"
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    # Test with dimensions matching model config
    batch_size = 2
    num_frames = model.config.sample_frames  # 49 frames
    height = model.config.sample_height      # 60 pixels
    width = model.config.sample_width        # 90 pixels
    
    # Create test input [B, C, T, H, W] with model dimensions
    clean_frames = torch.randn(
        batch_size, 3, num_frames,
        height, width,
        device=device, dtype=torch.float16
    )

    # First encode through VAE to get latent representation
    with torch.no_grad():
        vae = AutoencoderKLCogVideoX.from_pretrained(
            "THUDM/CogVideoX-5b",
            subfolder="vae",
            torch_dtype=torch.float16
        ).to(device)
        
        # VAE has 8x spatial downsampling
        vae_spatial_ratio = 8
        latents = vae.encode(clean_frames).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        
        # Calculate expected temporal dimensions
        target_frames = num_frames // model.config.temporal_compression_ratio
        
        # Handle temporal compression
        if latents.shape[2] > target_frames:
            start_idx = (latents.shape[2] - target_frames) // 2
            latents = latents[:, :, start_idx:start_idx + target_frames]
        
        # Verify dimensions match model analysis specifications
        assert latents.shape[1] == 16, f"Expected 16 latent channels (from model analysis), got {latents.shape[1]}"
        assert latents.shape[2] == target_frames, \
            f"Expected {target_frames} frames after compression (model analysis), got {latents.shape[2]}"
        
        # Handle patch embedding
        patch_size = model.config.patch_size
        expected_h = height // (vae_spatial_ratio * patch_size)
        expected_w = width // (vae_spatial_ratio * patch_size)
        
        # Verify spatial dimensions after VAE and patch embedding
        assert latents.shape[3] == height // vae_spatial_ratio, \
            f"Expected height {height // vae_spatial_ratio}, got {latents.shape[3]}"
        assert latents.shape[4] == width // vae_spatial_ratio, \
            f"Expected width {width // vae_spatial_ratio}, got {latents.shape[4]}"
    
    # Add noise to latents in [B, C, T, H, W] format
    noise = torch.randn_like(latents)
    timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size,), device=device)
    noisy_latents = scheduler.add_noise(latents, noise, timesteps)
    
    # Convert to [B, T, C, H, W] format for transformer
    noisy_frames = noisy_latents.permute(0, 2, 1, 3, 4)
    
    # Create dummy encoder hidden states
    encoder_hidden_states = torch.zeros(batch_size, 1, model.config.text_embed_dim, device=device, dtype=torch.float16)
    
    # Test model forward pass
    noise_pred = model(
        hidden_states=noisy_frames,
        timestep=timesteps.to(dtype=torch.float16),
        encoder_hidden_states=encoder_hidden_states,
    ).sample
    
    # Verify batch, temporal and channel dimensions match
    assert noise_pred.shape[:3] == noisy_frames.shape[:3], \
        f"Model output shape {noise_pred.shape} doesn't match input shape {noisy_frames.shape} in batch, temporal, or channel dimensions"
    
    # Verify spatial dimensions after patch embedding
    H_in, W_in = noisy_frames.shape[3:]
    H_out = H_in - (H_in % model.config.patch_size)  # Round down to nearest multiple of patch_size
    W_out = W_in - (W_in % model.config.patch_size)  # Round down to nearest multiple of patch_size
    
    assert noise_pred.shape[3:] == (H_out, W_out), \
        f"Expected spatial dimensions ({H_out}, {W_out}), got {noise_pred.shape[3:]}"
    
    # Verify no NaN values from activations
    assert not torch.isnan(noise_pred).any(), "Model output contains NaN values"
    
    # Convert predictions to scheduler format for loss computation
    noise_pred_scheduler = noise_pred.permute(0, 2, 1, 3, 4)
    
    # Compute loss with SNR rescaling
    loss = compute_loss_v_pred_with_snr(
        noise_pred_scheduler, noise, timesteps, scheduler,
        mask=None, noisy_frames=noisy_frames.permute(0, 2, 1, 3, 4)
    )
    assert not torch.isnan(loss).any(), "Loss contains NaN values"
    
    # Test gradient computation
    loss.backward()
    
    # Test optimizer step
    optimizer.step()
    optimizer.zero_grad()
    
    print("Training components test passed!")

def test_pipeline_components():
    """Test pipeline components."""
    pipeline = CogVideoXInpaintingPipeline(
        vae=AutoencoderKLCogVideoX.from_pretrained(
            "THUDM/CogVideoX-5b",
            subfolder="vae",
            torch_dtype=torch.float16
        ),
        transformer=CogVideoXTransformer3DModel.from_pretrained(
            "THUDM/CogVideoX-5b",
            subfolder="transformer",
            torch_dtype=torch.float16
        ),
        scheduler=CogVideoXDPMScheduler.from_pretrained(
            "THUDM/CogVideoX-5b",
            subfolder="scheduler"
        )
    )
    
    # Test dimension validation
    B, C, T, H, W = 1, 3, 16, 128, 128
    video = torch.randn(B, C, T, H, W, device=device)
    mask = torch.ones(B, 1, T, H, W, device=device)
    mask[:, :, :, H//4:3*H//4, W//4:3*W//4] = 0
    
    pipeline.validate_dimensions(video, mask)
    
    # Test mask preparation
    prepared_mask = pipeline.prepare_mask(mask)
    assert prepared_mask.shape[2] == T // pipeline.transformer.config.temporal_compression_ratio, \
        "Temporal dimension not properly compressed in mask"
    
    # Test latent preparation
    latents = pipeline.prepare_latents(B, T, H, W)
    assert latents.shape[1] == pipeline.transformer.config.in_channels, \
        "Wrong number of channels in latents"
    
    # Test boundary continuity
    continuity = pipeline.check_boundary_continuity(video, H//2, window_size=8)
    mean_diff, max_diff = continuity
    assert mean_diff is not None and max_diff is not None, \
        "Boundary continuity check failed"
    
    print("Pipeline components test passed!")
