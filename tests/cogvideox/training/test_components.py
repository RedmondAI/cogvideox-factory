"""Tests for training components."""

import torch
import pytest
from diffusers import (
    CogVideoXTransformer3DModel,
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler
)
from diffusers.utils.torch_utils import randn_tensor
from ..utils import create_layer_norm
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

def compute_loss_v_pred_with_snr(noise_pred, noise, timesteps, scheduler, mask=None, noisy_frames=None):
    """Compute v-prediction loss with SNR scaling."""
    # Get scheduler parameters
    alphas_cumprod = scheduler.alphas_cumprod
    alpha_prod_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1, 1)
    sigma_t = torch.sqrt(1 - alpha_prod_t)
    
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
