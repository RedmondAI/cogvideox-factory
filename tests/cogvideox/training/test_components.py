"""Tests for training components."""

import torch
import pytest
from diffusers import (
    CogVideoXTransformer3DModel,
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler
)

device = "cuda" if torch.cuda.is_available() else "cpu"

def compute_loss_v_pred_with_snr(noise_pred, noise, timesteps, scheduler, mask=None, noisy_frames=None):
    """Compute v-prediction loss with SNR scaling."""
    # Get SNR values for timesteps
    alphas_cumprod = scheduler.alphas_cumprod.to(device=noise_pred.device, dtype=noise_pred.dtype)
    timesteps = timesteps.to(noise_pred.device)
    snr = alphas_cumprod[timesteps] / (1 - alphas_cumprod[timesteps])
    
    # Min-SNR weighting
    snr_weights = torch.clamp(snr, max=scheduler.config.snr_max)
    loss_weights = snr_weights / snr
    
    # Compute MSE loss
    loss = torch.nn.functional.mse_loss(noise_pred, noise, reduction="none")
    loss = loss.mean(dim=list(range(1, len(loss.shape))))
    
    # Apply loss weights
    loss = loss * loss_weights
    
    # Apply mask if provided
    if mask is not None and noisy_frames is not None:
        assert mask.shape == noisy_frames.shape, "Mask and frames must have same shape"
        loss = loss * (1 - mask.mean([1, 2, 3, 4]))
    
    return loss.mean()

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
    
    # Create test input [B, C, T, H, W]
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
        latents = vae.encode(clean_frames).latent_dist.sample()
        latents = latents * vae.config.scaling_factor  # [B, 16, T//2, H//8, W//8]
    
    # Convert to [B, T, C, H, W] format for transformer
    latents = latents.permute(0, 2, 1, 3, 4)
    
    # Project through patch embedding
    B, T, C, H, W = latents.shape
    latents = latents.reshape(-1, C, H, W)  # [B*T, C, H, W]
    latents = model.patch_embed.proj(latents)  # [B*T, 3072, H//2, W//2]
    
    # Reshape back maintaining [B, T, C, H, W] format
    _, C_proj, H_proj, W_proj = latents.shape
    latents = latents.reshape(B, T, C_proj, H_proj, W_proj)
    
    # Convert to [B, C, T, H, W] for scheduler operations
    latents_scheduler = latents.permute(0, 2, 1, 3, 4)
    
    # Create noise and add noise in scheduler format
    noise = torch.randn_like(latents_scheduler)
    timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size,), device=device)
    noisy_frames = scheduler.add_noise(latents_scheduler, noise, timesteps)
    
    # Convert back to transformer format [B, T, C, H, W]
    noisy_frames = noisy_frames.permute(0, 2, 1, 3, 4)
    
    # Create dummy encoder hidden states
    encoder_hidden_states = torch.zeros(batch_size, 1, model.config.text_embed_dim, device=device, dtype=torch.float16)
    
    # Test model forward pass
    noise_pred = model(
        hidden_states=noisy_frames,
        timestep=timesteps.to(dtype=torch.float16),
        encoder_hidden_states=encoder_hidden_states,
    ).sample
    
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
    # Create dummy predictions and targets
    B, C, T, H, W = 2, 16, 8, 32, 32
    pred = torch.randn(B, C, T, H, W, device=device)
    target = torch.randn(B, C, T, H, W, device=device)
    
    # Test temporal difference loss
    temp_diff_pred = pred[:, :, 1:] - pred[:, :, :-1]
    temp_diff_target = target[:, :, 1:] - target[:, :, :-1]
    
    temp_loss = torch.nn.functional.mse_loss(temp_diff_pred, temp_diff_target)
    assert not torch.isnan(temp_loss).any(), "Temporal loss contains NaN values"
    
    print("Temporal loss test passed!")
