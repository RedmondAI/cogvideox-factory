"""Tests for model architecture and shapes."""

import torch
import pytest
from diffusers import (
    CogVideoXTransformer3DModel,
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler
)
import sys
from pathlib import Path

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

def test_vae_shapes():
    """Test VAE input/output shapes."""
    vae = AutoencoderKLCogVideoX.from_pretrained(
        "THUDM/CogVideoX-5b",
        subfolder="vae",
        torch_dtype=torch.float16
    ).to(device)
    
    # Test encoding
    x = torch.randn(1, 3, 8, 64, 64, device=device, dtype=torch.float16)
    latent = vae.encode(x).latent_dist.sample()
    assert latent.shape == (1, 16, 2, 8, 8), f"Unexpected latent shape: {latent.shape}"
    
    # Test decoding
    decoded = vae.decode(latent).sample
    assert decoded.shape == (1, 3, 8, 64, 64), f"Unexpected decoded shape: {decoded.shape}"

def test_transformer_shapes():
    """Test transformer input/output shapes."""
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        "THUDM/CogVideoX-5b",
        subfolder="transformer",
        torch_dtype=torch.float16
    ).to(device)
    
    # Use smaller test dimensions to avoid OOM
    batch_size = 1  
    num_frames = 16  
    height = 32  
    width = 32  
    
    # Create test inputs
    hidden_states = torch.randn(
        batch_size, num_frames, transformer.config.in_channels,
        height // transformer.config.patch_size,
        width // transformer.config.patch_size,
        device=device, dtype=torch.float16
    )
    encoder_hidden_states = torch.randn(
        batch_size, 1, transformer.config.text_embed_dim,
        device=device, dtype=torch.float16
    )
    timestep = torch.randint(0, 1000, (batch_size,), device=device)
    
    # Test forward pass
    output = transformer(
        hidden_states=hidden_states,
        timestep=timestep.to(dtype=torch.float16),
        encoder_hidden_states=encoder_hidden_states,
    ).sample
    
    assert output.shape == hidden_states.shape, \
        f"Unexpected transformer output shape: {output.shape}, expected {hidden_states.shape}"

def test_scheduler_config():
    """Test scheduler configuration."""
    scheduler = CogVideoXDPMScheduler.from_pretrained(
        "THUDM/CogVideoX-5b",
        subfolder="scheduler"
    )
    
    # Verify scheduler settings
    assert scheduler.config.num_train_timesteps == 1000
    assert scheduler.config.beta_schedule == "scaled_linear"
    assert scheduler.config.prediction_type == "v_prediction"
    assert scheduler.config.clip_sample is False
    assert scheduler.config.set_alpha_to_one is True
    assert scheduler.config.steps_offset == 0
    assert scheduler.config.rescale_betas_zero_snr is True

def test_training_components():
    """Test training loop components."""
    # Create models and optimizer
    model = CogVideoXTransformer3DModel.from_pretrained(
        "THUDM/CogVideoX-5b",
        subfolder="transformer",
        torch_dtype=torch.float16
    ).to(device)
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    assert model.is_gradient_checkpointing, "Gradient checkpointing should be enabled"
    
    # Verify positional embedding config
    assert model.config.use_rotary_positional_embeddings, "Rotary embeddings should be enabled"
    assert not model.config.use_learned_positional_embeddings, "Learned embeddings should be disabled"
    
    vae = AutoencoderKLCogVideoX.from_pretrained(
        "THUDM/CogVideoX-5b",
        subfolder="vae",
        torch_dtype=torch.float16
    ).to(device)
    
    scheduler = CogVideoXDPMScheduler.from_pretrained(
        "THUDM/CogVideoX-5b",
        subfolder="scheduler"
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    # Test with smaller dimensions
    batch_size = 1  # Reduced from 2
    num_frames = 16  # Reduced from 49
    height = 32  # Reduced from 60
    width = 32  # Reduced from 90
    
    # VAE has fixed 8-frame output and 8x spatial downsampling
    vae_spatial_ratio = 8
    target_frames = num_frames  # Use original frame count
    
    # Start with RGB frames [B, C, T, H, W]
    clean_frames = torch.randn(
        batch_size, 3, target_frames,
        height,  # Use smaller height
        width,   # Use smaller width
        device=device, dtype=torch.float16
    )
    
    # Test VAE encoding/decoding
    latent = vae.encode(clean_frames).latent_dist.sample()
    decoded = vae.decode(latent).sample  # Access .sample attribute
    
    # Handle fixed 8-frame output from VAE
    decoded = handle_vae_temporal_output(decoded, clean_frames.shape[2])
    
    # Verify no NaN values from VAE
    assert not torch.isnan(latent).any(), "VAE latent contains NaN values"
    assert not torch.isnan(decoded).any(), "VAE output contains NaN values"
    
    # Convert to [B, T, C, H, W] format for transformer
    clean_frames = clean_frames.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
    
    # Create position IDs for rotary embeddings
    position_ids = torch.arange(clean_frames.shape[1], device=device)
    
    # Apply patch embedding
    B, T, C, H, W = clean_frames.shape
    clean_frames = model.patch_embed.proj(clean_frames.reshape(-1, C, H, W))  # [B*T, 3072, H//2, W//2]
    
    # Reshape back maintaining [B, T, C, H, W] format
    _, C_latent, H_latent, W_latent = clean_frames.shape
    clean_frames = clean_frames.reshape(B, T, C_latent, H_latent, W_latent)
    
    # Convert to [B, C, T, H, W] for scheduler operations
    clean_frames_scheduler = clean_frames.permute(0, 2, 1, 3, 4)
    
    # Create noise and add noise in scheduler format
    noise = torch.randn_like(clean_frames_scheduler)
    timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size,), device=device)
    noisy_frames = scheduler.add_noise(clean_frames_scheduler, noise, timesteps)
    
    # Convert back to transformer format [B, T, C, H, W]
    noisy_frames = noisy_frames.permute(0, 2, 1, 3, 4)
    
    # Create dummy encoder hidden states
    encoder_hidden_states = torch.zeros(batch_size, 1, model.config.text_embed_dim, device=device, dtype=torch.float16)
    
    # Test model forward pass with position IDs
    noise_pred = model(
        hidden_states=noisy_frames,
        timestep=timesteps.to(dtype=torch.float16),
        encoder_hidden_states=encoder_hidden_states,
        position_ids=position_ids,
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
