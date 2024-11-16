"""Tests for model architecture and shapes."""

import torch
import pytest
from diffusers import (
    CogVideoXTransformer3DModel,
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler
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
    
    # Test with model's native dimensions
    batch_size = 2
    num_frames = transformer.config.sample_frames
    height = transformer.config.sample_height
    width = transformer.config.sample_width
    
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
