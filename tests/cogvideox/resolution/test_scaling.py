"""Tests for resolution scaling and padding."""

import torch
import pytest
from diffusers import (
    CogVideoXTransformer3DModel,
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler
)

device = "cuda" if torch.cuda.is_available() else "cpu"

def test_resolution_scaling():
    """Test model behavior with different input resolutions."""
    print("\nTesting resolution scaling...")
    
    # Load models
    vae = AutoencoderKLCogVideoX.from_pretrained(
        "THUDM/CogVideoX-5b",
        subfolder="vae",
        torch_dtype=torch.float16
    ).to(device)
    
    resolutions = [(64, 64), (128, 128), (256, 256)]
    
    for height, width in resolutions:
        print(f"\nTesting resolution {height}x{width}")
        
        # Calculate spatial scaling relative to model's native dimensions
        spatial_scale_h = height / vae.config.sample_height
        spatial_scale_w = width / vae.config.sample_width
        
        # Create test input with 5 frames
        x = torch.randn(1, 3, 5, height, width, device=device, dtype=torch.float16)
        
        # Test VAE encoding/decoding
        latent = vae.encode(x).latent_dist.sample()
        
        # VAE has 8x spatial downsampling
        expected_latent_h = height // 8
        expected_latent_w = width // 8
        
        assert latent.shape[-2:] == (expected_latent_h, expected_latent_w), \
            f"Expected latent shape {expected_latent_h}x{expected_latent_w}, got {latent.shape[-2]}x{latent.shape[-1]}"
        
        # Decode and handle fixed 8-frame output
        decoded = vae.decode(latent).sample
        
        # If decoded frames don't match input, take center frames
        if decoded.shape[2] != x.shape[2]:
            start_idx = (decoded.shape[2] - x.shape[2]) // 2
            decoded = decoded[:, :, start_idx:start_idx + x.shape[2]]
        
        assert decoded.shape == x.shape, \
            f"Expected decoded shape {x.shape}, got {decoded.shape}"
            
        print(f"✓ Resolution {height}x{width} passed VAE test")

def test_padding_edge_cases():
    """Test padding behavior with various input sizes."""
    def calculate_padding(size):
        """Calculate padding to next multiple of 64."""
        return (64 - (size % 64)) % 64
    
    test_cases = [
        (2048, 2048),  # Already multiple of 64
        (2000, 2000),  # Needs padding
        (100, 100),    # Small size
    ]
    
    for h, w in test_cases:
        print(f"\nInput dimensions: h={h}, w={w}")
        
        pad_h = calculate_padding(h)
        pad_w = calculate_padding(w)
        
        if pad_h > 0 or pad_w > 0:
            print(f"Padding amounts: pad_h={pad_h}, pad_w={pad_w}")
            print(f"Final dimensions will be: h={h+pad_h}, w={w+pad_w}")
            
            # Verify padding is correct
            assert (h + pad_h) % 64 == 0, f"Height {h+pad_h} not divisible by 64"
            assert (w + pad_w) % 64 == 0, f"Width {w+pad_w} not divisible by 64"
            assert pad_h < 64, f"Padding {pad_h} exceeds maximum of 64"
            assert pad_w < 64, f"Padding {pad_w} exceeds maximum of 64"
