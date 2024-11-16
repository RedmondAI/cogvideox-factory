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
    """Test resolution scaling with different input sizes."""
    vae = AutoencoderKLCogVideoX.from_pretrained(
        "THUDM/CogVideoX-5b",
        subfolder="vae",
        torch_dtype=torch.float16
    ).to(device)
    
    test_resolutions = [
        (64, 64),      # Small
        (480, 640),    # Medium
        (720, 1280),   # Large
    ]
    
    for height, width in test_resolutions:
        print(f"\nTesting resolution {height}x{width}")
        
        # Calculate scaling factors
        spatial_scale = height / vae.config.sample_size
        temporal_scale = 0.1  # Fixed temporal compression
        print(f"Scaling factors - Spatial: {spatial_scale:.2f}, Temporal: {temporal_scale:.2f}")
        
        # Create test input
        x = torch.randn(1, 3, 8, height, width, device=device, dtype=torch.float16)
        
        # Test encoding
        latent = vae.encode(x).latent_dist.sample()
        print(f"Encoded shape: {latent.shape}")
        
        # Test decoding
        decoded = vae.decode(latent).sample
        print(f"Decoded shape: {decoded.shape}")
        
        # Check chunk boundaries if using chunking
        if width > 256:
            base_size = 256
            effective_size = int(base_size * 0.75)  # 75% of base size
            overlap = 32
            
            # Calculate number of chunks needed
            num_chunks = (width + effective_size - 1) // effective_size
            chunk_positions = [i * effective_size for i in range(num_chunks)]
            chunk_boundaries = [pos + effective_size for pos in chunk_positions[:-1]]
            
            print(f"\nUsing chunks - Base size: {base_size}, Effective size: {effective_size}")
            print(f"Overlap: {overlap}, Number of chunks: {num_chunks}")
            print(f"Chunk start positions: {chunk_positions}")
            print(f"Chunk boundaries: {chunk_boundaries}")
            
            print("\nChecking chunk boundaries:")
            for boundary in chunk_boundaries:
                window_size = overlap
                left_region = decoded[..., boundary-window_size:boundary]
                right_region = decoded[..., boundary:boundary+window_size]
                
                # Calculate differences
                diff = torch.abs(left_region - right_region)
                mean_diff = diff.mean().item()
                max_diff = diff.max().item()
                
                print(f"Boundary {boundary}:")
                print(f"  Window size: {window_size} pixels")
                print(f"  Mean difference: {mean_diff:.6f}")
                print(f"  Max difference: {max_diff:.6f}")
                
                # Verify smooth transitions
                assert mean_diff < 0.1, f"Large mean difference at boundary {boundary}: {mean_diff}"
                assert max_diff < 0.2, f"Large max difference at boundary {boundary}: {max_diff}"

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
