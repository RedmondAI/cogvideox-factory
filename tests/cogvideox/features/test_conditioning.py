"""Tests for text conditioning and temporal features."""

import torch
import pytest
from diffusers import (
    CogVideoXTransformer3DModel,
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler
)

device = "cuda" if torch.cuda.is_available() else "cpu"

def test_text_conditioning():
    """Test text conditioning in transformer."""
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        "THUDM/CogVideoX-5b",
        subfolder="transformer",
        torch_dtype=torch.float16
    ).to(device)
    
    # Test with different text embedding sizes
    batch_size = 2
    seq_lengths = [1, 77, 226]  # Single token, typical length, max length
    
    for seq_len in seq_lengths:
        # Create dummy text embeddings
        encoder_hidden_states = torch.randn(
            batch_size, seq_len, transformer.config.text_embed_dim,
            device=device, dtype=torch.float16
        )
        
        # Create latent input
        hidden_states = torch.randn(
            batch_size,
            transformer.config.sample_frames,
            transformer.config.in_channels,
            transformer.config.sample_height // transformer.config.patch_size,
            transformer.config.sample_width // transformer.config.patch_size,
            device=device,
            dtype=torch.float16
        )
        timestep = torch.randint(0, 1000, (batch_size,), device=device)
        
        # Test forward pass
        output = transformer(
            hidden_states=hidden_states,
            timestep=timestep.to(dtype=torch.float16),
            encoder_hidden_states=encoder_hidden_states,
        ).sample
        
        assert output.shape == hidden_states.shape, \
            f"Output shape mismatch with seq_len {seq_len}"
        assert not torch.isnan(output).any(), \
            f"NaN values in output with seq_len {seq_len}"
    
    print("Text conditioning test passed!")

def test_temporal_smoothing():
    """Test temporal smoothing functionality."""
    # Create test sequence
    B, C, T, H, W = 1, 3, 16, 64, 64
    frames = torch.randn(B, C, T, H, W, device=device)
    
    # Test different window sizes
    window_sizes = [2, 4, 8]
    
    for window in window_sizes:
        # Apply temporal smoothing
        pad = window // 2
        padded = torch.nn.functional.pad(frames, (0, 0, 0, 0, pad, pad), mode='replicate')
        smoothed = torch.nn.functional.avg_pool3d(
            padded,
            kernel_size=(window, 1, 1),
            stride=1,
            padding=(0, 0, 0)
        )
        
        # Verify shape
        assert smoothed.shape == frames.shape, \
            f"Shape mismatch after smoothing with window {window}"
        
        # Verify temporal consistency
        temp_diff = torch.abs(smoothed[:, :, 1:] - smoothed[:, :, :-1])
        max_jump = temp_diff.max().item()
        assert max_jump < 1.0, f"Large temporal discontinuity: {max_jump}"
    
    print("Temporal smoothing tests passed!")

def test_temporal_smoothing_edge_cases():
    """Test temporal smoothing edge cases."""
    # Test single frame
    single_frame = torch.randn(1, 3, 1, 64, 64, device=device)
    smoothed = torch.nn.functional.avg_pool3d(
        single_frame,
        kernel_size=(1, 1, 1),
        stride=1,
        padding=0
    )
    assert torch.allclose(smoothed, single_frame), "Single frame smoothing failed"
    
    # Test very short sequence
    short_seq = torch.randn(1, 3, 2, 64, 64, device=device)
    padded = torch.nn.functional.pad(short_seq, (0, 0, 0, 0, 1, 1), mode='replicate')
    smoothed = torch.nn.functional.avg_pool3d(
        padded,
        kernel_size=(3, 1, 1),
        stride=1,
        padding=(0, 0, 0)
    )
    assert smoothed.shape == short_seq.shape, "Short sequence shape mismatch"
    
    # Test boundary conditions
    seq = torch.zeros(1, 3, 8, 64, 64, device=device)
    seq[:, :, 0] = 1.0  # Set first frame to 1
    padded = torch.nn.functional.pad(seq, (0, 0, 0, 0, 1, 1), mode='replicate')
    smoothed = torch.nn.functional.avg_pool3d(
        padded,
        kernel_size=(3, 1, 1),
        stride=1,
        padding=(0, 0, 0)
    )
    assert smoothed[:, :, 0].mean() > 0.5, "Start boundary smoothing issue"
    assert smoothed[:, :, -1].mean() < 0.5, "End boundary smoothing issue"
    
    print("Temporal smoothing edge cases test passed!")
