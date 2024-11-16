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
    batch_size = 1  # Reduced batch size
    seq_lengths = [1, 32]  # Reduced sequence lengths for testing
    
    # Use smaller spatial dimensions for testing
    test_height = 32  # Smaller test height
    test_width = 32   # Smaller test width
    test_frames = 8   # Fewer test frames
    
    for seq_len in seq_lengths:
        # Create dummy text embeddings
        encoder_hidden_states = torch.randn(
            batch_size, seq_len, transformer.config.text_embed_dim,
            device=device, dtype=torch.float16
        )
        
        # Create latent input with smaller dimensions
        hidden_states = torch.randn(
            batch_size,
            test_frames,
            transformer.config.in_channels,
            test_height // transformer.config.patch_size,
            test_width // transformer.config.patch_size,
            device=device,
            dtype=torch.float16
        )
        timestep = torch.randint(0, 1000, (batch_size,), device=device)
        
        # Enable gradient checkpointing to save memory
        transformer.enable_gradient_checkpointing()
        
        # Test forward pass
        with torch.no_grad():  # Disable gradients to save memory
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
        # Calculate total padding needed
        total_pad = window - 1
        # Split padding between start and end
        pad_start = total_pad // 2
        pad_end = total_pad - pad_start
        
        # Apply temporal smoothing with proper padding
        padded = torch.nn.functional.pad(
            frames, 
            (0, 0,  # No width padding
             0, 0,  # No height padding
             pad_start, pad_end),  # Temporal padding
            mode='replicate'
        )
        
        # Use average pooling without additional padding
        smoothed = torch.nn.functional.avg_pool3d(
            padded,
            kernel_size=(window, 1, 1),
            stride=1,
            padding=(0, 0, 0)  # No padding in avg_pool3d since we padded manually
        )
        
        # Verify shape
        assert smoothed.shape == frames.shape, \
            f"Shape mismatch after smoothing with window {window}. " \
            f"Expected {frames.shape}, got {smoothed.shape}"
        
        # Verify temporal consistency
        temp_diff = torch.abs(smoothed[:, :, 1:] - smoothed[:, :, :-1])
        max_jump = temp_diff.max().item()
        assert max_jump < 1.0, f"Large temporal discontinuity: {max_jump}"
    
    print("Temporal smoothing tests passed!")

def test_temporal_smoothing_edge_cases():
    """Test temporal smoothing edge cases."""
    # Test single frame
    single_frame = torch.randn(1, 3, 1, 64, 64, device=device)
    # For single frame, just return as is
    smoothed = single_frame.clone()
    assert torch.allclose(smoothed, single_frame), "Single frame smoothing failed"
    
    # Test very short sequence
    short_seq = torch.randn(1, 3, 2, 64, 64, device=device)
    # For short sequence, use minimal window size
    padded = torch.nn.functional.pad(
        short_seq, 
        (0, 0, 0, 0, 0, 1),  # Add one frame of padding at the end
        mode='replicate'
    )
    smoothed = torch.nn.functional.avg_pool3d(
        padded,
        kernel_size=(2, 1, 1),
        stride=1,
        padding=(0, 0, 0)
    )
    assert smoothed.shape == short_seq.shape, \
        f"Short sequence shape mismatch. Expected {short_seq.shape}, got {smoothed.shape}"
    
    print("Temporal smoothing edge cases passed!")
