"""Tests for memory usage and performance."""

import torch
import pytest
from diffusers import (
    CogVideoXTransformer3DModel,
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler
)

device = "cuda" if torch.cuda.is_available() else "cpu"

def test_memory_calculation():
    """Test memory usage calculation and optimization."""
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        "THUDM/CogVideoX-5b",
        subfolder="transformer",
        torch_dtype=torch.float16
    ).to(device)
    
    # Enable gradient checkpointing for memory efficiency
    transformer.gradient_checkpointing = True
    assert transformer.is_gradient_checkpointing, "Gradient checkpointing should be enabled"
    
    # Get initial memory usage
    if device == "cuda":
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
    
    # Test with reduced dimensions
    num_frames = 16  # Reduced from 49
    height = 32  # Reduced from 60
    width = 32  # Reduced from 90
    
    # Test with single batch size for memory efficiency
    batch_size = 1
    
    # Create inputs with smaller dimensions
    hidden_states = torch.randn(
        batch_size, 
        num_frames,  # Use reduced frames
        transformer.config.in_channels,
        height // transformer.config.patch_size,  # Use reduced height
        width // transformer.config.patch_size,   # Use reduced width
        device=device, 
        dtype=torch.float16
    )
    encoder_hidden_states = torch.randn(
        batch_size, 1, transformer.config.text_embed_dim,
        device=device, dtype=torch.float16
    )
    timestep = torch.randint(0, 1000, (batch_size,), device=device)
    
    # Create position IDs for rotary embeddings
    position_ids = torch.arange(num_frames, device=device)
    
    # Forward pass
    try:
        output = transformer(
            hidden_states=hidden_states,
            timestep=timestep.to(dtype=torch.float16),
            encoder_hidden_states=encoder_hidden_states,
            position_ids=position_ids,
        ).sample
        
        # Verify no NaN values
        assert not torch.isnan(output).any(), "Model output contains NaN values"
        
        if device == "cuda":
            # Check memory usage
            current_memory = torch.cuda.memory_allocated()
            memory_increase = current_memory - initial_memory
            print(f"Memory usage with gradient checkpointing: {memory_increase / 1024**2:.2f}MB")
            
            # Memory should be reasonable with gradient checkpointing
            max_memory = 12 * 1024  # 12GB max memory usage
            assert memory_increase < max_memory * 1024**2, \
                f"Memory usage {memory_increase/1024**2:.2f}MB exceeds limit {max_memory}GB"
            
            # Clean up
            torch.cuda.empty_cache()
            
        print("Memory calculation test passed!")
        
    except Exception as e:
        print(f"Error during forward pass: {str(e)}")
        raise

def test_metrics_cpu_fallback():
    """Test metrics calculation with CPU fallback."""
    # Create small test tensors
    pred = torch.randn(1, 3, 8, 64, 64)
    target = torch.randn(1, 3, 8, 64, 64)
    
    # Test on CPU
    try:
        # Calculate MSE
        mse_cpu = torch.nn.functional.mse_loss(pred, target)
        assert not torch.isnan(mse_cpu), "MSE calculation on CPU produced NaN"
        
        # Calculate PSNR
        max_pixel = 1.0
        psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse_cpu))
        assert not torch.isnan(psnr), "PSNR calculation on CPU produced NaN"
        
        print("Metrics CPU fallback test passed!")
    except Exception as e:
        pytest.fail(f"CPU fallback test failed: {str(e)}")
