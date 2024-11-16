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
    
    # Get initial memory usage
    if device == "cuda":
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
    
    # Test with different batch sizes
    batch_sizes = [1, 2, 4]
    for batch_size in batch_sizes:
        # Create inputs
        hidden_states = torch.randn(
            batch_size, 
            transformer.config.sample_frames,
            transformer.config.in_channels,
            transformer.config.sample_height // transformer.config.patch_size,
            transformer.config.sample_width // transformer.config.patch_size,
            device=device, 
            dtype=torch.float16
        )
        encoder_hidden_states = torch.randn(
            batch_size, 1, transformer.config.text_embed_dim,
            device=device, dtype=torch.float16
        )
        timestep = torch.randint(0, 1000, (batch_size,), device=device)
        
        # Forward pass
        _ = transformer(
            hidden_states=hidden_states,
            timestep=timestep.to(dtype=torch.float16),
            encoder_hidden_states=encoder_hidden_states,
        ).sample
        
        if device == "cuda":
            # Check memory usage
            current_memory = torch.cuda.memory_allocated()
            memory_increase = current_memory - initial_memory
            print(f"Batch size {batch_size} memory usage: {memory_increase / 1024**2:.2f}MB")
            
            # Memory should scale roughly linearly with batch size
            assert memory_increase < initial_memory * batch_size * 2, \
                f"Memory usage {memory_increase} exceeds expected linear scaling"
            
            # Clean up
            torch.cuda.empty_cache()

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
