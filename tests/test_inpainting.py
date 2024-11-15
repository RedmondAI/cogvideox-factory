# Run: python3 tests/test_inpainting.py

import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import pytest
import logging
import torch.nn.functional as F
from torch.cuda.amp import autocast
import gc

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_gpu_memory():
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2  # MB
    return 0

def create_test_data(test_dir, num_frames=64, corrupt_frame_idx=None):
    """Create test dataset with optional corrupted frame."""
    for resolution in ['720']:
        for type_ in ['RGB', 'MASK', 'GT']:
            dir_path = test_dir / f'sequence_001' / f'{type_}_{resolution}'
            dir_path.mkdir(parents=True, exist_ok=True)
            
            for i in range(1, num_frames + 1):
                if corrupt_frame_idx == i:
                    # Create corrupted frame for testing error handling
                    img = Image.new('RGB', (100, 100), 'black')  # Wrong resolution
                else:
                    if type_ == 'MASK':
                        img = Image.new('RGB', (1280, 720), 'black')
                        img.paste(Image.new('RGB', (400, 200), 'white'), (440, 260))
                    else:
                        color = 'red' if type_ == 'RGB' else 'blue'
                        img = Image.new('RGB', (1280, 720), color)
                
                img.save(dir_path / f'frame_{i:05d}.png')

def test_dataset_initialization():
    """Test dataset initialization with various configurations."""
    test_dir = Path("assets/inpainting_test_init")
    test_dir.mkdir(parents=True, exist_ok=True)
    create_test_data(test_dir)
    
    from dataset import VideoInpaintingDataset
    
    # Test valid initialization
    dataset = VideoInpaintingDataset(
        data_root=str(test_dir),
        max_num_frames=64,
        height=720,
        width=1280,
    )
    assert len(dataset) == 1
    
    # Test resolution limit
    with pytest.raises(ValueError, match="Resolution.*exceeds maximum"):
        VideoInpaintingDataset(
            data_root=str(test_dir),
            height=3000,  # Exceeds max_resolution
            width=1280,
        )
    
    # Test missing directory
    with pytest.raises(RuntimeError, match="No valid sequences found"):
        VideoInpaintingDataset(
            data_root="nonexistent_dir",
            height=720,
            width=1280,
        )
    
    print("Dataset initialization tests passed!")

def test_memory_efficiency():
    """Test memory usage during dataset operations."""
    test_dir = Path("assets/inpainting_test_memory")
    test_dir.mkdir(parents=True, exist_ok=True)
    create_test_data(test_dir, num_frames=100)  # Create longer sequence
    
    from dataset import VideoInpaintingDataset
    
    initial_memory = get_gpu_memory()
    
    dataset = VideoInpaintingDataset(
        data_root=str(test_dir),
        max_num_frames=100,
        height=720,
        width=1280,
    )
    
    # Test memory during loading
    memory_checkpoints = []
    for i in range(3):
        item = dataset[0]
        current_memory = get_gpu_memory()
        memory_checkpoints.append(current_memory)
        torch.cuda.empty_cache()
        gc.collect()
    
    # Check memory is released
    final_memory = get_gpu_memory()
    assert final_memory < max(memory_checkpoints), "Memory not properly released"
    
    print("Memory efficiency tests passed!")

def test_padding():
    """Test resolution padding and unpadding."""
    from cogvideox_video_inpainting_sft import pad_to_multiple, unpad
    
    # Create test tensor with non-multiple dimensions
    x = torch.randn(1, 3, 720, 1280)
    padded, pad_sizes = pad_to_multiple(x, multiple=64)
    
    # Check padded dimensions are multiples of 64
    assert padded.shape[-2] % 64 == 0, f"Height {padded.shape[-2]} not multiple of 64"
    assert padded.shape[-1] % 64 == 0, f"Width {padded.shape[-1]} not multiple of 64"
    
    # Test maximum dimension check
    with pytest.raises(ValueError, match="exceed maximum safe size"):
        pad_to_multiple(torch.randn(1, 3, 2000, 2000), max_dim=2048)
    
    # Check unpadding recovers original size
    unpadded = unpad(padded, pad_sizes)
    assert unpadded.shape == x.shape, f"Expected shape {x.shape}, got {unpadded.shape}"
    assert torch.allclose(unpadded, x), "Unpadded content doesn't match original"
    
    print("Padding test passed!")

def test_temporal_smoothing():
    """Test temporal smoothing function."""
    from cogvideox_video_inpainting_sft import temporal_smooth
    
    # Create test sequence with known pattern
    seq = torch.zeros(1, 10, 3, 64, 64)
    seq[:, 5:] = 1.0  # Sharp transition
    
    # Apply smoothing
    smoothed = temporal_smooth(seq, window_size=3)
    
    # Check middle frames unchanged
    assert torch.allclose(smoothed[:, :4], seq[:, :4])
    
    # Check transition is smoothed
    assert not torch.allclose(smoothed[:, 4:7], seq[:, 4:7])
    
    # Test different window sizes
    for window_size in [3, 5, 7]:
        smoothed = temporal_smooth(seq, window_size=window_size)
        assert smoothed.shape == seq.shape
    
    print("Temporal smoothing test passed!")

def test_metrics():
    """Test metric computation."""
    from cogvideox_video_inpainting_sft import compute_metrics
    
    # Create test tensors
    pred = torch.rand(1, 5, 3, 64, 64)
    gt = pred.clone()  # Perfect prediction
    mask = torch.ones_like(pred[:, :, :1])  # Full mask
    
    # Test perfect prediction
    metrics = compute_metrics(pred, gt, mask)
    assert metrics['masked_psnr'] > 40
    assert metrics['masked_ssim'] > 0.95
    assert metrics['temporal_consistency'] > 0.95
    
    # Test with noise
    noisy_pred = pred + torch.randn_like(pred) * 0.1
    metrics = compute_metrics(noisy_pred, gt, mask)
    assert metrics['masked_psnr'] < 40
    assert metrics['masked_ssim'] < 0.95
    
    # Test partial mask
    partial_mask = torch.zeros_like(mask)
    partial_mask[:, :, :, :32] = 1
    metrics = compute_metrics(pred, gt, partial_mask)
    
    print("Metrics test passed!")

def test_dataset():
    """Test the VideoInpaintingDataset."""
    from dataset import VideoInpaintingDataset
    
    # Create test directory
    test_dir = Path("assets/inpainting_test")
    test_dir.mkdir(parents=True, exist_ok=True)
    create_test_data(test_dir)
    
    # Create dataset
    dataset = VideoInpaintingDataset(
        data_root=str(test_dir),
        max_num_frames=64,
        height=720,
        width=1280,
        random_flip_h=0.5,
        random_flip_v=0.5,
    )
    
    # Test dataset size
    assert len(dataset) == 1, f"Expected 1 sequence, got {len(dataset)}"
    
    # Test single item
    item = dataset[0]
    assert "rgb" in item, "RGB frames missing from dataset item"
    assert "mask" in item, "Mask frames missing from dataset item"
    assert "gt" in item, "GT frames missing from dataset item"
    
    # Test shapes
    assert item["rgb"].shape == (64, 3, 720, 1280), f"Wrong RGB shape: {item['rgb'].shape}"
    assert item["mask"].shape == (64, 1, 720, 1280), f"Wrong mask shape: {item['mask'].shape}"
    assert item["gt"].shape == (64, 3, 720, 1280), f"Wrong GT shape: {item['gt'].shape}"
    
    print("Dataset test passed!")
    return dataset

def test_model_modification():
    """Test the model input channel modification and memory optimizations."""
    from diffusers import CogVideoXTransformer3DModel
    import torch.nn as nn
    
    # Create a small test model
    model = CogVideoXTransformer3DModel(
        in_channels=3,
        out_channels=3,
        num_layers=1,
        num_attention_heads=1,
    )
    
    # Test gradient checkpointing
    model.enable_gradient_checkpointing()
    assert any(hasattr(m, '_gradient_checkpointing') for m in model.modules()), "Gradient checkpointing not enabled"
    
    # Modify model for inpainting
    old_conv = model.conv_in
    model.config.in_channels += 1
    model.conv_in = nn.Conv3d(
        model.config.in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
    )
    
    # Initialize new weights
    with torch.no_grad():
        model.conv_in.weight[:, :3] = old_conv.weight
        model.conv_in.weight[:, 3:] = 0
        model.conv_in.bias = nn.Parameter(old_conv.bias.clone())
    
    # Test forward pass with chunked input
    batch_size, chunk_size = 1, 32
    test_input = torch.randn(batch_size, chunk_size, 4, 64, 64)  # 4 channels (RGB + mask)
    output = model(test_input)
    
    print("Model modification test passed!")
    return model

def test_pipeline():
    """Test the inpainting pipeline with chunked processing."""
    from cogvideox_video_inpainting_sft import CogVideoXInpaintingPipeline
    from diffusers import CogVideoXDPMScheduler
    
    # Get test dataset and model
    dataset = test_dataset()
    model = test_model_modification()
    scheduler = CogVideoXDPMScheduler()
    
    # Create pipeline with window processing
    pipeline = CogVideoXInpaintingPipeline(
        vae=model,  # Using transformer as fake VAE for testing
        transformer=model,
        scheduler=scheduler,
        window_size=32,
        overlap=8,
        vae_precision="fp16",
        max_resolution=2048,
    )
    
    # Get test batch
    batch = dataset[0]
    rgb = batch["rgb"].unsqueeze(0)
    mask = batch["mask"].unsqueeze(0)
    
    # Test pipeline processing
    with torch.no_grad():
        output = pipeline(
            rgb_frames=rgb[:, :32],  # Test with smaller sequence for memory
            mask_frames=mask[:, :32],
            num_inference_steps=2,  # Small number for testing
        )
    
    assert output.shape == rgb[:, :32].shape, f"Expected shape {rgb[:, :32].shape}, got {output.shape}"
    print("Pipeline test passed!")

def test_error_handling():
    """Test error handling in pipeline."""
    from cogvideox_video_inpainting_sft import CogVideoXInpaintingPipeline
    from diffusers import CogVideoXDPMScheduler
    
    # Get test model
    model = test_model_modification()
    scheduler = CogVideoXDPMScheduler()
    
    # Create pipeline
    pipeline = CogVideoXInpaintingPipeline(
        vae=model,
        transformer=model,
        scheduler=scheduler,
        window_size=32,
        overlap=8,
    )
    
    # Test with invalid input
    try:
        # Should raise error for wrong shape
        invalid_input = torch.randn(1, 32, 2, 64, 64)  # Wrong number of channels
        pipeline(invalid_input, invalid_input)
        assert False, "Should have raised error for invalid input"
    except Exception as e:
        assert "Error in pipeline inference" in str(e)
    
    print("Error handling test passed!")

def test_edge_cases():
    """Test edge cases."""
    from cogvideox_video_inpainting_sft import CogVideoXInpaintingPipeline
    from diffusers import CogVideoXDPMScheduler
    
    # Get test model and create pipeline
    model = test_model_modification()
    scheduler = CogVideoXDPMScheduler()
    pipeline = CogVideoXInpaintingPipeline(
        vae=model,
        transformer=model,
        scheduler=scheduler,
        window_size=32,
        overlap=8,
    )
    
    # Test cases
    test_cases = [
        # Very small mask
        {
            'name': 'small_mask',
            'rgb': torch.randn(1, 32, 3, 64, 64),
            'mask': torch.zeros(1, 32, 1, 64, 64),
            'mask_area': (28, 28, 36, 36),  # Small central area
        },
        # Full frame mask
        {
            'name': 'full_mask',
            'rgb': torch.randn(1, 32, 3, 64, 64),
            'mask': torch.ones(1, 32, 1, 64, 64),
        },
        # Irregular mask shape
        {
            'name': 'irregular_mask',
            'rgb': torch.randn(1, 32, 3, 64, 64),
            'mask': torch.rand(1, 32, 1, 64, 64) > 0.5,
        },
        # Minimum sequence length
        {
            'name': 'minimum_sequence',
            'rgb': torch.randn(1, 2, 3, 64, 64),
            'mask': torch.ones(1, 2, 1, 64, 64),
        },
    ]
    
    for case in test_cases:
        print(f"Testing {case['name']}...")
        
        # Create mask if specified
        if 'mask_area' in case:
            x1, y1, x2, y2 = case['mask_area']
            case['mask'][:, :, :, y1:y2, x1:x2] = 1
        
        # Run inference
        with torch.no_grad():
            output = pipeline(
                rgb_frames=case['rgb'],
                mask_frames=case['mask'].float(),
                num_inference_steps=2,
            )
        
        # Check output
        assert output.shape == case['rgb'].shape, f"Wrong output shape for {case['name']}"
        assert not torch.isnan(output).any(), f"NaN in output for {case['name']}"
    
    print("Edge cases test passed!")

def test_training_step():
    """Test a single training step with chunked processing."""
    import torch.nn.functional as F
    from diffusers import CogVideoXDPMScheduler
    
    # Get test dataset and model
    dataset = test_dataset()
    model = test_model_modification()
    scheduler = CogVideoXDPMScheduler()
    
    # Get test batch
    batch = dataset[0]
    rgb = batch["rgb"].unsqueeze(0)[:, :32]  # Use smaller sequence for testing
    mask = batch["mask"].unsqueeze(0)[:, :32]
    gt = batch["gt"].unsqueeze(0)[:, :32]
    
    # Test chunk processing
    chunk_size = 16
    overlap = 4
    chunk_losses = []
    
    for start_idx in range(0, rgb.shape[1] - chunk_size + 1, chunk_size - overlap):
        # Get chunk
        chunk_rgb = rgb[:, start_idx:start_idx + chunk_size]
        chunk_mask = mask[:, start_idx:start_idx + chunk_size]
        chunk_gt = gt[:, start_idx:start_idx + chunk_size]
        
        # Pad resolution
        chunk_rgb, rgb_pad = pad_to_multiple(chunk_rgb)
        chunk_mask, _ = pad_to_multiple(chunk_mask)
        chunk_gt, _ = pad_to_multiple(chunk_gt)
        
        # Sample noise
        noise = torch.randn_like(chunk_gt)
        timesteps = torch.tensor([500])
        
        # Model prediction
        model_input = torch.cat([chunk_gt, chunk_mask], dim=2)
        noise_pred = model(
            sample=model_input,
            timestep=timesteps,
            return_dict=False,
        )[0]
        
        # Remove padding
        noise_pred = unpad(noise_pred, (rgb_pad[0]//8, rgb_pad[1]//8))
        noise = unpad(noise, (rgb_pad[0]//8, rgb_pad[1]//8))
        
        # Calculate loss
        chunk_loss = F.mse_loss(noise_pred, noise, reduction="none")
        chunk_loss = (chunk_loss * (chunk_mask + 1.0)).mean()
        chunk_losses.append(chunk_loss)
    
    # Average chunk losses
    loss = torch.stack(chunk_losses).mean()
    
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive at start of training"
    
    print("Training step test passed!")
    print(f"Test loss: {loss.item()}")

def run_all_tests():
    """Run all tests with proper setup and teardown."""
    try:
        print("Running comprehensive test suite...")
        
        test_dataset_initialization()
        test_memory_efficiency()
        test_padding()
        test_temporal_smoothing()
        test_metrics()
        test_dataset()
        test_model_modification()
        test_pipeline()
        test_error_handling()
        test_edge_cases()
        test_training_step()
        
        print("\nAll tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        raise
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    training_dir = os.path.join(project_root, "training")
    sys.path.append(training_dir)
    run_all_tests()