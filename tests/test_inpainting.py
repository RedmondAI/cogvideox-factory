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
from accelerate.state import PartialState
from accelerate import Accelerator
from accelerate.utils import set_seed, DistributedDataParallelKwargs

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize accelerate state with DDP config
accelerator = Accelerator(
    kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    mixed_precision="fp16",
)
_ = PartialState()

# Set constants for A100 optimization
WINDOW_SIZE = 64
OVERLAP = 16
MAX_RESOLUTION = 2048

def get_gpu_memory():
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2  # MB
    return 0

def create_test_data(test_dir, num_frames=100, corrupt_frame_idx=None):
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
    create_test_data(test_dir, num_frames=100)  # Create enough frames for the test
    
    from dataset import VideoInpaintingDataset
    
    # Test valid initialization with A100-optimized parameters
    dataset = VideoInpaintingDataset(
        data_root=str(test_dir),
        max_num_frames=100,
        height=720,
        width=1280,
        window_size=WINDOW_SIZE,
        overlap=OVERLAP,
        max_resolution=MAX_RESOLUTION,
    )
    assert len(dataset) == 1
    
    # Test resolution limit
    with pytest.raises(ValueError, match="Resolution.*exceeds maximum"):
        VideoInpaintingDataset(
            data_root=str(test_dir),
            max_num_frames=100,
            height=2049,  # Exceeds max resolution
            width=1280,
        )
    
    # Test window size validation
    with pytest.raises(ValueError, match="Window size.*must be at least twice the overlap"):
        VideoInpaintingDataset(
            data_root=str(test_dir),
            max_num_frames=100,
            height=720,
            width=1280,
            window_size=16,  # Too small for overlap
            overlap=16,
        )
    
    print("Dataset initialization tests passed!")

def test_memory_efficiency():
    """Test memory usage during dataset operations."""
    if not accelerator.is_main_process:
        return  # Only run memory test on main process
        
    test_dir = Path("assets/inpainting_test_memory")
    test_dir.mkdir(parents=True, exist_ok=True)
    create_test_data(test_dir, num_frames=100)
    
    from dataset import VideoInpaintingDataset
    
    initial_memory = get_gpu_memory()
    
    dataset = VideoInpaintingDataset(
        data_root=str(test_dir),
        max_num_frames=100,
        height=720,
        width=1280,
        window_size=WINDOW_SIZE,
        overlap=OVERLAP,
        max_resolution=MAX_RESOLUTION,
    )
    
    # Test memory during loading with larger batch sizes
    memory_checkpoints = []
    batch_size = 2  # Test with same batch size as training
    for i in range(3):
        items = [dataset[0] for _ in range(batch_size)]
        # Explicitly delete tensors
        del items
        torch.cuda.empty_cache()
        gc.collect()
        current_memory = get_gpu_memory()
        memory_checkpoints.append(current_memory)
    
    # Check memory is released
    torch.cuda.empty_cache()
    gc.collect()
    final_memory = get_gpu_memory()
    tolerance = 50  # Increased tolerance for multi-GPU setup (in MB)
    assert abs(final_memory - initial_memory) < tolerance, f"Memory not properly released: initial={initial_memory:.1f}MB, final={final_memory:.1f}MB"
    
    print("Memory efficiency tests passed!")

def test_padding():
    """Test resolution padding and unpadding."""
    from cogvideox_video_inpainting_sft import pad_to_multiple, unpad
    
    # Test 1: Normal padding case
    x = torch.randn(1, 3, 720, 1280)
    padded, pad_sizes = pad_to_multiple(x, multiple=64)
    
    # Check padded dimensions are multiples of 64
    assert padded.shape[-2] % 64 == 0, f"Height {padded.shape[-2]} not multiple of 64"
    assert padded.shape[-1] % 64 == 0, f"Width {padded.shape[-1]} not multiple of 64"
    
    # Test 2: Input exceeds maximum dimension
    print("\nTesting input exceeding maximum...")
    with pytest.raises(ValueError, match="Input dimensions .* exceed maximum safe size"):
        pad_to_multiple(torch.randn(1, 3, 2049, 1280), max_dim=2048)
    
    # Test 3: Padding would reach maximum dimension
    print("\nTesting padding reaching maximum...")
    # 1985 when padded to next multiple of 64 will reach 2048, which should not be allowed
    h = w = 2048 - 64 + 1  # = 1985
    print(f"Using dimensions that will reach max after padding: {h}x{w}")
    with pytest.raises(ValueError, match="Padded dimensions .* must be strictly less than maximum size"):
        x_large = torch.randn(1, 3, h, w)
        pad_to_multiple(x_large, multiple=64, max_dim=2048)
    
    # Test 4: Just under the limit should work
    print("\nTesting dimensions just under the limit...")
    h = w = 2048 - 64  # = 1984, will pad to 2047
    x_under = torch.randn(1, 3, h, w)
    padded_under, _ = pad_to_multiple(x_under, multiple=64, max_dim=2048)
    assert max(padded_under.shape) < 2048, "Padded dimensions should be under max_dim"
    
    # Test 5: Unpadding recovers original size
    unpadded = unpad(padded, pad_sizes)
    assert unpadded.shape == x.shape, f"Expected shape {x.shape}, got {unpadded.shape}"
    assert torch.allclose(unpadded, x), "Unpadded content doesn't match original"
    
    print("Padding tests passed!")

def test_temporal_smoothing():
    """Test temporal smoothing function."""
    from cogvideox_video_inpainting_sft import temporal_smooth
    
    # Create test sequence with sharp transition in middle
    seq = torch.zeros(1, 10, 3, 64, 64)
    seq[:, 5:] = 1.0  # Sharp transition at frame 5
    
    # Apply smoothing with window size 3
    window_size = 3
    smoothed = temporal_smooth(seq, window_size=window_size)
    
    # Test 1: Check that frames far from transition are unchanged
    assert torch.allclose(smoothed[:, :3], seq[:, :3]), "Early frames should be unchanged"
    assert torch.allclose(smoothed[:, 7:], seq[:, 7:]), "Late frames should be unchanged"
    
    # Test 2: Check that transition region is smoothed
    mid_start = seq.shape[1] // 2 - window_size // 2
    mid_end = mid_start + window_size
    transition_region = smoothed[:, mid_start:mid_end]
    
    # The smoothed values should be between the original values (0 and 1)
    assert not torch.allclose(transition_region, seq[:, mid_start:mid_end]), \
        "Transition region should be smoothed"
    assert torch.all(transition_region >= 0.0) and torch.all(transition_region <= 1.0), \
        "Smoothed values should be between 0 and 1"
    
    # Test 3: Check monotonic increase in transition region
    transition_mean = transition_region.mean(dim=(0,2,3,4))  # Average across batch, channels, height, width
    assert torch.all(transition_mean[1:] >= transition_mean[:-1]), \
        "Transition should be monotonically increasing"
    
    print("Temporal smoothing tests passed!")

def test_metrics():
    """Test metric computation."""
    from cogvideox_video_inpainting_sft import compute_metrics
    
    # Create test tensors with larger spatial dimensions
    pred = torch.rand(1, 5, 3, 128, 128)  # Increased from 64x64 to 128x128
    gt = pred.clone()  # Perfect prediction
    mask = torch.ones_like(pred[:, :, :1])  # Full mask
    
    # Test 1: Perfect prediction
    metrics = compute_metrics(pred, gt, mask)
    assert metrics['masked_psnr'] > 40, "PSNR should be high for perfect prediction"
    assert metrics['masked_ssim'] > 0.95, "SSIM should be high for perfect prediction"
    assert metrics['temporal_consistency'] > 0.95, "Temporal consistency should be high for perfect prediction"
    
    # Test 2: Noisy prediction
    noisy_pred = pred + torch.randn_like(pred) * 0.1
    noisy_metrics = compute_metrics(noisy_pred, gt, mask)
    assert noisy_metrics['masked_psnr'] < metrics['masked_psnr'], "PSNR should be lower for noisy prediction"
    assert noisy_metrics['masked_ssim'] < metrics['masked_ssim'], "SSIM should be lower for noisy prediction"
    assert noisy_metrics['temporal_consistency'] < metrics['temporal_consistency'], "Temporal consistency should be lower for noisy prediction"
    
    # Test 3: Partial mask
    partial_mask = torch.zeros_like(mask)
    partial_mask[..., 32:96, 32:96] = 1  # Center region
    partial_metrics = compute_metrics(pred, gt, partial_mask)
    assert partial_metrics['masked_psnr'] > 40, "PSNR should be high in masked region"
    assert partial_metrics['masked_ssim'] > 0.95, "SSIM should be high in masked region"
    
    print("Metrics tests passed!")

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
    
    # Enable memory optimizations
    if hasattr(model.config, 'use_memory_efficient_attention'):
        model.config.use_memory_efficient_attention = True
        model.config.attention_mode = "xformers"
    
    # Enable gradient checkpointing
    model.gradient_checkpointing = True
    
    # Test memory optimizations
    assert model.gradient_checkpointing, "Gradient checkpointing not enabled in model"
    assert hasattr(model, 'gradient_checkpointing'), "Model doesn't support gradient checkpointing"
    
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
    with torch.cuda.amp.autocast(enabled=True):  # Test with mixed precision
        output = model(test_input)
    
    assert output.shape == (batch_size, chunk_size, 3, 64, 64), f"Wrong output shape: {output.shape}"
    print("Model modification tests passed!")

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
        window_size=WINDOW_SIZE,
        overlap=OVERLAP,
        vae_precision="fp16",
        max_resolution=MAX_RESOLUTION,
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
        window_size=WINDOW_SIZE,
        overlap=OVERLAP,
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
        window_size=WINDOW_SIZE,
        overlap=OVERLAP,
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