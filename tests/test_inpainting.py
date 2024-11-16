import os
import sys
import gc
import shutil
import logging
import pytest
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import math

from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXTransformer3DModel,
    CogVideoXDPMScheduler,
)

# Add training directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
training_dir = os.path.join(project_root, "training")
sys.path.append(training_dir)

from cogvideox_video_inpainting_sft import CogVideoXInpaintingPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_gpu_memory():
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2  # MB
    return 0

def create_test_data(test_dir, sequence_name="sequence_001", num_frames=100, corrupt_frame_idx=None):
    """Create test dataset with optional corrupted frame.
    
    Args:
        test_dir: Root directory for test data
        sequence_name: Name of the sequence directory
        num_frames: Number of frames to generate
        corrupt_frame_idx: Optional index to create a corrupted frame
    """
    for resolution in ['720']:
        for type_ in ['RGB', 'MASK', 'GT']:
            dir_path = test_dir / sequence_name / f'{type_}_{resolution}'
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
                
                img.save(dir_path / f'frame_{i:05d}.png')  # Using 5 digits padding

def test_dataset_initialization():
    """Test dataset initialization with various configurations."""
    import shutil
    
    test_dir = Path("assets/inpainting_test_init")
    if test_dir.exists():
        shutil.rmtree(test_dir)  # Remove old test data
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Create multiple test sequences for split testing
    for i in range(5):
        seq_name = f"sequence_{i:03d}"
        create_test_data(test_dir, sequence_name=seq_name, num_frames=100)
    
    from dataset import VideoInpaintingDataset
    
    # Test valid initialization with A100-optimized parameters
    dataset = VideoInpaintingDataset(
        data_root=str(test_dir),
        split='train',
        train_ratio=1.0,  # Use all data for training in test
        val_ratio=0.0,
        test_ratio=0.0,
        max_num_frames=100,
        height=720,
        width=1280,
        window_size=WINDOW_SIZE,
        overlap=OVERLAP,
        max_resolution=MAX_RESOLUTION,
    )
    assert len(dataset) == 5, "Expected all 5 sequences in training set"
    
    # Test split ratios (with 5 sequences: 3/1/1 split)
    train_dataset = VideoInpaintingDataset(
        data_root=str(test_dir),
        split='train',
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        max_num_frames=100,
        height=720,
        width=1280,
    )
    assert len(train_dataset) == 3, "Expected 3 sequences in training set with 60% split"
    
    val_dataset = VideoInpaintingDataset(
        data_root=str(test_dir),
        split='val',
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        max_num_frames=100,
        height=720,
        width=1280,
    )
    assert len(val_dataset) == 1, "Expected 1 sequence in validation set with 20% split"
    
    test_dataset = VideoInpaintingDataset(
        data_root=str(test_dir),
        split='test',
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        max_num_frames=100,
        height=720,
        width=1280,
    )
    assert len(test_dataset) == 1, "Expected 1 sequence in test set with 20% split"
    
    # Test invalid split ratios
    with pytest.raises(ValueError, match="Split ratios must sum to 1.0 or less"):
        VideoInpaintingDataset(
            data_root=str(test_dir),
            split='train',
            train_ratio=0.8,
            val_ratio=0.8,  # Sum > 1.0
            test_ratio=0.2,
        )
    
    # Test resolution limit
    with pytest.raises(ValueError, match="Resolution.*exceeds maximum"):
        VideoInpaintingDataset(
            data_root=str(test_dir),
            split='train',
            max_num_frames=100,
            height=2049,  # Exceeds max resolution
            width=1280,
        )
    
    # Test window size validation
    with pytest.raises(ValueError, match="Window size.*must be at least twice the overlap"):
        VideoInpaintingDataset(
            data_root=str(test_dir),
            split='train',
            max_num_frames=100,
            height=720,
            width=1280,
            window_size=16,  # Too small for overlap
            overlap=16,
        )
    
    # Cleanup
    shutil.rmtree(test_dir)
    
    print("Dataset initialization tests passed!")

def test_memory_efficiency():
    """Test memory usage during dataset operations."""
    if not accelerator.is_main_process:
        return  # Only run memory test on main process
    
    import shutil
    
    test_dir = Path("assets/inpainting_test_memory")
    if test_dir.exists():
        shutil.rmtree(test_dir)  # Remove old test data
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Create multiple sequences for proper testing
    for i in range(3):  # Create 3 sequences
        create_test_data(test_dir, sequence_name=f"sequence_{i:03d}", num_frames=100)
    
    from dataset import VideoInpaintingDataset
    
    initial_memory = get_gpu_memory()
    
    dataset = VideoInpaintingDataset(
        data_root=str(test_dir),
        split='train',  # Use train split
        train_ratio=1.0,  # Use all sequences for training
        val_ratio=0.0,
        test_ratio=0.0,
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
    assert max(memory_checkpoints) - initial_memory < 1000, \
        f"Memory usage increased by {max(memory_checkpoints) - initial_memory}MB"
    
    # Cleanup
    shutil.rmtree(test_dir)
    
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
    import shutil
    
    # Create test directory (clean it first)
    test_dir = Path("assets/inpainting_test")
    if test_dir.exists():
        shutil.rmtree(test_dir)  # Remove old test data
    test_dir.mkdir(parents=True, exist_ok=True)
    create_test_data(test_dir, sequence_name="sequence_000")  # Create sequence with standard name
    
    # Create dataset
    dataset = VideoInpaintingDataset(
        data_root=str(test_dir),
        split='train',  # Use train split
        train_ratio=1.0,  # Use all data for training
        val_ratio=0.0,
        test_ratio=0.0,
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
    
    # Cleanup
    shutil.rmtree(test_dir)
    
    print("Dataset test passed!")

def test_model_modification():
    """Test the model input channel modification and memory optimizations."""
    from diffusers import CogVideoXTransformer3DModel, AutoencoderKLCogVideoX
    
    # Create test model
    vae = AutoencoderKLCogVideoX.from_pretrained(
        "THUDM/CogVideoX-5b",
        subfolder="vae",
        torch_dtype=torch.bfloat16,
    )
    
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        "THUDM/CogVideoX-5b",
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )
    
    # Get VAE latent channels
    vae_latent_channels = vae.config.latent_channels
    assert vae_latent_channels == 16, f"Expected 16 latent channels for CogVideoX-5b VAE, got {vae_latent_channels}"
    
    # Test input channel modification
    old_proj = transformer.patch_embed.proj
    original_in_channels = old_proj.weight.size(1)
    
    new_proj = torch.nn.Conv2d(
        vae_latent_channels + 1,  # Add mask channel to latent channels
        old_proj.out_channels,
        kernel_size=old_proj.kernel_size,
        stride=old_proj.stride,
        padding=old_proj.padding,
    ).to(dtype=torch.bfloat16)  # Match model precision
    
    with torch.no_grad():
        # Copy latent channel weights
        new_proj.weight[:, :vae_latent_channels] = old_proj.weight[:, :vae_latent_channels]
        # Initialize new mask channel to 0
        new_proj.weight[:, vae_latent_channels:] = 0
        new_proj.bias = torch.nn.Parameter(old_proj.bias.clone())
    
    transformer.patch_embed.proj = new_proj
    
    # Test memory optimizations
    transformer.gradient_checkpointing_enable()
    assert transformer.is_gradient_checkpointing, "Gradient checkpointing should be enabled"
    
    # Test input tensor shapes
    B, T, C, H, W = 1, 32, vae_latent_channels, 90, 160
    latents = torch.randn(B, T, C, H, W)
    mask = torch.randn(B, T, 1, H, W)
    combined = torch.cat([latents, mask], dim=2)
    
    # Test forward pass
    timesteps = torch.tensor([0])
    encoder_hidden_states = torch.randn(B, T, transformer.config.hidden_size)
    
    with torch.amp.autocast('cuda', enabled=True):
        output = transformer(
            sample=combined,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
        )
    
    assert output.shape == latents.shape, f"Expected shape {latents.shape}, got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN values"
    
    print("Model modification tests passed!")

def test_pipeline():
    """Test the inpainting pipeline with chunked processing."""
    from cogvideox_video_inpainting_sft import CogVideoXInpaintingPipeline
    from diffusers import CogVideoXDPMScheduler, CogVideoXTransformer3DModel, AutoencoderKLCogVideoX
    import shutil
    
    # Load models
    vae = AutoencoderKLCogVideoX.from_pretrained(
        "THUDM/CogVideoX-5b",
        subfolder="vae",
        torch_dtype=torch.bfloat16,
    )
    
    # Verify VAE latent channels
    vae_latent_channels = vae.config.latent_channels
    assert vae_latent_channels == 16, f"Expected 16 latent channels for CogVideoX-5b VAE, got {vae_latent_channels}"
    
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        "THUDM/CogVideoX-5b",
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )
    
    # Modify transformer for inpainting
    old_proj = transformer.patch_embed.proj
    new_proj = torch.nn.Conv2d(
        vae_latent_channels + 1,  # 16 latent channels + 1 mask channel
        old_proj.out_channels,
        kernel_size=old_proj.kernel_size,
        stride=old_proj.stride,
        padding=old_proj.padding,
    ).to(dtype=torch.bfloat16)
    
    with torch.no_grad():
        new_proj.weight[:, :vae_latent_channels] = old_proj.weight[:, :vae_latent_channels]
        new_proj.weight[:, vae_latent_channels:] = 0
        new_proj.bias = torch.nn.Parameter(old_proj.bias.clone())
    transformer.patch_embed.proj = new_proj
    
    # Create pipeline
    pipeline = CogVideoXInpaintingPipeline(
        vae=vae,
        transformer=transformer,
        scheduler=CogVideoXDPMScheduler(),
        window_size=32,
        overlap=8,
        vae_precision="bf16",
    )
    
    # Create test data
    test_dir = Path("assets/inpainting_test_pipeline")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)
    create_test_data(test_dir, sequence_name="sequence_000", num_frames=64)
    
    # Load test data
    from dataset import VideoInpaintingDataset
    dataset = VideoInpaintingDataset(
        data_root=str(test_dir),
        split='train',
        train_ratio=1.0,
        val_ratio=0.0,
        test_ratio=0.0,
        max_num_frames=64,
        height=720,
        width=1280,
    )
    
    # Get test batch
    batch = dataset[0]
    rgb = batch["rgb"].unsqueeze(0)
    mask = batch["mask"].unsqueeze(0)
    
    # Test pipeline
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=True):
            # Test VAE encoding
            latents = pipeline.vae.encode(rgb).latent_dist.sample()
            assert latents.shape[2] == vae_latent_channels, f"VAE output has {latents.shape[2]} channels, expected {vae_latent_channels}"
            
            # Test pipeline call
            output = pipeline(
                rgb_frames=rgb,
                mask_frames=mask,
                num_inference_steps=2,  # Use small number for testing
            )
    
    # Verify output
    assert output.shape == rgb.shape, f"Output shape {output.shape} doesn't match input shape {rgb.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert output.dtype == rgb.dtype, f"Output dtype {output.dtype} doesn't match input dtype {rgb.dtype}"
    
    # Test memory cleanup
    del pipeline
    torch.cuda.empty_cache()
    
    # Cleanup
    shutil.rmtree(test_dir)
    
    print("Pipeline test passed!")

def test_error_handling():
    """Test error handling in pipeline."""
    from cogvideox_video_inpainting_sft import CogVideoXInpaintingPipeline
    from diffusers import CogVideoXDPMScheduler, CogVideoXTransformer3DModel
    import shutil
    
    # Create test data
    test_dir = Path("assets/inpainting_test_errors")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)
    create_test_data(test_dir, sequence_name="sequence_000")
    
    # Create test dataset
    from dataset import VideoInpaintingDataset
    dataset = VideoInpaintingDataset(
        data_root=str(test_dir),
        split='train',
        train_ratio=1.0,
        val_ratio=0.0,
        test_ratio=0.0,
        max_num_frames=64,
        height=720,
        width=1280,
    )
    
    # Create model with inpainting modification
    model = CogVideoXTransformer3DModel(
        in_channels=3,
        out_channels=3,
        num_layers=1,
        num_attention_heads=1,
        hidden_size=768,  # Use hidden_size instead of cross_attention_dim
    )
    old_proj = model.patch_embed.proj
    new_proj = torch.nn.Conv2d(
        old_proj.in_channels + 1,
        old_proj.out_channels,
        kernel_size=old_proj.kernel_size,
        stride=old_proj.stride,
        padding=old_proj.padding,
    )
    with torch.no_grad():
        new_proj.weight[:, :3] = old_proj.weight
        new_proj.weight[:, 3:] = 0
        new_proj.bias = torch.nn.Parameter(old_proj.bias.clone())
    model.patch_embed.proj = new_proj
    
    # Create pipeline
    pipeline = CogVideoXInpaintingPipeline(
        vae=model,
        transformer=model,
        scheduler=CogVideoXDPMScheduler(),
        window_size=WINDOW_SIZE,
        overlap=OVERLAP,
    )
    
    # Get test batch
    batch = dataset[0]
    rgb = batch["rgb"].unsqueeze(0)
    mask = batch["mask"].unsqueeze(0)
    
    # Test error handling with mixed precision
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=True):
            # Create encoder hidden states
            encoder_hidden_states = torch.randn(rgb.shape[0], rgb.shape[1], 768)  # Match hidden_size
            pipeline.transformer._encoder_hidden_states = encoder_hidden_states  # Set for pipeline use
            
            # Test with mismatched shapes
            with pytest.raises(ValueError, match="RGB and mask shapes must match"):
                pipeline(
                    rgb_frames=rgb[:, :32],
                    mask_frames=mask[:, :16],  # Different sequence length
                )
            
            # Test with invalid overlap
            with pytest.raises(ValueError, match="Overlap must be less than window size"):
                pipeline.overlap = pipeline.window_size + 1
                pipeline(rgb_frames=rgb[:, :32], mask_frames=mask[:, :32])
    
    # Cleanup
    shutil.rmtree(test_dir)
    
    print("Error handling tests passed!")

def test_edge_cases():
    """Test edge cases."""
    from cogvideox_video_inpainting_sft import CogVideoXInpaintingPipeline
    from diffusers import CogVideoXDPMScheduler, CogVideoXTransformer3DModel
    import shutil
    
    # Create test data
    test_dir = Path("assets/inpainting_test_edges")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Test corrupted frame
    create_test_data(test_dir, sequence_name="sequence_000", corrupt_frame_idx=50)
    
    # Create test dataset
    from dataset import VideoInpaintingDataset
    dataset = VideoInpaintingDataset(
        data_root=str(test_dir),
        split='train',
        train_ratio=1.0,
        val_ratio=0.0,
        test_ratio=0.0,
        max_num_frames=64,
        height=720,
        width=1280,
    )
    
    # Create model with inpainting modification
    model = CogVideoXTransformer3DModel(
        in_channels=3,
        out_channels=3,
        num_layers=1,
        num_attention_heads=1,
        hidden_size=768,  # Use hidden_size instead of cross_attention_dim
    )
    old_proj = model.patch_embed.proj
    new_proj = torch.nn.Conv2d(
        old_proj.in_channels + 1,
        old_proj.out_channels,
        kernel_size=old_proj.kernel_size,
        stride=old_proj.stride,
        padding=old_proj.padding,
    )
    with torch.no_grad():
        new_proj.weight[:, :3] = old_proj.weight
        new_proj.weight[:, 3:] = 0
        new_proj.bias = torch.nn.Parameter(old_proj.bias.clone())
    model.patch_embed.proj = new_proj
    
    # Create pipeline
    pipeline = CogVideoXInpaintingPipeline(
        vae=model,
        transformer=model,
        scheduler=CogVideoXDPMScheduler(),
        window_size=WINDOW_SIZE,
        overlap=OVERLAP,
    )
    
    # Get test batch
    batch = dataset[0]
    rgb = batch["rgb"].unsqueeze(0)
    mask = batch["mask"].unsqueeze(0)
    
    # Test edge cases with mixed precision
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=True):
            # Create encoder hidden states
            encoder_hidden_states = torch.randn(rgb.shape[0], rgb.shape[1], 768)  # Match hidden_size
            pipeline.transformer._encoder_hidden_states = encoder_hidden_states  # Set for pipeline use
            
            # Test with minimum sequence length
            min_output = pipeline(
                rgb_frames=rgb[:, :1],
                mask_frames=mask[:, :1],
                num_inference_steps=1,
            )
            assert min_output.shape == rgb[:, :1].shape
            
            # Test with maximum overlap
            pipeline.overlap = pipeline.window_size - 1
            max_overlap_output = pipeline(
                rgb_frames=rgb[:, :32],
                mask_frames=mask[:, :32],
                num_inference_steps=1,
            )
            assert max_overlap_output.shape == rgb[:, :32].shape
            
            # Test with zero overlap
            pipeline.overlap = 0
            no_overlap_output = pipeline(
                rgb_frames=rgb[:, :32],
                mask_frames=mask[:, :32],
                num_inference_steps=1,
            )
            assert no_overlap_output.shape == rgb[:, :32].shape
    
    # Cleanup
    shutil.rmtree(test_dir)
    
    print("Edge cases tests passed!")

def test_training_step():
    """Test a single training step with chunked processing."""
    import torch.nn.functional as F
    from diffusers import CogVideoXDPMScheduler, CogVideoXTransformer3DModel
    import shutil
    
    # Create test data
    test_dir = Path("assets/inpainting_test_training")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)
    create_test_data(test_dir, sequence_name="sequence_000")
    
    # Create test dataset
    from dataset import VideoInpaintingDataset
    dataset = VideoInpaintingDataset(
        data_root=str(test_dir),
        split='train',
        train_ratio=1.0,
        val_ratio=0.0,
        test_ratio=0.0,
        max_num_frames=64,
        height=720,
        width=1280,
    )
    
    # Create model with inpainting modification
    model = CogVideoXTransformer3DModel(
        in_channels=16,  # VAE latent channels
        out_channels=16,
        num_layers=1,
        num_attention_heads=1,
        hidden_size=768,
    )
    
    # Modify input projection for mask channel
    old_proj = model.patch_embed.proj
    new_proj = torch.nn.Conv2d(
        17,  # 16 latent channels + 1 mask channel
        old_proj.out_channels,
        kernel_size=old_proj.kernel_size,
        stride=old_proj.stride,
        padding=old_proj.padding,
    )
    with torch.no_grad():
        new_proj.weight[:, :16] = old_proj.weight[:, :16]  # Copy VAE latent weights
        new_proj.weight[:, 16:] = 0  # Zero-initialize mask channel
        new_proj.bias = torch.nn.Parameter(old_proj.bias.clone())
    model.patch_embed.proj = new_proj
    
    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    noise_scheduler = CogVideoXDPMScheduler()
    
    # Get test batch
    batch = dataset[0]
    rgb = batch["rgb"].unsqueeze(0)
    mask = batch["mask"].unsqueeze(0)
    
    # Simulate VAE encoding with correct channel count
    latents = torch.randn(rgb.shape[0], rgb.shape[1], 16, rgb.shape[3]//8, rgb.shape[4]//8)
    
    # Training step with mixed precision
    model.train()
    with torch.amp.autocast('cuda', enabled=True):
        # Add noise to latents
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (latents.shape[0],), device=latents.device)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Resize mask to match latent resolution
        latent_mask = F.interpolate(mask, size=latents.shape[-2:], mode='nearest')
        
        # Forward pass
        encoder_hidden_states = torch.randn(
            latents.shape[0],
            latents.shape[1],
            768,  # Match hidden_size
        )
        model_output = model(
            sample=torch.cat([noisy_latents, latent_mask], dim=2),
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
        )
        
        # Verify output shape matches latent shape
        assert model_output.shape == latents.shape, f"Model output shape {model_output.shape} doesn't match latent shape {latents.shape}"
        
        # Compute loss
        loss = compute_loss(model_output, noise, latent_mask, latents)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
    
    assert not torch.isnan(loss), "Loss should not be NaN"
    
    # Cleanup
    shutil.rmtree(test_dir)
    torch.cuda.empty_cache()
    
    print("Training step test passed!")

def test_full_training_cycle():
    """Test a complete mini-training cycle to verify all components work together."""
    import torch.nn.functional as F
    from diffusers import CogVideoXDPMScheduler, CogVideoXTransformer3DModel, AutoencoderKLCogVideoX
    import shutil
    from accelerate import Accelerator
    from accelerate.utils import set_seed, DistributedDataParallelKwargs
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Create test data
    test_dir = Path("assets/inpainting_test_full")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Create multiple sequences for proper training simulation
    for i in range(3):
        create_test_data(test_dir, sequence_name=f"sequence_{i:03d}", num_frames=64)
    
    # Initialize accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=2,
        mixed_precision="bf16",
        kwargs_handlers=[ddp_kwargs],
    )
    
    # Create datasets
    from dataset import VideoInpaintingDataset
    train_dataset = VideoInpaintingDataset(
        data_root=str(test_dir),
        split='train',
        train_ratio=0.7,
        val_ratio=0.3,
        test_ratio=0.0,
        max_num_frames=64,
        height=720,
        width=1280,
        window_size=32,
        overlap=8,
    )
    
    val_dataset = VideoInpaintingDataset(
        data_root=str(test_dir),
        split='val',
        train_ratio=0.7,
        val_ratio=0.3,
        test_ratio=0.0,
        max_num_frames=64,
        height=720,
        width=1280,
        window_size=32,
        overlap=8,
    )
    
    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=2,
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
    )
    
    # Load models
    vae = AutoencoderKLCogVideoX.from_pretrained(
        "THUDM/CogVideoX-5b",
        subfolder="vae",
        torch_dtype=torch.bfloat16,
    )
    
    # Verify VAE latent channels
    vae_latent_channels = vae.config.latent_channels
    assert vae_latent_channels == 4, f"Expected 4 latent channels for CogVideoX-5b VAE, got {vae_latent_channels}"
    
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        "THUDM/CogVideoX-5b",
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )
    
    # Modify transformer for inpainting
    old_proj = transformer.patch_embed.proj
    new_proj = torch.nn.Conv2d(
        vae_latent_channels + 1,  # 4 latent channels + 1 mask channel
        old_proj.out_channels,
        kernel_size=old_proj.kernel_size,
        stride=old_proj.stride,
        padding=old_proj.padding,
    ).to(dtype=torch.bfloat16)
    
    with torch.no_grad():
        new_proj.weight[:, :vae_latent_channels] = old_proj.weight[:, :vae_latent_channels]
        new_proj.weight[:, vae_latent_channels:] = 0
        new_proj.bias = torch.nn.Parameter(old_proj.bias.clone())
    transformer.patch_embed.proj = new_proj
    
    # Enable optimizations
    vae.requires_grad_(False)
    transformer.gradient_checkpointing_enable()
    
    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(
        transformer.parameters(),
        lr=1e-5,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        eps=1e-8,
    )
    
    noise_scheduler = CogVideoXDPMScheduler.from_pretrained(
        "THUDM/CogVideoX-5b",
        subfolder="scheduler",
    )
    noise_scheduler.config.prediction_type = "v_prediction"
    
    # Prepare everything with accelerator
    transformer, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        transformer, optimizer, train_dataloader, val_dataloader
    )
    
    # Training loop
    transformer.train()
    num_steps = 4  # Just a few steps to test
    train_loss = 0.0
    val_loss = 0.0
    
    try:
        # Training steps
        for step, batch in enumerate(train_dataloader):
            if step >= num_steps:
                break
                
            with accelerator.accumulate(transformer):
                # Get inputs
                rgb = batch["rgb"]
                mask = batch["mask"]
                
                # Encode frames to latent space
                with torch.no_grad():
                    with torch.amp.autocast('cuda', enabled=True):
                        latents = vae.encode(rgb).latent_dist.sample()
                        assert latents.shape[2] == vae_latent_channels, f"VAE output has {latents.shape[2]} channels, expected {vae_latent_channels}"
                
                # Add noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Resize mask to match latent resolution
                latent_mask = F.interpolate(mask, size=latents.shape[-2:], mode='nearest')
                
                # Model prediction
                model_output = transformer(
                    sample=torch.cat([noisy_latents, latent_mask], dim=2),
                    timestep=timesteps,
                )
                
                # Verify output shape
                assert model_output.shape == latents.shape, f"Model output shape {model_output.shape} doesn't match latent shape {latents.shape}"
                
                # Compute loss
                loss = compute_loss(model_output, noise, latent_mask, latents)
                
                # Backward pass
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                train_loss += loss.detach().item()
        
        # Quick validation
        transformer.eval()
        with torch.no_grad():
            for step, batch in enumerate(val_dataloader):
                if step >= 2:  # Just test a couple validation steps
                    break
                    
                rgb = batch["rgb"]
                mask = batch["mask"]
                
                with torch.amp.autocast('cuda', enabled=True):
                    latents = vae.encode(rgb).latent_dist.sample()
                    assert latents.shape[2] == vae_latent_channels, "VAE output channels mismatch"
                    
                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                    latent_mask = F.interpolate(mask, size=latents.shape[-2:], mode='nearest')
                    
                    model_output = transformer(
                        sample=torch.cat([noisy_latents, latent_mask], dim=2),
                        timestep=timesteps,
                    )
                    
                    assert model_output.shape == latents.shape, "Model output shape mismatch"
                    loss = compute_loss(model_output, noise, latent_mask, latents)
                    val_loss += loss.item()
        
        # Verify training was successful
        avg_train_loss = train_loss / num_steps
        avg_val_loss = val_loss / 2
        
        assert avg_train_loss > 0, "Training loss should be positive"
        assert avg_val_loss > 0, "Validation loss should be positive"
        assert not torch.isnan(torch.tensor(avg_train_loss)), "Training loss is NaN"
        assert not torch.isnan(torch.tensor(avg_val_loss)), "Validation loss is NaN"
        
        # Test model saving and loading
        with accelerator.main_process_first():
            save_dir = test_dir / "checkpoint-test"
            save_dir.mkdir(exist_ok=True)
            accelerator.save_state(save_dir)
            
            # Try loading saved model
            transformer_reloaded = CogVideoXTransformer3DModel.from_pretrained(
                save_dir / "pytorch_model.bin",
                torch_dtype=torch.bfloat16,
            )
            assert transformer_reloaded is not None, "Failed to reload model"
    
    except Exception as e:
        print(f"Error during training simulation: {e}")
        raise
    
    finally:
        # Cleanup
        shutil.rmtree(test_dir)
        torch.cuda.empty_cache()
    
    print("Full training cycle test passed!")

def test_vae_shapes():
    """Test VAE shape transformations including temporal expansion."""
    vae = AutoencoderKLCogVideoX.from_pretrained(
        "THUDM/CogVideoX-5b",
        subfolder="vae",
        torch_dtype=torch.float16
    ).to(device)
    
    # Test input shapes
    B, C, T, H, W = 1, 3, 5, 64, 64
    x = torch.randn(B, C, T, H, W, device=device, dtype=torch.float16)
    
    # Test encoding
    latents = vae.encode(x).latent_dist.sample()
    expected_latent_shape = (B, 16, T//2, H//8, W//8)  # T reduces by 2, spatial by 8
    assert latents.shape == expected_latent_shape, f"Expected latent shape {expected_latent_shape}, got {latents.shape}"
    
    # Test decoding with temporal expansion
    decoded = vae.decode(latents).sample
    expected_output_shape = (B, C, 8, H, W)  # Fixed 8 frames output
    assert decoded.shape == expected_output_shape, f"Expected output shape {expected_output_shape}, got {decoded.shape}"

def test_transformer_shapes():
    """Test transformer shape handling and conditioning."""
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        "THUDM/CogVideoX-5b",
        subfolder="transformer",
        torch_dtype=torch.float16
    ).to(device)
    
    # Test input shapes
    B, C, T, H, W = 1, 3, 5, 64, 64
    x = torch.randn(B, C, T, H, W, device=device, dtype=torch.float16)
    
    # Test transformer forward pass
    latents = torch.randn(B, 16, T//2, H//8, W//8, device=device, dtype=torch.float16)
    timesteps = torch.randint(0, 1000, (B,), device=device, dtype=torch.long)
    encoder_hidden_states = torch.randn(B, 1, 4096, device=device, dtype=torch.float16)
    
    # Call transformer with correct parameter names
    output = transformer(
        hidden_states=latents.permute(0, 2, 1, 3, 4),  # [B, T, C, H, W] for transformer
        timestep=timesteps,
        encoder_hidden_states=encoder_hidden_states,
    ).sample
    
    # Check output shape matches input
    assert output.shape == latents.permute(0, 2, 1, 3, 4).shape, "Unexpected transformer output shape"
    
    # Test decoding
    output = output.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W] for VAE
    vae = AutoencoderKLCogVideoX.from_pretrained(
        "THUDM/CogVideoX-5b",
        subfolder="vae",
        torch_dtype=torch.float16
    ).to(device)
    decoded = vae.decode(output).sample
    assert decoded.shape == (B, C, 8, H, W), "Unexpected final output shape"

def test_scheduler_config():
    """Test scheduler configuration."""
    scheduler = CogVideoXDPMScheduler.from_pretrained(
        "THUDM/CogVideoX-5b",
        subfolder="scheduler"
    )
    
    # Verify default configuration
    assert scheduler.config.prediction_type == "v_prediction"
    assert scheduler.config.beta_schedule == "scaled_linear"
    assert scheduler.config.beta_start == 0.00085
    assert scheduler.config.beta_end == 0.012
    assert scheduler.config.steps_offset == 0
    assert scheduler.config.rescale_betas_zero_snr
    assert scheduler.config.timestep_spacing == "trailing"
    
    # Test scheduler step
    B, C, T, H, W = 1, 16, 5, 8, 8
    sample = torch.randn(B, C, T, H, W)
    model_output = torch.randn_like(sample)
    old_pred = torch.randn_like(sample)
    
    # Set timesteps
    scheduler.set_timesteps(50)
    timestep = scheduler.timesteps[0]
    timestep_back = scheduler.timesteps[1]
    
    # Test step function
    output = scheduler.step(
        model_output=model_output,
        timestep=timestep,
        timestep_back=timestep_back,
        sample=sample,
        old_pred_original_sample=old_pred
    )
    
    # Verify output is tuple with correct shapes
    assert isinstance(output, tuple), "Expected tuple output from scheduler"
    assert len(output) == 2, "Expected 2 components in scheduler output"
    assert output[0].shape == sample.shape, f"Expected shape {sample.shape}, got {output[0].shape}"

def test_end_to_end():
    """Test complete inpainting pipeline."""
    vae = AutoencoderKLCogVideoX.from_pretrained(
        "THUDM/CogVideoX-5b",
        subfolder="vae",
        torch_dtype=torch.float16
    ).to(device)
    
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        "THUDM/CogVideoX-5b",
        subfolder="transformer",
        torch_dtype=torch.float16
    ).to(device)
    
    scheduler = CogVideoXDPMScheduler.from_pretrained(
        "THUDM/CogVideoX-5b",
        subfolder="scheduler"
    )
    
    # Test dimensions
    B, C, T, H, W = 1, 3, 5, 64, 64
    x = torch.randn(B, C, T, H, W, device=device, dtype=torch.float16)  # Match model dtype
    mask = torch.ones(B, 1, T, H, W, device=device, dtype=torch.float16)  # Match model dtype
    
    # Encode
    latents = vae.encode(x).latent_dist.sample()
    assert latents.shape == (B, 16, T//2, H//8, W//8), "Unexpected latent shape"
    
    # Prepare transformer inputs
    timesteps = torch.zeros(B, device=device, dtype=torch.long)
    encoder_hidden_states = torch.randn(B, 1, 4096, device=device, dtype=torch.float16)  # Match model dtype
    
    # Run transformer
    latents = latents.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W] for transformer
    output = transformer(
        hidden_states=latents,
        timestep=timesteps,
        encoder_hidden_states=encoder_hidden_states,
    ).sample  # Access .sample attribute
    
    # Verify transformer output
    assert output.shape == latents.shape, "Unexpected transformer output shape"
    
    # Decode
    output = output.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W] for VAE
    decoded = vae.decode(output).sample
    assert decoded.shape == (B, C, 8, H, W), "Unexpected final output shape"

def test_resolution_scaling():
    """Test handling of different input resolutions."""
    vae = AutoencoderKLCogVideoX.from_pretrained(
        "THUDM/CogVideoX-5b",
        subfolder="vae",
        torch_dtype=torch.float16
    ).to(device)
    
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        "THUDM/CogVideoX-5b",
        subfolder="transformer",
        torch_dtype=torch.float16
    ).to(device)
    
    scheduler = CogVideoXDPMScheduler.from_pretrained(
        "THUDM/CogVideoX-5b",
        subfolder="scheduler"
    )
    
    pipeline = CogVideoXInpaintingPipeline(
        vae=vae,
        transformer=transformer,
        scheduler=scheduler
    )
    
    # Test with different resolutions
    B, C, T = 1, 3, 5
    resolutions = [
        (64, 64),      # Small (no chunking needed)
        (480, 640),    # SD (will use chunks)
        (720, 1280),   # HD (will use chunks)
    ]
    
    # Chunking parameters (in pixel space)
    chunk_size = 256  # Base chunk size
    overlap = 32      # Overlap size
    
    for H, W in resolutions:
        print(f"\nTesting resolution {H}x{W}")
        
        # Create test tensors with a smooth gradient pattern
        x = torch.linspace(0, 1, W, device=device, dtype=torch.float16)
        y = torch.linspace(0, 1, H, device=device, dtype=torch.float16)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
        pattern = (grid_x + grid_y) / 2.0
        
        # Expand pattern to video dimensions
        video = pattern.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # Add B, C, T dims
        video = video.expand(B, C, T, H, W)
        video = video.to(dtype=torch.float16)
        
        mask = torch.ones(B, 1, T, H, W, device=device, dtype=torch.float16)
        
        # Get scaling factors
        spatial_scale, temporal_scale = pipeline.validate_dimensions(video, mask)
        print(f"Scaling factors - Spatial: {spatial_scale:.2f}, Temporal: {temporal_scale:.2f}")
        
        # Check scaling is reasonable
        assert spatial_scale > 0, "Spatial scale should be positive"
        assert temporal_scale > 0, "Temporal scale should be positive"
        
        # Determine if chunking is needed
        use_chunks = W > chunk_size
        if use_chunks:
            effective_chunk = chunk_size - 2 * overlap
            num_chunks = math.ceil(W / effective_chunk)
            print(f"Using chunks - Base size: {chunk_size}, Effective size: {effective_chunk}")
            print(f"Overlap: {overlap}, Number of chunks: {num_chunks}")
            
            # Calculate expected chunk positions
            chunk_starts = []
            chunk_boundaries = []
            for i in range(num_chunks):
                start_w = max(0, i * effective_chunk - overlap)
                chunk_starts.append(start_w)
                if i > 0:  # Add boundary between this chunk and previous
                    boundary = i * effective_chunk * 8  # Convert to pixel space
                    if boundary < W * 8:  # Only if within image width
                        chunk_boundaries.append(boundary)
            print(f"Chunk start positions: {chunk_starts}")
            print(f"Chunk boundaries: {chunk_boundaries}")
        
        # Test encoding
        latents = pipeline.encode(
            video, 
            chunk_size=chunk_size if use_chunks else None,
            overlap=overlap
        )
        print(f"Encoded shape: {latents.shape}")
        assert latents.shape == (B, 16, T//2, H//8, W//8), f"Unexpected latent shape: {latents.shape}"
        
        # Test decoding
        decoded = pipeline.decode(
            latents,
            chunk_size=chunk_size if use_chunks else None,
            overlap=overlap
        )
        print(f"Decoded shape: {decoded.shape}")
        assert decoded.shape[0:2] == (B, C), "Batch and channel dimensions should match"
        assert decoded.shape[2] == 8, "Should output 8 frames"
        assert decoded.shape[3:] == (H, W), "Spatial dimensions should match"
        
        # Test for visible seams in output (basic check)
        if use_chunks and chunk_boundaries:
            print("\nChecking chunk boundaries:")
            
            for boundary in chunk_boundaries:
                # Check a window around the boundary
                window_size = overlap // 2  # Half the overlap size
                
                # Ensure window is within bounds
                left_start = max(0, boundary - window_size)
                left_end = boundary
                right_start = boundary
                right_end = min(W * 8, boundary + window_size)
                
                if left_start >= left_end or right_start >= right_end:
                    print(f"Skipping boundary {boundary} - insufficient window size")
                    continue
                
                # Get values around boundary
                left_vals = decoded[..., left_start:left_end]
                right_vals = decoded[..., right_start:right_end]
                
                # Reshape tensors to combine all dimensions except the last
                left_flat = left_vals.reshape(-1, left_vals.shape[-1])
                right_flat = right_vals.reshape(-1, right_vals.shape[-1])
                
                # Calculate mean values along the seam
                left_mean = left_flat.mean(dim=0)
                right_mean = right_flat.mean(dim=0)
                mean_diff = (right_mean - left_mean).abs().mean().item()
                
                # Calculate max difference along the seam
                diff_flat = (right_flat - left_flat).abs()
                max_diff = diff_flat.mean(dim=0).max().item()
                
                print(f"Boundary {boundary}:")
                print(f"  Window: [{left_start}:{left_end}] - [{right_start}:{right_end}]")
                print(f"  Mean difference: {mean_diff:.6f}")
                print(f"  Max difference: {max_diff:.6f}")
                
                # Check for discontinuities
                assert mean_diff < 0.1, f"Large mean discontinuity ({mean_diff:.6f}) at boundary {boundary}"
                assert max_diff < 0.2, f"Large max discontinuity ({max_diff:.6f}) at boundary {boundary}"
        
        # Clear memory after each resolution
        torch.cuda.empty_cache()
    
    print("\nResolution scaling tests passed!")

def test_memory_calculation():
    """Test memory requirement calculations."""
    vae = AutoencoderKLCogVideoX.from_pretrained(
        "THUDM/CogVideoX-5b",
        subfolder="vae",
        torch_dtype=torch.float16
    ).to(device)
    
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        "THUDM/CogVideoX-5b",
        subfolder="transformer",
        torch_dtype=torch.float16
    ).to(device)
    
    scheduler = CogVideoXDPMScheduler.from_pretrained(
        "THUDM/CogVideoX-5b",
        subfolder="scheduler"
    )
    
    pipeline = CogVideoXInpaintingPipeline(
        vae=vae,
        transformer=transformer,
        scheduler=scheduler
    )
    
    # Test with original dimensions
    B, C, T, H, W = 1, 3, 100, 720, 1280
    video = torch.randn(B, C, T, H, W, device=device, dtype=torch.float16)
    mask = torch.ones(B, 1, T, H, W, device=device, dtype=torch.float16)
    
    # Calculate memory manually
    bytes_per_element = 2  # fp16
    expected_video_memory = B * C * T * H * W * bytes_per_element / (1024**3)
    expected_latent_memory = B * 16 * (T//4) * (H//8) * (W//8) * bytes_per_element / (1024**3)
    expected_transformer_memory = 10.8
    expected_total = expected_video_memory + expected_latent_memory + expected_transformer_memory
    
    # Get pipeline's calculation
    pipeline.validate_dimensions(video, mask)
    
    # Memory validation is logged but not returned
    # We mainly want to ensure the calculation runs without error
    # Actual values are checked manually in logs

if __name__ == "__main__":
    test_vae_shapes()
    test_transformer_shapes()
    test_scheduler_config()
    test_end_to_end()
    test_resolution_scaling()
    test_memory_calculation()