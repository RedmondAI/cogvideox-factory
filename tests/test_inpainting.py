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
        timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (latents.shape[0],))
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
            encoder_hidden_states=encoder_hidden_states,
            timestep=timesteps,
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
        test_full_training_cycle()
        
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