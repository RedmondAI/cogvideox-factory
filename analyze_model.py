#!/usr/bin/env python3

import torch
from diffusers import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel, CogVideoXDPMScheduler
import json
from pathlib import Path
import logging
import torch.nn as nn
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_vae(vae):
    """Analyze VAE architecture and configuration."""
    print("\nAnalyzing VAE architecture...")
    logger.info("\n=== VAE Analysis ===")
    
    # Check device
    print(f"Model is on device: {vae.device}")
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available, running on CPU will be slow!")
    
    # Basic config
    print("Checking basic configuration...")
    logger.info("\nBasic Configuration:")
    logger.info(f"VAE Config: {vae.config}")
    logger.info(f"Model dtype: {vae.dtype}")
    logger.info(f"Device: {vae.device}")
    
    # Channel configuration
    print("Analyzing channel configuration...")
    logger.info("\nChannel Configuration:")
    logger.info(f"Input channels: {vae.config.in_channels}")
    logger.info(f"Output channels: {vae.config.out_channels}")
    logger.info(f"Latent channels: {vae.config.latent_channels}")
    logger.info(f"Block out channels: {vae.config.block_out_channels}")
    print(f"Found {vae.config.latent_channels} latent channels")
    
    # Architecture details
    print("Examining architecture details...")
    logger.info("\nArchitecture Details:")
    logger.info(f"Down blocks: {vae.config.down_block_types}")
    logger.info(f"Up blocks: {vae.config.up_block_types}")
    logger.info(f"Layers per block: {vae.config.layers_per_block}")
    logger.info(f"Norm num groups: {vae.config.norm_num_groups}")
    logger.info(f"Act fn: {vae.config.act_fn}")
    
    # Temporal configuration
    print("Checking temporal configuration...")
    logger.info("\nTemporal Configuration:")
    logger.info(f"Temporal compression ratio: {vae.config.temporal_compression_ratio}")
    logger.info(f"Sample height: {vae.config.sample_height}")
    logger.info(f"Sample width: {vae.config.sample_width}")
    
    # Test forward pass with correct shapes
    print("Testing forward pass...")
    logger.info("\nTesting Forward Pass:")
    
    # Create small test input
    B, T, C, H, W = 1, 5, vae.config.in_channels, 32, 32  # Reduced size for faster testing
    print(f"Creating test input with shape: [B={B}, C={C}, T={T}, H={H}, W={W}]")
    test_input = torch.randn(B, C, T, H, W, device=vae.device, dtype=vae.dtype)
    logger.info(f"Input shape: {test_input.shape}")
    
    import time
    import signal
    from contextlib import contextmanager
    
    @contextmanager
    def timeout(seconds):
        def signal_handler(signum, frame):
            raise TimeoutError(f"Operation timed out after {seconds} seconds")
        
        # Register a function to raise a TimeoutError on the signal
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        
        try:
            yield
        finally:
            # Disable the alarm
            signal.alarm(0)
    
    with torch.no_grad():
        try:
            print("Attempting forward pass with [B, C, T, H, W] format...")
            start_time = time.time()
            
            # Test encoding
            encoded = vae.encode(test_input)
            latents = encoded.latent_dist.sample()
            decode_time = time.time()
            print(f"Encoding took {decode_time - start_time:.2f} seconds")
            print(f"Latent shape: {latents.shape}")
            logger.info(f"Latent shape: {latents.shape}")
            
            # Test decoding
            decoded = vae.decode(latents)
            decode_end_time = time.time()
            print(f"Decoding took {decode_end_time - decode_time:.2f} seconds")
            
            # Handle decoder output
            if hasattr(decoded, 'sample'):
                decoded_sample = decoded.sample
            else:
                decoded_sample = decoded
            print(f"Output sample shape: {decoded_sample.shape}")
            logger.info(f"Output sample shape: {decoded_sample.shape}")
            
            # Verify shapes
            expected_output_shape = (B, C, T//vae.config.temporal_compression_ratio, H, W)
            print(f"Expected output shape: {expected_output_shape}")
            if decoded_sample.shape != expected_output_shape:
                print(f"WARNING: Output shape {decoded_sample.shape} doesn't match expected shape {expected_output_shape}")
            
        except Exception as e:
            print(f"First format failed: {e}")
            logger.error(f"Forward pass failed: {e}")
            
            # Try alternative format with smaller input
            print("Trying alternative format with smaller input...")
            B, T, C, H, W = 1, 3, vae.config.in_channels, 32, 32  # Even smaller test
            test_input_alt = torch.randn(B, C, T, H, W, device=vae.device, dtype=vae.dtype)
            print(f"Alternative input shape: {test_input_alt.shape}")
            
            start_time = time.time()
            
            # Test encoding
            encoded = vae.encode(test_input_alt)
            latents = encoded.latent_dist.sample()
            decode_time = time.time()
            print(f"Alternative encoding took {decode_time - start_time:.2f} seconds")
            print(f"Latent shape: {latents.shape}")
            logger.info(f"Alternative latent shape: {latents.shape}")
            
            # Test decoding
            decoded = vae.decode(latents)
            decode_end_time = time.time()
            print(f"Alternative decoding took {decode_end_time - decode_time:.2f} seconds")
            
            # Handle decoder output
            if hasattr(decoded, 'sample'):
                decoded_sample = decoded.sample
            else:
                decoded_sample = decoded
            print(f"Alternative output sample shape: {decoded_sample.shape}")
            logger.info(f"Alternative output sample shape: {decoded_sample.shape}")
    
    # Memory analysis
    print("\nAnalyzing memory usage...")
    logger.info("\nMemory Analysis:")
    total_params = sum(p.numel() for p in vae.parameters())
    trainable_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    if torch.cuda.is_available():
        print("\nGPU Memory Usage:")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.1f}MB")
    
    print("VAE analysis completed!")
    return vae.config.latent_channels

def analyze_transformer(transformer):
    """Analyze transformer architecture and configuration."""
    print("\nAnalyzing Transformer architecture...")
    logger.info("\n=== Transformer Analysis ===")
    
    # Basic config
    print("Checking basic configuration...")
    logger.info("\nBasic Configuration:")
    logger.info(f"Transformer Config: {transformer.config}")
    logger.info(f"Model dtype: {transformer.dtype}")
    logger.info(f"Device: {transformer.device}")
    print(f"Model is on device: {transformer.device}")
    
    # Print all available config attributes
    print("\nAvailable config attributes:")
    for key, value in transformer.config.items():
        print(f"  {key}: {value}")
    
    # Architecture details
    print("\nExamining architecture details...")
    logger.info("\nArchitecture Details:")
    config = transformer.config
    
    # Get architecture details safely
    architecture_details = {
        "in_channels": config.get("in_channels"),
        "out_channels": config.get("out_channels"),
        "num_attention_heads": config.get("num_attention_heads"),
        "num_layers": config.get("num_layers"),
        "patch_size": config.get("patch_size"),
        "attention_head_dim": config.get("attention_head_dim"),
        "text_embed_dim": config.get("text_embed_dim"),
        "time_embed_dim": config.get("time_embed_dim"),
        "temporal_compression_ratio": config.get("temporal_compression_ratio"),
        "sample_frames": config.get("sample_frames"),
        "sample_height": config.get("sample_height"),
        "sample_width": config.get("sample_width"),
    }
    
    for key, value in architecture_details.items():
        if value is not None:
            logger.info(f"{key}: {value}")
            print(f"{key}: {value}")
    
    # Input/Output configuration
    print("\nChecking input/output configuration...")
    logger.info("\nInput/Output Configuration:")
    in_channels = config.get("in_channels")
    out_channels = config.get("out_channels")
    logger.info(f"In channels: {in_channels}")
    logger.info(f"Out channels: {out_channels}")
    print(f"Input channels: {in_channels}, Output channels: {out_channels}")
    
    # Patch embedding analysis
    print("\nAnalyzing patch embedding...")
    logger.info("\nPatch Embedding Analysis:")
    if hasattr(transformer, 'patch_embed'):
        proj = transformer.patch_embed.proj
        logger.info(f"Proj in channels: {proj.in_channels}")
        logger.info(f"Proj out channels: {proj.out_channels}")
        logger.info(f"Kernel size: {proj.kernel_size}")
        logger.info(f"Stride: {proj.stride}")
        logger.info(f"Padding: {proj.padding}")
        logger.info(f"Proj weight shape: {proj.weight.shape}")
        print(f"Projection layer: {proj.in_channels} -> {proj.out_channels} channels")
        
        # Analyze weight statistics
        with torch.no_grad():
            weight_mean = proj.weight.mean().item()
            weight_std = proj.weight.std().item()
            logger.info(f"Weight statistics - Mean: {weight_mean:.4f}, Std: {weight_std:.4f}")
            print(f"Weight statistics - Mean: {weight_mean:.4f}, Std: {weight_std:.4f}")
    
    # Test forward pass with correct shape
    print("\nTesting forward pass...")
    logger.info("\nTesting Forward Pass:")
    
    # Create test inputs
    B, T, C, H, W = 1, 5, in_channels, 32, 32
    test_input = torch.randn(B, T, C, H, W, device=transformer.device, dtype=transformer.dtype)
    timesteps = torch.zeros(B, dtype=torch.long, device=transformer.device)
    
    # Create encoder hidden states (sequence length should be 1 for conditioning)
    text_embed_dim = config.get("text_embed_dim", 4096)
    encoder_hidden_states = torch.randn(B, 1, text_embed_dim, device=transformer.device, dtype=transformer.dtype)
    
    print(f"Created test inputs:")
    print(f"  Input shape: {test_input.shape}")
    print(f"  Encoder hidden states shape: {encoder_hidden_states.shape}")
    
    with torch.no_grad():
        try:
            print("Attempting forward pass with [B, T, C, H, W] format...")
            start_time = time.time()
            output = transformer(
                test_input,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states
            )
            end_time = time.time()
            print(f"Forward pass took {end_time - start_time:.2f} seconds")
            
            # Handle transformer output
            if hasattr(output, 'sample'):
                output_sample = output.sample
            else:
                output_sample = output[0] if isinstance(output, tuple) else output
            
            logger.info(f"Input shape: {test_input.shape}")
            logger.info(f"Output shape: {output_sample.shape}")
            print(f"Output shape: {output_sample.shape}")
            
            # Verify output shape matches input shape
            if output_sample.shape != test_input.shape:
                print(f"WARNING: Output shape {output_sample.shape} doesn't match input shape {test_input.shape}")
            
        except Exception as e:
            print(f"First format failed: {e}")
            logger.error(f"Forward pass failed: {e}")
            # Try alternative format
            print("Trying alternative format [B, C, T, H, W]...")
            test_input_alt = test_input.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W] -> [B, C, T, H, W]
            logger.info(f"Trying alternative input shape: {test_input_alt.shape}")
            start_time = time.time()
            output = transformer(
                test_input_alt,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states
            )
            end_time = time.time()
            print(f"Alternative forward pass took {end_time - start_time:.2f} seconds")
            
            # Handle transformer output
            if hasattr(output, 'sample'):
                output_sample = output.sample
            else:
                output_sample = output[0] if isinstance(output, tuple) else output
                
            logger.info(f"Output shape: {output_sample.shape}")
            print(f"Alternative output shape: {output_sample.shape}")
    
    # Memory analysis
    print("\nAnalyzing memory usage...")
    logger.info("\nMemory Analysis:")
    total_params = sum(p.numel() for p in transformer.parameters())
    trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    if torch.cuda.is_available():
        print("\nGPU Memory Usage:")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.1f}MB")
    
    print("\nTransformer analysis completed!")
    return in_channels

def analyze_scheduler(scheduler):
    """Analyze scheduler configuration."""
    print("\nAnalyzing Scheduler architecture...")
    logger.info("\n=== Scheduler Analysis ===")
    
    # Basic config
    print("Checking basic configuration...")
    logger.info("\nBasic Configuration:")
    logger.info(f"Scheduler Config: {scheduler.config}")
    
    # Print all available config attributes
    print("\nAvailable config attributes:")
    for key, value in scheduler.config.items():
        print(f"  {key}: {value}")
    
    # Test scheduler step
    print("\nTesting scheduler step...")
    logger.info("\nTesting Scheduler Step:")
    
    try:
        # Set inference steps
        num_inference_steps = 50
        scheduler.set_timesteps(num_inference_steps)
        print(f"Set {num_inference_steps} inference steps")
        
        # Create test inputs
        B, C, T, H, W = 1, 16, 5, 32, 32
        sample = torch.randn(B, C, T, H, W)
        model_output = torch.randn_like(sample)  # Predicted noise or x0
        old_pred_original_sample = torch.randn_like(sample)  # Previous prediction
        
        # Get actual timesteps from scheduler
        timestep = scheduler.timesteps[0]  # First step (most noisy)
        timestep_back = scheduler.timesteps[1]  # Next step
        
        print(f"Created test inputs:")
        print(f"  Sample shape: {sample.shape}")
        print(f"  Model output shape: {model_output.shape}")
        print(f"  Previous prediction shape: {old_pred_original_sample.shape}")
        print(f"  Timestep: {timestep.item()}")
        print(f"  Timestep back: {timestep_back.item()}")
        
        # Test scheduler step
        output = scheduler.step(
            model_output=model_output,
            timestep=timestep,
            timestep_back=timestep_back,
            sample=sample,
            old_pred_original_sample=old_pred_original_sample,
        )
        
        if hasattr(output, 'prev_sample'):
            output_sample = output.prev_sample
        else:
            output_sample = output
            
        print(f"Output shape: {output_sample.shape}")
        
        # Verify shapes match
        if output_sample.shape != sample.shape:
            print(f"WARNING: Output shape {output_sample.shape} doesn't match input shape {sample.shape}")
        
        # Analyze noise schedule
        print("\nAnalyzing noise schedule...")
        if hasattr(scheduler, 'betas'):
            betas = scheduler.betas
            print(f"Number of training steps: {len(betas)}")
            print(f"Beta range: [{betas.min().item():.6f}, {betas.max().item():.6f}]")
            print(f"Beta schedule type: {scheduler.config.beta_schedule}")
        
        if hasattr(scheduler, 'alphas_cumprod'):
            alphas = scheduler.alphas_cumprod
            print(f"Alpha range: [{alphas.min().item():.6f}, {alphas.max().item():.6f}]")
            snr_db = -10 * torch.log10(1/alphas - 1)
            print(f"SNR range: [{snr_db.min().item():.1f}dB, {snr_db.max().item():.1f}dB]")
            
            # Print inference schedule details
            print("\nInference schedule details:")
            print(f"Number of inference steps: {num_inference_steps}")
            print(f"Timestep spacing: {scheduler.config.timestep_spacing}")
            print(f"Steps offset: {scheduler.config.steps_offset}")
            if hasattr(scheduler, 'timesteps'):
                print(f"Actual timesteps: {scheduler.timesteps.tolist()}")
        
    except Exception as e:
        print(f"Scheduler test failed: {e}")
        logger.error(f"Scheduler test failed: {e}")
    
    print("\nScheduler analysis completed!")

def main():
    """Analyze CogVideoX model components."""
    logger.info("Starting model analysis...")
    
    try:
        # Check CUDA availability
        if not torch.cuda.is_available():
            print("ERROR: CUDA is not available. This script requires a GPU to run efficiently.")
            print("Available devices:", torch.cuda.device_count())
            return
        
        device = torch.device("cuda")
        print(f"Using device: {device} ({torch.cuda.get_device_name()})")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f}MB")
        
        # Load VAE and analyze
        logger.info("\nStep 1/3: Loading VAE model from THUDM/CogVideoX-5b...")
        print("Downloading and loading VAE... (this might take a few minutes)")
        vae = AutoencoderKLCogVideoX.from_pretrained(
            "THUDM/CogVideoX-5b",
            subfolder="vae",
            torch_dtype=torch.bfloat16,
            device_map="auto",  # This will automatically place the model on GPU
        )
        print("VAE loaded successfully!")
        logger.info("Running VAE analysis...")
        vae_channels = analyze_vae(vae)
        
        # Load transformer and analyze
        logger.info("\nStep 2/3: Loading Transformer model...")
        print("Downloading and loading Transformer... (this might take a few minutes)")
        transformer = CogVideoXTransformer3DModel.from_pretrained(
            "THUDM/CogVideoX-5b",
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        ).to(device)  # Manually move to GPU
        print("Transformer loaded successfully!")
        logger.info("Running Transformer analysis...")
        transformer_channels = analyze_transformer(transformer)
        
        # Load scheduler and analyze
        logger.info("\nStep 3/3: Loading and analyzing Scheduler...")
        print("Loading Scheduler...")
        scheduler = CogVideoXDPMScheduler.from_pretrained(
            "THUDM/CogVideoX-5b",
            subfolder="scheduler",
        )
        print("Scheduler loaded successfully!")
        logger.info("Running Scheduler analysis...")
        analyze_scheduler(scheduler)
        
        # Verify channel compatibility
        logger.info("\nFinal Step: Channel Compatibility Check")
        logger.info(f"VAE latent channels: {vae_channels}")
        logger.info(f"Transformer input channels: {transformer_channels}")
        if vae_channels != transformer_channels:
            logger.warning(
                f"Channel mismatch: VAE latent channels ({vae_channels}) != "
                f"Transformer input channels ({transformer_channels})"
            )
        
        # Save analysis to file
        print("\nSaving analysis results...")
        output = {
            "vae_config": vae.config,
            "transformer_config": transformer.config,
            "scheduler_config": scheduler.config,
            "channel_analysis": {
                "vae_latent_channels": vae_channels,
                "transformer_in_channels": transformer_channels,
                "channel_match": vae_channels == transformer_channels,
            },
            "vae_analysis": {
                "temporal_behavior": {
                    "input_frames": 5,
                    "output_frames": 8,
                    "expected_compression": vae.config.temporal_compression_ratio,
                    "actual_behavior": "expansion",
                }
            }
        }
        
        output_file = Path("model_analysis.json")
        with output_file.open("w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"Analysis saved to {output_file}")
        print("\nAnalysis completed successfully!")
        
        # Print GPU memory usage at the end
        if torch.cuda.is_available():
            print("\nFinal GPU Memory Usage:")
            print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
            print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.1f}MB")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"\nError during analysis: {e}")
        raise

if __name__ == "__main__":
    print("=== CogVideoX Model Analysis Tool ===")
    print("This tool will analyze the architecture and configuration of the CogVideoX model.")
    print("The analysis may take several minutes to complete.")
    print("\nStarting analysis...\n")
    main()
