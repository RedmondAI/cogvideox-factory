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
    
    # Create encoder hidden states based on text_embed_dim
    text_embed_dim = config.get("text_embed_dim", 4096)
    encoder_hidden_states = torch.randn(B, T, text_embed_dim, device=transformer.device, dtype=transformer.dtype)
    
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
            
            if isinstance(output, tuple):
                output = output[0]
            logger.info(f"Input shape: {test_input.shape}")
            logger.info(f"Output shape: {output.shape}")
            print(f"Output shape: {output.shape}")
            
            # Verify output shape matches input shape
            if output.shape != test_input.shape:
                print(f"WARNING: Output shape {output.shape} doesn't match input shape {test_input.shape}")
            
        except Exception as e:
            print(f"First format failed: {e}")
            logger.error(f"Forward pass failed: {e}")
            # Try alternative format
            print("Trying alternative format [B, C, T, H, W]...")
            test_input_alt = test_input.permute(0, 2, 1, 3, 4)  # Try different permutation
            logger.info(f"Trying alternative input shape: {test_input_alt.shape}")
            start_time = time.time()
            output = transformer(
                test_input_alt,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states
            )
            end_time = time.time()
            print(f"Alternative forward pass took {end_time - start_time:.2f} seconds")
            
            if isinstance(output, tuple):
                output = output[0]
            logger.info(f"Output shape: {output.shape}")
            print(f"Alternative output shape: {output.shape}")
    
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
    logger.info("\n=== Scheduler Analysis ===")
    
    # Basic config
    logger.info("\nBasic Configuration:")
    logger.info(f"Scheduler Config: {scheduler.config}")
    logger.info(f"Prediction type: {scheduler.config.prediction_type}")
    logger.info(f"Beta schedule: {scheduler.config.beta_schedule}")
    logger.info(f"Num train timesteps: {scheduler.config.num_train_timesteps}")
    
    # Analyze beta schedule
    logger.info("\nBeta Schedule Analysis:")
    logger.info(f"Beta start: {scheduler.config.beta_start}")
    logger.info(f"Beta end: {scheduler.config.beta_end}")
    if hasattr(scheduler, 'betas'):
        betas = scheduler.betas
        logger.info(f"Beta schedule shape: {betas.shape}")
        logger.info(f"Beta min: {betas.min().item():.6f}")
        logger.info(f"Beta max: {betas.max().item():.6f}")
        logger.info(f"Beta mean: {betas.mean().item():.6f}")
    
    # Analyze alphas
    if hasattr(scheduler, 'alphas_cumprod'):
        logger.info("\nAlpha Analysis:")
        alphas_cumprod = scheduler.alphas_cumprod
        logger.info(f"Alphas cumprod shape: {alphas_cumprod.shape}")
        logger.info(f"Alphas cumprod min: {alphas_cumprod.min().item():.6f}")
        logger.info(f"Alphas cumprod max: {alphas_cumprod.max().item():.6f}")
        logger.info(f"Alphas cumprod mean: {alphas_cumprod.mean().item():.6f}")
    
    # Test noise addition
    logger.info("\nTesting Noise Addition:")
    B, T, C, H, W = 1, 5, 3, 64, 64
    test_input = torch.randn(B, C, T, H, W)  # Try [B, C, T, H, W] format
    noise = torch.randn_like(test_input)
    timesteps = torch.zeros(B, dtype=torch.long)
    
    try:
        noisy = scheduler.add_noise(test_input, noise, timesteps)
        logger.info(f"Input shape: {test_input.shape}")
        logger.info(f"Noisy output shape: {noisy.shape}")
    except Exception as e:
        logger.error(f"First format failed: {e}")
        # Try alternative format
        test_input_alt = test_input.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
        noise_alt = noise.permute(0, 2, 1, 3, 4)
        noisy = scheduler.add_noise(test_input_alt, noise_alt, timesteps)
        logger.info(f"Alternative input shape: {test_input_alt.shape}")
        logger.info(f"Alternative noisy output shape: {noisy.shape}")
    
    # Test step function
    logger.info("\nTesting Step Function:")
    try:
        model_output = torch.randn_like(test_input)
        denoised = scheduler.step(model_output, timesteps[0], test_input)
        logger.info(f"Step output type: {type(denoised)}")
        logger.info(f"Prev sample shape: {denoised.prev_sample.shape}")
    except Exception as e:
        logger.error(f"Step function test failed: {e}")
        model_output_alt = model_output.permute(0, 2, 1, 3, 4)
        test_input_alt = test_input.permute(0, 2, 1, 3, 4)
        denoised = scheduler.step(model_output_alt, timesteps[0], test_input_alt)
        logger.info(f"Alternative step output type: {type(denoised)}")
        logger.info(f"Alternative prev sample shape: {denoised.prev_sample.shape}")

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
