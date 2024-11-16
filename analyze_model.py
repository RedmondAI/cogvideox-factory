#!/usr/bin/env python3

import torch
from diffusers import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel, CogVideoXDPMScheduler
import json
from pathlib import Path
import logging
import torch.nn as nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_vae(vae):
    """Analyze VAE architecture and configuration."""
    logger.info("\n=== VAE Analysis ===")
    
    # Basic config
    logger.info("\nBasic Configuration:")
    logger.info(f"VAE Config: {vae.config}")
    logger.info(f"Model dtype: {vae.dtype}")
    logger.info(f"Device: {vae.device}")
    
    # Channel configuration
    logger.info("\nChannel Configuration:")
    logger.info(f"Input channels: {vae.config.in_channels}")
    logger.info(f"Output channels: {vae.config.out_channels}")
    logger.info(f"Latent channels: {vae.config.latent_channels}")
    
    # Architecture details
    logger.info("\nArchitecture Details:")
    logger.info(f"Down blocks: {vae.config.down_block_types}")
    logger.info(f"Up blocks: {vae.config.up_block_types}")
    logger.info(f"Block out channels: {vae.config.block_out_channels}")
    logger.info(f"Layers per block: {vae.config.layers_per_block}")
    
    # Temporal configuration
    logger.info("\nTemporal Configuration:")
    logger.info(f"Temporal compression ratio: {vae.config.temporal_compression_ratio}")
    logger.info(f"Sample height: {vae.config.sample_height}")
    logger.info(f"Sample width: {vae.config.sample_width}")
    
    # Test forward pass
    logger.info("\nTesting Forward Pass:")
    B, T, C, H, W = 1, 8, vae.config.in_channels, 64, 64
    test_input = torch.randn(B, T, C, H, W, device=vae.device, dtype=vae.dtype)
    with torch.no_grad():
        encoded = vae.encode(test_input)
        latents = encoded.latent_dist.sample()
        logger.info(f"Input shape: {test_input.shape}")
        logger.info(f"Latent shape: {latents.shape}")
        decoded = vae.decode(latents)
        logger.info(f"Output shape: {decoded.shape}")
        
    # Memory analysis
    logger.info("\nMemory Analysis:")
    total_params = sum(p.numel() for p in vae.parameters())
    trainable_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    return vae.config.latent_channels

def analyze_transformer(transformer):
    """Analyze transformer architecture and configuration."""
    logger.info("\n=== Transformer Analysis ===")
    
    # Basic config
    logger.info("\nBasic Configuration:")
    logger.info(f"Transformer Config: {transformer.config}")
    logger.info(f"Model dtype: {transformer.dtype}")
    logger.info(f"Device: {transformer.device}")
    
    # Architecture details
    logger.info("\nArchitecture Details:")
    logger.info(f"Hidden size: {transformer.config.hidden_size}")
    logger.info(f"Intermediate size: {transformer.config.intermediate_size}")
    logger.info(f"Num attention heads: {transformer.config.num_attention_heads}")
    logger.info(f"Num hidden layers: {transformer.config.num_hidden_layers}")
    
    # Input/Output configuration
    logger.info("\nInput/Output Configuration:")
    logger.info(f"In channels: {transformer.config.in_channels}")
    logger.info(f"Out channels: {transformer.config.out_channels}")
    
    # Patch embedding analysis
    logger.info("\nPatch Embedding Analysis:")
    if hasattr(transformer, 'patch_embed'):
        proj = transformer.patch_embed.proj
        logger.info(f"Proj in channels: {proj.in_channels}")
        logger.info(f"Proj out channels: {proj.out_channels}")
        logger.info(f"Kernel size: {proj.kernel_size}")
        logger.info(f"Stride: {proj.stride}")
        logger.info(f"Padding: {proj.padding}")
        logger.info(f"Proj weight shape: {proj.weight.shape}")
    
    # Test forward pass
    logger.info("\nTesting Forward Pass:")
    B, T, C, H, W = 1, 8, transformer.config.in_channels, 64, 64
    test_input = torch.randn(B, T, C, H, W, device=transformer.device, dtype=transformer.dtype)
    timesteps = torch.zeros(B, dtype=torch.long, device=transformer.device)
    with torch.no_grad():
        output = transformer(test_input, timestep=timesteps)
        logger.info(f"Input shape: {test_input.shape}")
        logger.info(f"Output shape: {output[0].shape if isinstance(output, tuple) else output.shape}")
    
    # Memory analysis
    logger.info("\nMemory Analysis:")
    total_params = sum(p.numel() for p in transformer.parameters())
    trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    return transformer.config.in_channels

def analyze_scheduler(scheduler):
    """Analyze scheduler configuration."""
    logger.info("\n=== Scheduler Analysis ===")
    
    # Basic config
    logger.info("\nBasic Configuration:")
    logger.info(f"Scheduler Config: {scheduler.config}")
    logger.info(f"Prediction type: {scheduler.config.prediction_type}")
    logger.info(f"Beta schedule: {scheduler.config.beta_schedule}")
    logger.info(f"Num train timesteps: {scheduler.config.num_train_timesteps}")
    
    # Test noise addition
    logger.info("\nTesting Noise Addition:")
    B, T, C, H, W = 1, 8, 3, 64, 64
    test_input = torch.randn(B, T, C, H, W)
    noise = torch.randn_like(test_input)
    timesteps = torch.zeros(B, dtype=torch.long)
    noisy = scheduler.add_noise(test_input, noise, timesteps)
    logger.info(f"Input shape: {test_input.shape}")
    logger.info(f"Noisy output shape: {noisy.shape}")

def main():
    """Analyze CogVideoX model components."""
    logger.info("Loading models...")
    
    # Load models with explicit dtype
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
    
    scheduler = CogVideoXDPMScheduler.from_pretrained(
        "THUDM/CogVideoX-5b",
        subfolder="scheduler",
    )
    
    # Analyze each component
    vae_channels = analyze_vae(vae)
    transformer_channels = analyze_transformer(transformer)
    analyze_scheduler(scheduler)
    
    # Verify channel compatibility
    logger.info("\n=== Channel Compatibility Check ===")
    logger.info(f"VAE latent channels: {vae_channels}")
    logger.info(f"Transformer input channels: {transformer_channels}")
    if vae_channels != transformer_channels:
        logger.warning(
            f"Channel mismatch: VAE latent channels ({vae_channels}) != "
            f"Transformer input channels ({transformer_channels})"
        )
    
    # Save analysis to file
    output = {
        "vae_config": vae.config,
        "transformer_config": transformer.config,
        "scheduler_config": scheduler.config,
        "channel_analysis": {
            "vae_latent_channels": vae_channels,
            "transformer_in_channels": transformer_channels,
            "channel_match": vae_channels == transformer_channels,
        }
    }
    
    output_file = Path("model_analysis.json")
    with output_file.open("w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"\nAnalysis saved to {output_file}")

if __name__ == "__main__":
    main()
