"""Utility functions for CogVideoX tests."""

import torch
import torch.nn as nn

def create_layer_norm(hidden_dim: int, model_config, device, dtype):
    """Create layer normalization with correct configuration."""
    return nn.LayerNorm(
        hidden_dim,
        eps=model_config.norm_eps,
        elementwise_affine=model_config.norm_elementwise_affine,
        device=device,
        dtype=dtype
    )
