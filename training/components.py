"""Core components for CogVideoX training."""

import torch
import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

def pad_to_multiple(x: torch.Tensor, multiple: int = 64, max_dim: int = 2048) -> tuple[torch.Tensor, tuple[int, int]]:
    """Pad tensor to multiple with size safety check."""
    h, w = x.shape[-2:]
    print(f"Input dimensions: h={h}, w={w}")
    
    # Check if input dimensions exceed maximum
    if h >= max_dim or w >= max_dim:
        raise ValueError(f"Input dimensions ({h}, {w}) exceed or equal maximum safe size {max_dim}")
    
    # Calculate padding
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    print(f"Padding amounts: pad_h={pad_h}, pad_w={pad_w}")
    print(f"Final dimensions will be: h={h+pad_h}, w={w+pad_w}")
    
    # Check if padded dimensions would exceed maximum
    if h + pad_h >= max_dim or w + pad_w >= max_dim:
        raise ValueError(f"Padded dimensions ({h+pad_h}, {w+pad_w}) would exceed or equal maximum size {max_dim}")
    
    # Pad tensor
    x_padded = F.pad(x, (0, pad_w, 0, pad_h))
    return x_padded, (pad_h, pad_w)

def unpad(x: torch.Tensor, pad_sizes: tuple[int, int]) -> torch.Tensor:
    """Remove padding from tensor."""
    pad_h, pad_w = pad_sizes
    if pad_h > 0:
        x = x[..., :-pad_h, :]
    if pad_w > 0:
        x = x[..., :-pad_w]
    return x

def temporal_smooth(frames: torch.Tensor, window_size: int = 5) -> torch.Tensor:
    """Apply temporal smoothing at chunk boundaries.
    
    Args:
        frames: Input frames tensor of shape [B, T, C, H, W]
        window_size: Size of the smoothing window
    
    Returns:
        Smoothed frames tensor of same shape
    """
    if frames.shape[1] <= window_size:
        return frames
    
    smoothed = frames.clone()
    half_window = window_size // 2
    
    # Create weights for smooth transition
    weights = torch.linspace(0, 1, window_size, device=frames.device)
    weights = weights.view(1, -1, 1, 1, 1)
    
    # Apply smoothing in the middle region
    mid_start = frames.shape[1] // 2 - half_window
    mid_end = mid_start + window_size
    
    # Get the frames before and after transition
    pre_transition = frames[:, mid_start-1:mid_start-1+window_size]
    post_transition = frames[:, mid_start:mid_start+window_size]
    
    # Blend between pre and post transition frames
    smoothed[:, mid_start:mid_end] = (
        pre_transition * (1 - weights) +
        post_transition * weights
    )
    
    return smoothed

def compute_metrics(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    """Compute metrics with memory efficiency.
    
    Args:
        pred: Predicted frames [B, T, C, H, W]
        gt: Ground truth frames [B, T, C, H, W]
        mask: Binary mask [B, T, 1, H, W]
    
    Returns:
        Dict of metrics including PSNR, SSIM, and temporal consistency
    """
    metrics = {}
    
    # Move to CPU for metric computation
    with torch.no_grad():
        pred_cpu = pred.detach().cpu()
        gt_cpu = gt.detach().cpu()
        mask_cpu = mask.detach().cpu()
        
        # Initialize metric computers on CPU
        psnr = PeakSignalNoiseRatio()
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0)  # Specify data range
        
        # Compute metrics for masked regions
        masked_pred = pred_cpu * mask_cpu
        masked_gt = gt_cpu * mask_cpu
        
        # Reshape for 2D metrics (combine batch and time dimensions)
        B, T, C, H, W = pred_cpu.shape
        masked_pred_2d = masked_pred.view(B*T, C, H, W)
        masked_gt_2d = masked_gt.view(B*T, C, H, W)
        
        metrics['masked_psnr'] = psnr(masked_pred_2d, masked_gt_2d)
        metrics['masked_ssim'] = ssim(masked_pred_2d, masked_gt_2d)
        
        # Temporal consistency (compute on GPU if memory allows)
        if pred.device.type == "cuda" and torch.cuda.memory_allocated() < torch.cuda.max_memory_allocated() * 0.8:
            device = pred.device
            pred_diff = (pred[:, 1:] - pred[:, :-1]).abs().mean()
            gt_diff = (gt[:, 1:] - gt[:, :-1]).abs().mean()
            metrics['temporal_consistency'] = 1.0 - (pred_diff - gt_diff).abs().item()
        else:
            pred_diff = (pred_cpu[:, 1:] - pred_cpu[:, :-1]).abs().mean()
            gt_diff = (gt_cpu[:, 1:] - gt_cpu[:, :-1]).abs().mean()
            metrics['temporal_consistency'] = 1.0 - (pred_diff - gt_diff).abs().item()
    
    return metrics

def compute_loss(model_pred: torch.Tensor, noise: torch.Tensor, mask: torch.Tensor, latents: torch.Tensor | None = None) -> torch.Tensor:
    """Compute the loss for video inpainting training.
    
    Args:
        model_pred: Model prediction tensor [B, T, C, H, W]
        noise: Target noise tensor [B, T, C, H, W]
        mask: Binary mask tensor [B, T, 1, H, W]
        latents: Optional original latents for perceptual loss [B, T, C, H, W]
    
    Returns:
        Loss value as a scalar tensor
    """
    # Compute MSE loss in masked regions
    mse_loss = F.mse_loss(model_pred * mask, noise * mask, reduction='none')
    
    # Average over all dimensions except batch
    mse_loss = mse_loss.mean(dim=[1, 2, 3, 4])
    
    # Add temporal consistency loss if sequence length > 1
    if model_pred.shape[1] > 1:
        # Compute temporal gradients
        pred_grad = model_pred[:, 1:] - model_pred[:, :-1]
        noise_grad = noise[:, 1:] - noise[:, :-1]
        mask_grad = mask[:, 1:]  # Mask for gradient regions
        
        # Temporal consistency loss in masked regions
        temp_loss = F.mse_loss(
            pred_grad * mask_grad,
            noise_grad * mask_grad,
            reduction='none'
        ).mean(dim=[1, 2, 3, 4])
        
        # Combine losses
        return mse_loss + 0.5 * temp_loss
    
    return mse_loss

def compute_loss_v_pred(noise_pred: torch.Tensor, noise: torch.Tensor, alpha_prod_t: torch.Tensor, sigma_t: torch.Tensor, mask: torch.Tensor | None = None, noisy_frames: torch.Tensor | None = None) -> torch.Tensor:
    """Compute v-prediction loss."""
    # Compute target
    v_target = noise * alpha_prod_t.sqrt() - sigma_t * noisy_frames if noisy_frames is not None else noise
    
    # Apply mask if provided
    if mask is not None:
        return F.mse_loss(noise_pred * mask, v_target * mask)
    
    return F.mse_loss(noise_pred, v_target)

def compute_loss_v_pred_with_snr(noise_pred: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor, scheduler, mask: torch.Tensor | None = None, noisy_frames: torch.Tensor | None = None) -> torch.Tensor:
    """Compute v-prediction loss with SNR scaling."""
    # Get scheduler parameters and ensure float16 precision
    alphas_cumprod = scheduler.alphas_cumprod.to(device=noise_pred.device, dtype=noise_pred.dtype)
    alpha_prod_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1, 1)
    sigma_t = torch.sqrt(1 - alpha_prod_t)
    
    # Ensure all tensors are in float16
    noise = noise.to(dtype=noise_pred.dtype)
    if noisy_frames is not None:
        noisy_frames = noisy_frames.to(dtype=noise_pred.dtype)
    if mask is not None:
        mask = mask.to(dtype=noise_pred.dtype)
    
    # Adjust noise and noisy_frames to match model output spatial dimensions
    if noise.shape != noise_pred.shape:
        H_out, W_out = noise_pred.shape[3:]
        noise = torch.nn.functional.interpolate(
            noise.reshape(-1, *noise.shape[2:]), 
            size=(H_out, W_out), 
            mode='bilinear'
        ).reshape(*noise.shape[:2], *noise_pred.shape[2:])
    
    if noisy_frames is not None and noisy_frames.shape != noise_pred.shape:
        H_out, W_out = noise_pred.shape[3:]
        noisy_frames = torch.nn.functional.interpolate(
            noisy_frames.reshape(-1, *noisy_frames.shape[2:]), 
            size=(H_out, W_out), 
            mode='bilinear'
        ).reshape(*noisy_frames.shape[:2], *noise_pred.shape[2:])
    
    # Compute target
    v_target = noise * alpha_prod_t.sqrt() - sigma_t * noisy_frames if noisy_frames is not None else noise
    
    # Apply mask if provided
    if mask is not None:
        # Adjust mask to match model output spatial dimensions
        if mask.shape != noise_pred.shape:
            H_out, W_out = noise_pred.shape[3:]
            mask = torch.nn.functional.interpolate(
                mask.reshape(-1, *mask.shape[2:]), 
                size=(H_out, W_out), 
                mode='nearest'
            ).reshape(*mask.shape[:2], *noise_pred.shape[2:])
        masked_pred = noise_pred * mask
        masked_target = v_target * mask
        return F.mse_loss(masked_pred, masked_target)
    
    return F.mse_loss(noise_pred, v_target)

def handle_vae_temporal_output(decoded: torch.Tensor, target_frames: int) -> torch.Tensor:
    """Handle potential temporal expansion from VAE.
    
    Args:
        decoded: Tensor from VAE decoder with shape [B, C, T, H, W]
        target_frames: Number of frames expected in output
        
    Returns:
        Tensor with target number of frames [B, C, target_frames, H, W]
    """
    if decoded.shape[2] == target_frames:
        return decoded
    
    # If VAE expanded temporal dimension, extract center frames
    if decoded.shape[2] > target_frames:
        start_idx = (decoded.shape[2] - target_frames) // 2
        return decoded[:, :, start_idx:start_idx + target_frames]
    
    # If VAE compressed temporal dimension, interpolate
    return F.interpolate(
        decoded,
        size=(target_frames, decoded.shape[3], decoded.shape[4]),
        mode='trilinear',
        align_corners=False
    )

# Import pipeline class from main training script
from .cogvideox_video_inpainting_sft import CogVideoXInpaintingPipeline
