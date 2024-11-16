"""Argument parsing for CogVideoX training."""

import argparse

def get_args():
    """Get arguments for CogVideoX training."""
    parser = argparse.ArgumentParser(description="CogVideoX training arguments")
    
    # Model configuration
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=0)  # Add local_rank for distributed training
    
    # Data configuration
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--video_dir", type=str, default="RGB_720")
    parser.add_argument("--mask_dir", type=str, default="MASK_720")
    parser.add_argument("--gt_dir", type=str, default="GT_720")
    parser.add_argument("--image_size", type=int, default=720)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    
    # Training configuration
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_num_frames", type=int, default=49)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_warmup_steps", type=int, default=1000)
    parser.add_argument("--checkpointing_steps", type=int, default=2000)
    parser.add_argument("--validation_steps", type=int, default=500)
    
    # Optimization settings
    parser.add_argument("--mixed_precision", type=str, choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--use_flash_attention", action="store_true")
    parser.add_argument("--vae_precision", type=str, default="fp16")
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    
    # Processing settings
    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--overlap", type=int, default=8)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--random_flip_h", type=float, default=0.0)
    parser.add_argument("--random_flip_v", type=float, default=0.0)
    
    # Additional features
    parser.add_argument("--enable_model_cpu_offload", action="store_true")
    parser.add_argument("--enable_slicing", action="store_true")
    parser.add_argument("--enable_tiling", action="store_true")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    parser.add_argument("--deepspeed_config", type=str, default=None)
    
    args = parser.parse_args()
    
    # Set defaults for optional parameters
    if args.height is None:
        args.height = args.image_size
    if args.width is None:
        args.width = args.image_size
        
    return args
