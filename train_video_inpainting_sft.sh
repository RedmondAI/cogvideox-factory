#!/bin/bash

# Default values for the script
PRETRAINED_MODEL_NAME_OR_PATH="THUDM/CogVideoX-5b"  # Using 5b model for better quality outputs
OUTPUT_DIR="video-inpainting-model"
DATA_ROOT="/var/lib/docker/dataset/"

# Optimized batch parameters for 8x A100 80GB
TRAIN_BATCH_SIZE=4  # Per-GPU batch size
GRADIENT_ACCUMULATION_STEPS=1  # With 8 GPUs, effective batch size = 32
MAX_NUM_FRAMES=100
NUM_TRAIN_EPOCHS=100
LEARNING_RATE=1e-5
LR_WARMUP_STEPS=100
CHECKPOINTING_STEPS=500
VALIDATION_STEPS=100
MIXED_PRECISION="bf16"  # Using bfloat16 since CogVideoX-5b weights are stored in bf16

# Memory and processing parameters optimized for 80GB A100s with NVLink
WINDOW_SIZE=64       # Larger window size for better temporal consistency
OVERLAP=16          # 25% overlap for smooth transitions
CHUNK_SIZE=64       # Process more frames at once
USE_8BIT_ADAM=true  # Memory efficient optimizer
USE_FLASH_ATTENTION=true
GRADIENT_CHECKPOINTING=true
VAE_PRECISION="bf16"

# Data augmentation parameters
RANDOM_FLIP_H=0.5  # Horizontal flip augmentation
RANDOM_FLIP_V=0.5  # Vertical flip augmentation

# Performance optimization
ENABLE_XFORMERS_MEMORY_EFFICIENT_ATTENTION=true
NUM_WORKERS=8  # Dataloader workers per GPU

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --pretrained_model_name_or_path)
      PRETRAINED_MODEL_NAME_OR_PATH="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --data_root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --train_batch_size)
      TRAIN_BATCH_SIZE="$2"
      shift 2
      ;;
    --gradient_accumulation_steps)
      GRADIENT_ACCUMULATION_STEPS="$2"
      shift 2
      ;;
    --learning_rate)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --window_size)
      WINDOW_SIZE="$2"
      shift 2
      ;;
    --overlap)
      OVERLAP="$2"
      shift 2
      ;;
    --chunk_size)
      CHUNK_SIZE="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the training script with DeepSpeed ZeRO-3 optimization
deepspeed --num_gpus=8 \
  training/cogvideox_video_inpainting_sft.py \
  --pretrained_model_name_or_path="$PRETRAINED_MODEL_NAME_OR_PATH" \
  --output_dir="$OUTPUT_DIR" \
  --data_root="$DATA_ROOT" \
  --train_batch_size=$TRAIN_BATCH_SIZE \
  --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
  --max_num_frames=$MAX_NUM_FRAMES \
  --num_train_epochs=$NUM_TRAIN_EPOCHS \
  --learning_rate=$LEARNING_RATE \
  --lr_warmup_steps=$LR_WARMUP_STEPS \
  --checkpointing_steps=$CHECKPOINTING_STEPS \
  --validation_steps=$VALIDATION_STEPS \
  --mixed_precision="$MIXED_PRECISION" \
  --enable_xformers_memory_efficient_attention=$ENABLE_XFORMERS_MEMORY_EFFICIENT_ATTENTION \
  --random_flip_h=$RANDOM_FLIP_H \
  --random_flip_v=$RANDOM_FLIP_V \
  --window_size=$WINDOW_SIZE \
  --overlap=$OVERLAP \
  --chunk_size=$CHUNK_SIZE \
  --use_8bit_adam=$USE_8BIT_ADAM \
  --use_flash_attention=$USE_FLASH_ATTENTION \
  --gradient_checkpointing=$GRADIENT_CHECKPOINTING \
  --vae_precision="$VAE_PRECISION" \
  --allow_tf32 \
  --report_to="wandb" \
  --dataloader_num_workers=$NUM_WORKERS \
  --deepspeed_config=configs/zero3.json