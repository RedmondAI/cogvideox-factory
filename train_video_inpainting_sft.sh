#!/bin/bash

# Model configuration
PRETRAINED_MODEL_NAME_OR_PATH="THUDM/CogVideoX-5b"
OUTPUT_DIR="./cogvideox-inpainting"
DATA_ROOT="/var/lib/docker/dataset"  # Updated to correct dataset path

# Training hyperparameters
TRAIN_BATCH_SIZE=4  # Increased to match deepspeed config
GRADIENT_ACCUMULATION_STEPS=8  # Adjusted for total batch size of 32
MAX_NUM_FRAMES=49  # Set to match model's native frame count
NUM_TRAIN_EPOCHS=100
LEARNING_RATE=1e-5
LR_WARMUP_STEPS=1000
CHECKPOINTING_STEPS=2000
VALIDATION_STEPS=500

# Dataset settings
VIDEO_DIR="RGB_720"
MASK_DIR="MASK_720"
GT_DIR="GT_720"
IMAGE_SIZE=720

# Model settings
MIXED_PRECISION="bf16"  # Using bfloat16 for better numerical stability
ENABLE_XFORMERS_MEMORY_EFFICIENT_ATTENTION=true
RANDOM_FLIP_H=0.5
RANDOM_FLIP_V=0.5
WINDOW_SIZE=32
OVERLAP=8
CHUNK_SIZE=64  # Increased for more efficient processing

# Memory optimizations
USE_8BIT_ADAM=true
USE_FLASH_ATTENTION=true
GRADIENT_CHECKPOINTING="--gradient_checkpointing"  # Fixed flag
VAE_PRECISION="bf16"  # Changed to match mixed precision setting
ENABLE_MODEL_CPU_OFFLOAD="--enable_model_cpu_offload"  # Fixed flag
ENABLE_SLICING=true
ENABLE_TILING=true

# Performance settings
NUM_WORKERS=8

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

# Run training
deepspeed --include localhost:0,1,2,3,4,5,6,7 \
  training/cogvideox_video_inpainting_sft.py \
  --pretrained_model_name_or_path=$PRETRAINED_MODEL_NAME_OR_PATH \
  --output_dir=$OUTPUT_DIR \
  --data_root=$DATA_ROOT \
  --video_dir=$VIDEO_DIR \
  --mask_dir=$MASK_DIR \
  --gt_dir=$GT_DIR \
  --image_size=$IMAGE_SIZE \
  --train_batch_size=$TRAIN_BATCH_SIZE \
  --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
  --max_num_frames=$MAX_NUM_FRAMES \
  --num_train_epochs=$NUM_TRAIN_EPOCHS \
  --learning_rate=$LEARNING_RATE \
  --lr_warmup_steps=$LR_WARMUP_STEPS \
  --checkpointing_steps=$CHECKPOINTING_STEPS \
  --validation_steps=$VALIDATION_STEPS \
  --mixed_precision=$MIXED_PRECISION \
  --enable_xformers_memory_efficient_attention=$ENABLE_XFORMERS_MEMORY_EFFICIENT_ATTENTION \
  --random_flip_h=$RANDOM_FLIP_H \
  --random_flip_v=$RANDOM_FLIP_V \
  --window_size=$WINDOW_SIZE \
  --overlap=$OVERLAP \
  --chunk_size=$CHUNK_SIZE \
  --use_8bit_adam=$USE_8BIT_ADAM \
  --use_flash_attention=$USE_FLASH_ATTENTION \
  $GRADIENT_CHECKPOINTING \
  --vae_precision=$VAE_PRECISION \
  $ENABLE_MODEL_CPU_OFFLOAD \
  --enable_slicing=$ENABLE_SLICING \
  --enable_tiling=$ENABLE_TILING \
  --allow_tf32 \
  --report_to=wandb \
  --dataloader_num_workers=$NUM_WORKERS \
  --deepspeed_config=configs/zero3.json