#!/bin/bash

# Model configuration
PRETRAINED_MODEL_NAME_OR_PATH="THUDM/CogVideoX-5b"
OUTPUT_DIR="./cogvideox-inpainting"
DATA_ROOT="./data/inpainting"

# Training hyperparameters
TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4
MAX_NUM_FRAMES=64
NUM_TRAIN_EPOCHS=100
LEARNING_RATE=1e-5
LR_WARMUP_STEPS=1000
CHECKPOINTING_STEPS=2000
VALIDATION_STEPS=500

# Model settings
MIXED_PRECISION="bf16"
ENABLE_XFORMERS_MEMORY_EFFICIENT_ATTENTION=true
RANDOM_FLIP_H=0.5
RANDOM_FLIP_V=0.5
WINDOW_SIZE=32
OVERLAP=8
CHUNK_SIZE=4

# Memory optimizations
USE_8BIT_ADAM=true
USE_FLASH_ATTENTION=true
GRADIENT_CHECKPOINTING=true
VAE_PRECISION="fp32"
USE_CPU_OFFLOAD=true
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
  --use_cpu_offload=$USE_CPU_OFFLOAD \
  --enable_slicing=$ENABLE_SLICING \
  --enable_tiling=$ENABLE_TILING \
  --allow_tf32 \
  --report_to="wandb" \
  --dataloader_num_workers=$NUM_WORKERS \
  --deepspeed_config=configs/zero3.json