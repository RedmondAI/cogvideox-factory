#!/bin/bash

# Default values for the script
PRETRAINED_MODEL_NAME_OR_PATH="cogvideo-5b"
OUTPUT_DIR="video-inpainting-model"
DATA_ROOT="/var/lib/docker/dataset/"
TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4
MAX_NUM_FRAMES=100
NUM_TRAIN_EPOCHS=100
LEARNING_RATE=1e-5
LR_WARMUP_STEPS=100
CHECKPOINTING_STEPS=500
VALIDATION_STEPS=100
MIXED_PRECISION="fp16"
ENABLE_XFORMERS_MEMORY_EFFICIENT_ATTENTION=True
RANDOM_FLIP_H=0.5
RANDOM_FLIP_V=0.5

# Memory and processing parameters
WINDOW_SIZE=32
OVERLAP=8
CHUNK_SIZE=32
USE_8BIT_ADAM=True
USE_FLASH_ATTENTION=True
GRADIENT_CHECKPOINTING=True

# New parameters for enhanced features
TEMPORAL_WINDOW_SIZE=5
MASK_LOSS_WEIGHT=2.0
SCHEDULER_NUM_STEPS=1000
SCHEDULER_TYPE="scaled_linear"
VAE_PRECISION="fp16"

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
    --temporal_window_size)
      TEMPORAL_WINDOW_SIZE="$2"
      shift 2
      ;;
    --mask_loss_weight)
      MASK_LOSS_WEIGHT="$2"
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

# Run the training script
python training/cogvideox_video_inpainting_sft.py \
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
  --temporal_window_size=$TEMPORAL_WINDOW_SIZE \
  --mask_loss_weight=$MASK_LOSS_WEIGHT \
  --scheduler_num_steps=$SCHEDULER_NUM_STEPS \
  --scheduler_type="$SCHEDULER_TYPE" \
  --vae_precision="$VAE_PRECISION" \
  --allow_tf32 \
  --report_to="wandb" \
  --dataloader_num_workers=8