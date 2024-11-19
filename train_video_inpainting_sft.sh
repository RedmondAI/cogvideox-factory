#!/bin/bash

# Configure DeepSpeed launch
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Model configuration
PRETRAINED_MODEL_NAME_OR_PATH="THUDM/CogVideoX-5b"
OUTPUT_DIR="./cogvideox-inpainting"
DATA_ROOT="/var/lib/docker/dataset"

# Training hyperparameters
TRAIN_BATCH_SIZE=1  # Reduced batch size for memory efficiency
EVAL_BATCH_SIZE=1   # Same as train batch size for consistency
GRADIENT_ACCUMULATION_STEPS=32  # Increased to maintain effective batch size
MAX_NUM_FRAMES=32  # Reduced from 49 for memory efficiency
NUM_TRAIN_EPOCHS=100
LEARNING_RATE=1e-5
LR_WARMUP_STEPS=1000
CHECKPOINTING_STEPS=2000
VALIDATION_STEPS=500

# Dataset settings
VIDEO_DIR="RGB_480"
MASK_DIR="MASK_480"
GT_DIR="GT_480"
IMAGE_SIZE=480

# Model settings
MIXED_PRECISION="bf16"  # Using bfloat16 for better numerical stability
RANDOM_FLIP_H=0.5
RANDOM_FLIP_V=0.5
WINDOW_SIZE=16  # Reduced from 32
OVERLAP=4  # Reduced from 8
CHUNK_SIZE=32  # Reduced from 64
VAE_PRECISION="bf16"
NUM_WORKERS=2  # Reduced for better memory management

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
    --eval_batch_size)
      EVAL_BATCH_SIZE="$2"
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

# Run training with memory optimizations
deepspeed training/cogvideox_video_inpainting_sft.py \
    --pretrained_model_name_or_path $PRETRAINED_MODEL_NAME_OR_PATH \
    --output_dir $OUTPUT_DIR \
    --data_root $DATA_ROOT \
    --video_dir $VIDEO_DIR \
    --mask_dir $MASK_DIR \
    --gt_dir $GT_DIR \
    --image_size $IMAGE_SIZE \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --max_num_frames $MAX_NUM_FRAMES \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --lr_warmup_steps $LR_WARMUP_STEPS \
    --checkpointing_steps $CHECKPOINTING_STEPS \
    --validation_steps $VALIDATION_STEPS \
    --mixed_precision $MIXED_PRECISION \
    --vae_precision $VAE_PRECISION \
    --window_size $WINDOW_SIZE \
    --overlap $OVERLAP \
    --chunk_size $CHUNK_SIZE \
    --random_flip_h $RANDOM_FLIP_H \
    --random_flip_v $RANDOM_FLIP_V \
    --num_workers $NUM_WORKERS \
    --enable_xformers_memory_efficient_attention \
    --gradient_checkpointing \
    --deepspeed_config configs/deepspeed/ds_config_zero3.json \
    --ignore_text_encoder \
    --logging_dir logs \
    --report_to wandb \
    "$@"