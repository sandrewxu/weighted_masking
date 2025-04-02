#!/bin/bash

# --- Configuration ---
# Activate your Python environment (replace 'llada_env' if yours has a different name)
# conda activate 477 

# Set dataset path or Hugging Face dataset name
DATA_PATH="allenai/llama-3-tulu-v2-sft-subset"

# Set the output directory for LoRA adapters
OUTPUT_DIR="./llada-lora-sft"

# --- Training Arguments ---
# Optimized for A100 80GB GPU
MODEL_NAME="GSAI-ML/LLaDA-8B-Instruct"
EPOCHS=3
BATCH_SIZE=8  # Increased for A100
GRAD_ACCUM_STEPS=4  # Effective batch size = 32
LEARNING_RATE=2e-4  # Standard for LoRA fine-tuning
MAX_SEQ_LEN=2048  # Increased for better context learning
WARMUP_STEPS=500  # Increased for better stability
LORA_R=16  # Increased for better adaptation
LORA_ALPHA=32  # Increased proportionally with r
MAX_SAMPLES=  # Removed to use full dataset

# --- Run the Training ---
echo "Starting LLaDA LoRA fine-tuning..."
echo "Dataset: $DATA_PATH"
echo "Output Dir: $OUTPUT_DIR"
echo "Effective batch size: $((BATCH_SIZE * GRAD_ACCUM_STEPS))"
echo "Training for $EPOCHS epochs"

python finetuning.py \
    --model_name="$MODEL_NAME" \
    --data_path="$DATA_PATH" \
    --output_dir="$OUTPUT_DIR" \
    --epochs=$EPOCHS \
    --batch_size=$BATCH_SIZE \
    --gradient_accumulation_steps=$GRAD_ACCUM_STEPS \
    --lr=$LEARNING_RATE \
    --max_seq_len=$MAX_SEQ_LEN \
    --warmup_steps=$WARMUP_STEPS \
    --lora_r=$LORA_R \
    --lora_alpha=$LORA_ALPHA

echo "Fine-tuning finished."

