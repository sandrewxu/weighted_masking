#!/bin/bash
#SBATCH --job-name=llada_tulu_ft     # Job name
#SBATCH --output=logs/llada_tulu_ft_%j.out # Standard output log (%j expands to job ID)
#SBATCH --error=logs/llada_tulu_ft_%j.err  # Standard error log (%j expands to job ID)
#SBATCH --nodes=1                    # Request 1 node
#SBATCH --ntasks-per-node=1          # Run 1 task per node (the python script)
#SBATCH --gres=gpu:a100:1            # Request 1 A100 GPU
#SBATCH --cpus-per-task=8            # Request 8 CPU cores for data loading
#SBATCH --mem=256GB                   # Increased memory for larger sequence length
#SBATCH --time=24:00:00              # Increased time for 3 epochs on full dataset
#SBATCH --partition=gpu              # Specify the partition (check your cluster's name)

# --- Setup ---
# Create log and output directories if they don't exist
mkdir -p logs
mkdir -p ./llada-lora-sft-tulu-3epoch # Matches OUTPUT_DIR below

# Load necessary modules (adjust based on your HPC environment)
echo "Loading modules..."
module purge
module load StdEnv  # Load standard environment first
module load CUDA   # Load default CUDA version
module load miniconda  # Load miniconda

# Activate your Python environment (replace '477' if yours has a different name)
echo "Activating conda environment..."
source ~/.bashrc
source activate 477 
# If activation fails, ensure conda is initialized in your ~/.bashrc or run manually before sbatch

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1

# --- Job Information ---
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Starting time: $(date)"
echo "Directory: $(pwd)"
echo "Using GPU(s): $CUDA_VISIBLE_DEVICES" # Should be set by SLURM

# Print GPU information
nvidia-smi

# --- Set Arguments (optimized for A100 80GB) ---
DATA_PATH="allenai/llama-3-tulu-v2-sft-subset"
OUTPUT_DIR="./llada-lora-sft-tulu-3epoch"
MODEL_NAME="GSAI-ML/LLaDA-8B-Instruct"
EPOCHS=3
BATCH_SIZE=4
GRAD_ACCUM_STEPS=8
LEARNING_RATE=2e-4
MAX_SEQ_LEN=1024
WARMUP_STEPS=500
LORA_R=16
LORA_ALPHA=32

# --- Run the Training Script ---
echo "Starting LLaDA LoRA fine-tuning on Tulu dataset..."
echo "Dataset: $DATA_PATH"
echo "Output Dir: $OUTPUT_DIR"
echo "Effective batch size: $((BATCH_SIZE * GRAD_ACCUM_STEPS))"
echo "Training for $EPOCHS epochs"
echo "Max sequence length: $MAX_SEQ_LEN"

# Execute the python script with arguments
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

# --- Completion ---
echo "Fine-tuning finished at: $(date)" 
