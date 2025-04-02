#!/bin/bash
#SBATCH --job-name=llada_msmarco_ft  # Job name
#SBATCH --output=logs/llada_msmarco_ft_%j.out # Standard output log (%j expands to job ID)
#SBATCH --error=logs/llada_msmarco_ft_%j.err  # Standard error log (%j expands to job ID)
#SBATCH --nodes=1                    # Request 1 node
#SBATCH --ntasks-per-node=1          # Run 1 task per node (the python script)
#SBATCH --gres=gpu:a100:1            # Request 8 GPUs per node (adjust if different)
#SBATCH --cpus-per-task=1           # Request 12 CPU cores per task (adjust based on dataloading needs)
#SBATCH --mem=128GB                   # Memory per node (adjust based on needs/limits)
#SBATCH --time=12:00:00              # Maximum job run time (HH:MM:SS)
#SBATCH --partition=gpu              # Specify the partition (check your cluster's name)

# --- Setup ---
# Create log and output directories if they don't exist
mkdir -p logs
mkdir -p ./llada-lora-sft # Matches OUTPUT_DIR below

# Load necessary modules (adjust based on your HPC environment)
echo "Loading modules..."
module purge
module load cuda/12.1 # Or your cluster's CUDA version
module load miniconda # Or python module

# Activate your Python environment (replace '477' if yours has a different name)
echo "Activating conda environment..."
source activate 477 
# If activation fails, ensure conda is initialized in your ~/.bashrc or run manually before sbatch

# --- Job Information ---
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Starting time: $(date)"
echo "Directory: $(pwd)"
echo "Using GPU(s): $CUDA_VISIBLE_DEVICES" # Should be set by SLURM

# Print GPU information
nvidia-smi

# --- Set Arguments (copied from LLaDA_finetune.sh) ---
DATA_PATH="microsoft/ms_marco"
DATA_SUBSET="v2.1"
OUTPUT_DIR="./llada-lora-sft"
MODEL_NAME="GSAI-ML/LLaDA-8B-Instruct"
EPOCHS=3
BATCH_SIZE=8
GRAD_ACCUM_STEPS=4
LEARNING_RATE=1e-4
MAX_SEQ_LEN=1024
WARMUP_STEPS=100
LORA_R=8
LORA_ALPHA=16

# --- Run the Training Script ---
echo "Starting LLaDA LoRA fine-tuning on MS MARCO..."
echo "Dataset: $DATA_PATH ($DATA_SUBSET)"
echo "Output Dir: $OUTPUT_DIR"

# Execute the python script with arguments
# Note: device_map='auto' in finetuning.py might automatically use available GPUs allocated by SLURM.
# If you need explicit multi-GPU handling (e.g., DDP), the python script would need modifications.
python finetuning.py \
    --model_name="$MODEL_NAME" \
    --data_path="$DATA_PATH" \
    --data_subset="$DATA_SUBSET" \
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
