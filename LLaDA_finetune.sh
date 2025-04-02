    #!/bin/bash

    # --- Configuration ---
    # Activate your Python environment (replace 'llada_env' if yours has a different name)
    # conda activate 477 

    # Set dataset path or Hugging Face dataset name
    # *** MUST CHANGE THIS ***
    DATA_PATH="microsoft/ms_marco" 
DATA_SUBSET="v2.1" # Specify the subset
    # Example for local path: DATA_PATH="/path/to/your/local_dataset.jsonl" 

    # Set the output directory for LoRA adapters
    OUTPUT_DIR="./llada-lora-sft"

    # --- Training Arguments ---
    # Adjust these as needed
    MODEL_NAME="GSAI-ML/LLaDA-8B-Instruct"
    EPOCHS=1
    BATCH_SIZE=2  # Per GPU batch size
    GRAD_ACCUM_STEPS=1 # Effective batch size = BATCH_SIZE * GRAD_ACCUM_STEPS
    LEARNING_RATE=1e-4 # Often lower for LoRA (e.g., 1e-4, 2e-4)
    MAX_SEQ_LEN=256 # Adjust based on GPU memory and data
    WARMUP_STEPS=100
    LORA_R=8
    LORA_ALPHA=16

    # --- Run the Training ---
    echo "Starting LLaDA LoRA fine-tuning..."
    echo "Dataset: $DATA_PATH"
    echo "Output Dir: $OUTPUT_DIR"

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

    echo "Fine-tuning finished."

