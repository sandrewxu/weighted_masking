import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
# Use AutoModel if needed for LLaDA architecture
from transformers import AutoModel, AutoTokenizer, get_scheduler 
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import argparse
import os
import json
from datasets import load_dataset

MASK_TOKEN_ID = 126336  # [MASK] token for LLaDA


class SFTDataset(Dataset):
    def __init__(self, data_path_or_name, tokenizer, max_seq_length=256, split="train", max_samples=None): 
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        print(f"Using pad token ID: {self.pad_token_id}")

        # Define special tokens (Verify for the specific LLaDA model)
        self.start_id_token = "<start_id>" if "<start_id>" in tokenizer.vocab else "<s>"
        self.end_id_token = "<end_id>" if "<end_id>" in tokenizer.vocab else "</s>"
        self.eot_id_token = "<eot_id>" if "<eot_id>" in tokenizer.vocab else "</s>"
        self.bos_token = tokenizer.bos_token if tokenizer.bos_token is not None else "<s>"
        self.eos_token = tokenizer.eos_token if tokenizer.eos_token is not None else "</s>"

        # Load data using the datasets library
        self.data = self.load_data(data_path_or_name, split, max_samples)

    def load_data(self, data_path_or_name, split, max_samples=None):
        print(f"Loading dataset '{data_path_or_name}' split '{split}'...")
        try:
            dataset = load_dataset(data_path_or_name, split=split)
            original_count = len(dataset)
            
            # Filter for examples with at least 2 messages
            dataset = dataset.filter(lambda example: 
                example.get('messages') and 
                len(example['messages']) >= 2
            )
            
            filtered_count = len(dataset)
            print(f"Loaded {original_count} examples, kept {filtered_count} examples after filtering.")

            # Limit dataset size if specified
            if max_samples is not None and max_samples > 0:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
                print(f"Limited dataset to {len(dataset)} examples for testing")

            return dataset
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        messages = item.get('messages', [])

        # Simply take the first two messages
        if len(messages) < 2:
            print(f"Warning: Skipping item {idx} due to insufficient messages")
            dummy_ids = [self.pad_token_id] * self.max_seq_length
            return {
                "input_ids": torch.tensor(dummy_ids, dtype=torch.long),
                "prompt_length": torch.tensor(0, dtype=torch.long),
            }

        user_message = messages[0]["content"]
        assistant_message = messages[1]["content"]

        # Apply LLaDA Template
        prompt_section = f"{self.bos_token}{self.start_id_token}user{self.end_id_token}\n{user_message}{self.eot_id_token}"
        response_section = f"{self.start_id_token}assistant{self.end_id_token}\n{assistant_message}{self.eos_token}"

        prompt_tokens = self.tokenizer.encode(prompt_section, add_special_tokens=False)
        response_tokens = self.tokenizer.encode(response_section, add_special_tokens=False)

        input_ids = prompt_tokens + response_tokens
        prompt_length = len(prompt_tokens)

        # Truncate
        if len(input_ids) > self.max_seq_length:
            input_ids = input_ids[:self.max_seq_length]
            prompt_length = min(prompt_length, self.max_seq_length)

        # Pad
        padding_length = self.max_seq_length - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + [self.pad_token_id] * padding_length

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "prompt_length": torch.tensor(prompt_length, dtype=torch.long),
        }


# --- Use the correct LLaDA forward_process ---
MASK_TOKEN_ID = 126336
def forward_process(input_ids, eps=1e-3, device="cuda"): # Use the dynamic version
    b, l = input_ids.shape
    t = torch.rand(b, device=device) # Noise level per sequence
    p_mask_prob = (1 - eps) * t + eps # Dynamic mask probability per sequence
    p_mask_prob = p_mask_prob[:, None].repeat(1, l) # Expand to sequence length
    masked_indices_bool = torch.rand((b, l), device=device) < p_mask_prob
    noisy_batch = torch.where(masked_indices_bool, MASK_TOKEN_ID, input_ids)
    return noisy_batch, masked_indices_bool, p_mask_prob

# Remove the old, incorrect compute_llada_sft_loss function if it's still there

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Tokenizer has no pad_token, setting it to eos_token ({tokenizer.eos_token})")

    # Use AutoModel for LLaDA if required
    model = AutoModel.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    
    # Tie the weights before device mapping
    model.tie_weights()
    
    # Now apply device mapping
    model = model.to(device)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj"]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"Loading dataset with max_samples={args.max_samples}")
    dataset = SFTDataset(
        args.data_path,
        tokenizer,
        max_seq_length=args.max_seq_len,
        max_samples=args.max_samples
    )

    if not dataset or len(dataset) == 0:
        print("Failed to load dataset or dataset is empty. Exiting.")
        return
    print(f"Dataset size: {len(dataset)} examples")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(dataloader) * args.epochs // args.gradient_accumulation_steps
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)

    model.train()
    step_count = 0

    for epoch in range(args.epochs):
        total_epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for i, batch in enumerate(pbar):
            if batch is None or not torch.is_tensor(batch['input_ids']):
                print(f"Warning: Skipping invalid batch at index {i}")
                continue

            input_ids = batch["input_ids"].to(device)
            prompt_lengths = batch["prompt_length"].to(device)

            if torch.any(prompt_lengths <= 0):
                print(f"Warning: Skipping batch {i} due to invalid prompt_length <= 0")
                continue

            noisy_batch, masked_indices_bool, p_mask_prob = forward_process(input_ids, device=device)

            token_positions = torch.arange(noisy_batch.shape[1], device=device).expand(noisy_batch.size(0), -1)
            is_prompt_mask = token_positions < prompt_lengths.unsqueeze(1)
            noisy_batch[is_prompt_mask] = input_ids[is_prompt_mask]
            masked_indices_bool[is_prompt_mask] = False

            if not torch.any(masked_indices_bool):
                if (i + 1) % args.gradient_accumulation_steps == 0:
                    pass
                continue

            with torch.autocast(device_type=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32):
                outputs = model(input_ids=noisy_batch)
                logits = outputs.logits

                target_ids = input_ids[masked_indices_bool]
                pred_logits = logits[masked_indices_bool]
                prob_values = p_mask_prob[masked_indices_bool]

                if pred_logits.shape[0] == 0:
                    loss = torch.tensor(0.0, device=device, requires_grad=True)
                else:
                    token_loss = F.cross_entropy(pred_logits, target_ids, reduction='none')
                    normalized_token_loss = token_loss / (prob_values + 1e-6)

                    answer_lengths = (input_ids.shape[1] - prompt_lengths).unsqueeze(1).expand(-1, input_ids.shape[1])
                    answer_length_at_masked = answer_lengths[masked_indices_bool].clamp(min=1)

                    final_token_loss = normalized_token_loss / answer_length_at_masked
                    loss = final_token_loss.sum() / input_ids.shape[0]

                    if torch.isnan(loss):
                        print(f"Warning: NaN loss encountered at step {step_count}. Skipping batch {i}.")
                        loss = torch.tensor(0.0, device=device, requires_grad=True)
                        optimizer.zero_grad()
                        continue

            loss_item = loss.item()
            loss = loss / args.gradient_accumulation_steps

            if not torch.isnan(loss):
                loss.backward()
            else:
                print(f"Skipping backward for batch {i} due to NaN loss.")

            if (i + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                step_count += 1
                pbar.set_postfix({"Loss": loss_item, "LR": lr_scheduler.get_last_lr()[0]})

            total_epoch_loss += loss_item

        avg_loss = total_epoch_loss / len(dataloader) if len(dataloader) > 0 else 0
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

    print(f"Saving LoRA adapter model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning for LLaDA")
    parser.add_argument("--model_name", type=str, default="GSAI-ML/LLaDA-8B-Instruct", help="Base LLaDA model")
    parser.add_argument("--data_path", type=str, default="allenai/llama-3-tulu-v2-sft-subset", help="Path to training data (local path or HF dataset name)")
    parser.add_argument("--output_dir", type=str, default="./llada-lora-sft", help="Directory to save LoRA adapters")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Accumulate gradients over N steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Number of warmup steps for scheduler")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to load")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
    