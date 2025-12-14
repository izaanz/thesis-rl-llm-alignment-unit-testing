import os
import sys
import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from unsloth import FastLanguageModel

# Ensure we can import from utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_processing import clean_imports_and_fix_logic, format_scot

# I trained on RTX 5090 - 32 GB VRAM - (So change the parameters to accomodate your choice of GPU)
# Config
MAX_SEQ_LENGTH = 4096
OUTPUT_DIR = "./sft_output" # Change this path manually if needed
DATASET_NAME = "KodCode/KodCode-Light-RL-10K"
SEED = 3407

def prepare_data():
    print(f"⬇️ Downloading {DATASET_NAME}...")
    ds = load_dataset(DATASET_NAME, split="train")
    
    print("🔍 Filtering invalid entries...")
    ds = ds.filter(lambda x: x.get('solution') and x.get('test'))
    
    print("🧹 Cleaning dataset...")
    ds = ds.map(clean_imports_and_fix_logic)

    print("📝 Applying SCoT formatting...")
    # Formatting map
    ds = ds.map(format_scot)
    return ds

def run_sft():
    train_dataset = prepare_data()
    print("\n🚀 Phase 1: SFT Training (Unsloth)...")
    
    # 1. Load Model (Unsloth optimized)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    # 2. Add LoRA Adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        bias="none",
        use_gradient_checkpointing="unsloth", 
        random_state=SEED,
    )

    # 3. CRITICAL: Prompt Masking
    # We split on "<|im_start|>assistant\n" so loss is only calculated on the output, not the prompt.
    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    # 4. Training Arguments
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,  # Effective Batch = 16
        max_steps=600,                  # ~1.18 Epochs
        learning_rate=2e-4,             # High LR for LoRA SFT
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=50,
        optim="adamw_8bit",             # Saves VRAM
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=SEED,
        report_to="none",
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        data_collator=collator,
        dataset_num_proc=2,
        packing=False, # Packing conflicts with masking collator
        args=args,
    )

    trainer_stats = trainer.train()
    
    print("💾 Saving model...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✅ SFT Complete & Saved.")

if __name__ == "__main__":
    run_sft()