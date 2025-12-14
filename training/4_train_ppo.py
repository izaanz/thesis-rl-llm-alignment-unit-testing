import torch
import gc
import re
import uuid
import os
import sys
import numpy as np
from peft import PeftModel
from transformers import AutoTokenizer, BitsAndBytesConfig
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from datasets import load_dataset

# Import shared utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.reward_engine import execute_code_safely, calculate_staircase_reward, print_thesis_status

# Note: If you are reproducing this then please make sure you have enough VRAM since PPO is quite heavy on compute
# Took hours of tuning to fix Out of Memory error (OOM)

# --- CONFIGURATION ---
SFT_ADAPTER_PATH = "/path/to/sft_output" # Update manually
# High Learning Rate (1e-5) as in thesis notes
LEARNING_RATE = 1.0e-5
KL_COEF = 0.1
NUM_STEPS = 160 

def run_ppo():
    print(f"\n🚀 Phase 2: PPO FINAL SPRINT...")
    
    # 1. Load SFT Model as Base
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    base = Qwen2ForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct", quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
    base.config.use_cache = True
    
    peft_model = PeftModel.from_pretrained(base, SFT_ADAPTER_PATH, is_trainable=True)
    model_ppo = AutoModelForCausalLMWithValueHead(peft_model)
    model_ppo.is_peft_model = True

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    # 2. Config
    config = PPOConfig(
        learning_rate=LEARNING_RATE,
        batch_size=16, 
        mini_batch_size=1, 
        gradient_accumulation_steps=16, 
        init_kl_coef=KL_COEF,
        adap_kl_ctrl=True,
        target=6.0,
        ppo_epochs=2,
        seed=42
    )
    ppo_trainer = PPOTrainer(config=config, model=model_ppo, tokenizer=tokenizer)
    
    # 3. Data Loader
    DATA_FILE = "data/train-00000-of-00001.parquet"
    print(f"📂 Loading PPO prompts from {DATA_FILE}...")
    # We load strictly from local 
    ds = load_dataset("parquet", data_files=DATA_FILE, split="train")
    ds = ds.filter(lambda x: x.get('solution') and x.get('test'))

    def strict_fmt(x): 
        code = x['solution'].replace("```python", "").replace("```", "").strip()
        # Explicit SCoT formatting
        prompt = (f"<|im_start|>system\nYou are an expert QA Engineer. Write a Pytest suite for the following function.<|im_end|>\n"
                  f"<|im_start|>user\nUse Structured Chain-of-Thought (SCoT) reasoning:\n"
                  f"1. [Sequence]: Analyze flow.\n2. [Branch]: Identify conditionals.\n3. [Plan]: List tests.\n4. [Code]: Write Pytest.\n\n"
                  f"Function:\n```python\n{code}\n```<|im_end|>\n<|im_start|>assistant\n")
        return {"prompt": prompt, "code_body": code, "entry_point": x.get('entry_point', 'solution')}
    
    loader = torch.utils.data.DataLoader(ds.map(strict_fmt), batch_size=config.batch_size, shuffle=True, collate_fn=lambda x: x)
    
    # 4. Training Loop
    step = 0
    device = ppo_trainer.accelerator.device
    
    print(f"   Running on {device} with LR={config.learning_rate}")
    
    for batch in loader:
        step += 1
        if step > NUM_STEPS: break
        
        prompts = [x['prompt'] for x in batch]
        q_tensors = [tokenizer(q, return_tensors="pt")['input_ids'].squeeze().to(device) for q in prompts]
        
        # Generation (Temperature 0.7 for creativity/logic balance)
        generation_kwargs = {
            "min_length": -1, "top_k": 0.0, "top_p": 0.9, "do_sample": True,
            "pad_token_id": tokenizer.pad_token_id, "max_new_tokens": 1024, "temperature": 0.7,
        }
        r_tensors = ppo_trainer.generate(q_tensors, return_prompt=False, **generation_kwargs)
        responses = [tokenizer.decode(r.squeeze()) for r in r_tensors]
        
        rewards, log_data = [], []
        
        # Calculate Rewards
        for i, (resp, item) in enumerate(zip(responses, batch)):
            # Regex to extract code block from response
            pattern = r"```(?:python)?\s*(.*?)(?:```|$)"
            matches = re.findall(pattern, resp, re.DOTALL | re.IGNORECASE)
            code_generated = matches[-1].strip() if matches else resp.replace("```python", "").strip()
            
            stats = execute_code_safely(item['code_body'], code_generated, item['entry_point'])
            final_score, msg_str = calculate_staircase_reward(stats, code_generated)
            stats['msg'] = msg_str
            stats['score'] = final_score
            stats['id'] = str(uuid.uuid4())[:6]
            
            rewards.append(torch.tensor(final_score, dtype=torch.float32).to(device))
            log_data.append(stats)
            
        print_thesis_status(step, log_data, phase="PPO-SPRINT")
        ppo_trainer.step(q_tensors, r_tensors, rewards)
        
        torch.cuda.empty_cache()

    model_ppo.save_pretrained("/workspace/thesis_project/ppo/final")
    print("✅ PPO Phase Complete.")

if __name__ == "__main__":
    run_ppo()