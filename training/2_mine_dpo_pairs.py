import os
import sys
import json
import re
import gc
import concurrent.futures
import torch
from datasets import load_dataset
from huggingface_hub import login, HfApi, snapshot_download
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# Import shared utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.reward_engine import execute_code_safely, calculate_staircase_reward, print_thesis_status
from utils.data_processing import clean_imports_and_fix_logic

# --- CONFIGURATION ---
# Note: Add your HF Token here or via ENV vars
HF_TOKEN = "hf_..." # Placeholder
BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
BACKUP_FILENAME = "dpo_progress.json"
TARGET_PROMPTS = 2500   
BATCH_SIZE = 250        
N_CANDIDATES = 8
TEMPERATURE = 0.8 # High temp for diversity mining
ADAPTER_PATH = "/path/to/sft_output" # Update this path manually

def process_single(args):
    """Worker function for concurrent execution."""
    code_gen, entry, code_body, prompt_text = args
    # Extract code block
    pattern = r"```(?:python)?\s*(.*?)(?:```|$)"
    matches = re.findall(pattern, code_gen, re.DOTALL | re.IGNORECASE)
    clean_code = matches[-1].strip() if matches else code_gen.replace("```python", "").strip()
    
    stats = execute_code_safely(code_body, clean_code, entry)
    score_val, msg = calculate_staircase_reward(stats, clean_code)
    
    return {
        "text": clean_code, "score": score_val, "stats": stats, "msg": msg
    }

def run_mining():
    # Login if needed
    if HF_TOKEN and not HF_TOKEN.startswith("hf_"):
        print("⚠️ Warning: No HF Token provided.")
    else:
        login(token=HF_TOKEN)

    # Load & Clean Data
    print("🧹 Loading and Cleaning Dataset...")
    full_ds = load_dataset("KodCode/KodCode-Light-RL-10K", split="train")
    full_ds = full_ds.filter(lambda x: x.get('solution') and x.get('test'))
    full_ds = full_ds.map(clean_imports_and_fix_logic)

    # Initialize vLLM
    print("🚀 Initializing vLLM...")
    llm = LLM(model=BASE_MODEL, enable_lora=True, max_lora_rank=16, gpu_memory_utilization=0.9, max_model_len=4096)
    sampling_params = SamplingParams(temperature=TEMPERATURE, top_p=0.9, max_tokens=1024, n=N_CANDIDATES)
    lora_req = LoRARequest("sft_adapter", 1, ADAPTER_PATH)

    dpo_master_list = []
    
    # Mining Loop
    for idx in range(0, TARGET_PROMPTS, BATCH_SIZE):
        batch_end = min(idx + BATCH_SIZE, TARGET_PROMPTS)
        print(f"\n🔄 Processing Batch: {idx} to {batch_end}...")
        
        current_batch = full_ds.select(range(idx, batch_end))
        prompts, metadata = [], []
        
        for item in current_batch:
            code = item['solution'].replace("```python", "").replace("```", "").strip()
            # Hardcoded SCoT prompt from thesis
            prompt_str = (
                f"<|im_start|>system\nYou are an expert QA Engineer. Write a Pytest suite for the following function.<|im_end|>\n"
                f"<|im_start|>user\nUse Structured Chain-of-Thought (SCoT) reasoning:\n"
                f"1. [Sequence]: Analyze the sequential flow of operations.\n"
                f"2. [Branch]: Identify all 'if/else' conditionals and edge cases.\n"
                f"3. [Loop]: Identify any loops and boundary conditions.\n"
                f"4. [Plan]: List the specific test cases needed for 100% coverage.\n"
                f"5. [Code]: Write the final Pytest code.\n\n"
                f"Function:\n```python\n{code}\n```<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            prompts.append(prompt_str)
            metadata.append({"entry_point": item.get('entry_point', 'solution'), "code_body": code})
        
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_req)
        
        # Parallel Execution of Generated Tests
        with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            for i, output in enumerate(outputs):
                meta = metadata[i]
                tasks = [(o.text, meta['entry_point'], meta['code_body'], output.prompt) for o in output.outputs]
                results = list(executor.map(process_single, tasks))
                
                # --- STRICT SELECTION LOGIC (Thesis Requirement) ---
                # 1. Filter for ELITE Candidates (Passed + 99% Cov + >50% Mutation)
                chosen_candidates = [
                    r for r in results 
                    if r['stats']['passed'] and r['stats']['cov'] >= 0.99 and r['stats']['mut'] >= 0.50
                ]
                
                if not chosen_candidates: continue

                # 2. Pick Best Chosen
                chosen_candidates.sort(key=lambda x: x['score'], reverse=True)
                best = chosen_candidates[0]
                
                # 3. Pick Worst Rejected (Ascending sort)
                results.sort(key=lambda x: x['score'], reverse=False) 
                worst = results[0]
                
                # 4. Check Margin (Thesis requires gap >= 0.40)
                if (best['score'] - worst['score']) < 0.40: continue

                # Save pair
                dpo_master_list.append({
                    "prompt": output.prompt,
                    "chosen": [{"content": output.prompt, "role": "user"}, {"content": best['text'], "role": "assistant"}],
                    "rejected": [{"content": output.prompt, "role": "user"}, {"content": worst['text'], "role": "assistant"}],
                    "metadata": {"score_chosen": best['score'], "score_rejected": worst['score']}
                })

        # Checkpoint save
        with open(BACKUP_FILENAME, "w") as f: 
            json.dump({"data": dpo_master_list}, f)
        print(f"   📈 TOTAL Pairs: {len(dpo_master_list)}")
        gc.collect()

if __name__ == "__main__":
    run_mining()