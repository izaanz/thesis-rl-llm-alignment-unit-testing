import os
import sys
import shutil
import json
import subprocess
import argparse

# --- STUDENT NOTE: This script runs the HumanEval+ benchmark using the EvalPlus library.
# It includes some specific hacks we needed for Kaggle/Colab GPUs (T4/P100) to work with vLLM.

def setup_environment():
    """Sets up the environment variables and fixes CUDA paths."""
    print("🛠️ Setting up environment for vLLM...")
    
    # Force Triton backend - found this was more stable on T4/RTX GPUs
    os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"
    
    # Fix for "missing libcuda.so" error common in Colab/Kaggle environments
    libcuda_path = "/usr/local/cuda/lib64/libcuda.so"
    stub_path = "/usr/local/cuda/lib64/stubs/libcuda.so"
    
    if not os.path.exists(libcuda_path) and os.path.exists(stub_path):
        print(f"   🩹 Applying libcuda symlink fix...")
        try:
            # Equivalent to !ln -s command
            os.symlink(stub_path, libcuda_path)
        except OSError as e:
            print(f"   ⚠️ Could not create symlink (might already exist): {e}")

def clean_cache():
    """Deletes previous results to ensure a fresh run."""
    if os.path.exists("evalplus_results"):
        print("🗑️ Deleting old cached results to prevent data mixing...")
        shutil.rmtree("evalplus_results")
    else:
        print("✨ No old cache found (Clean run).")

def apply_base_mode_hack(model_path):
    """
    Hack: Temporarily removes 'chat_template' from tokenizer_config.json.
    Why? If the model is too 'chatty' (says "Here is the code"), EvalPlus might fail to parse it.
    Forcing it to look like a raw code completion model often boosts the score.
    """
    config_path = os.path.join(model_path, "tokenizer_config.json")
    backup_path = config_path + ".bak"
    
    if not os.path.exists(config_path):
        print(f"⚠️ Warning: Could not find tokenizer config at {config_path}")
        return False

    print(f"🔧 Applying 'Base Mode' hack to: {config_path}")
    
    # 1. Backup the original file (So we don't break the model permanently!)
    shutil.copy(config_path, backup_path)
    
    # 2. Modify the JSON
    with open(config_path, 'r') as f:
        data = json.load(f)
    
    if "chat_template" in data:
        del data["chat_template"]
        with open(config_path, 'w') as f:
            json.dump(data, f, indent=4)
        print("   ✅ Chat template removed. Model will behave like a raw code completer.")
        return True
    else:
        print("   ℹ️ No chat template found (already in base mode?).")
        return False

def restore_base_mode_hack(model_path):
    """Restores the original tokenizer config from the backup."""
    config_path = os.path.join(model_path, "tokenizer_config.json")
    backup_path = config_path + ".bak"
    
    if os.path.exists(backup_path):
        print("↺ Restoring original tokenizer config...")
        shutil.move(backup_path, config_path)

def run_evaluation(model_path, tp_size=1, force_base=False):
    setup_environment()
    clean_cache()
    
    hack_applied = False
    if force_base:
        hack_applied = apply_base_mode_hack(model_path)
    
    print(f"\n🚀 Starting EvalPlus evaluation on: {model_path}")
    print(f"   ⚙️ Tensor Parallelism: {tp_size}")
    
    # Construct the command
    # We use subprocess instead of '!' so this runs as a python script
    cmd = [
        "evalplus.evaluate",
        "--model", model_path,
        "--dataset", "humaneval", # Standard HumanEval (EvalPlus runs on top of this)
        "--greedy",               # Deterministic results
        "--dtype", "half",        # Save VRAM
        "--tp", str(tp_size)      # Split across GPUs
    ]
    
    if force_base:
        cmd.append("--force-base-prompt")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Evaluation failed with error: {e}")
    finally:
        # Always cleanup the hack, even if the eval crashes!
        if hack_applied:
            restore_base_mode_hack(model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HumanEval+ Benchmarks")
    parser.add_argument("--model", type=str, required=True, help="Path or HuggingFace ID of the model")
    parser.add_argument("--tp", type=int, default=1, help="Tensor Parallelism (Number of GPUs)")
    parser.add_argument("--force-base", action="store_true", help="Strip chat template to force code-completion mode")
    
    args = parser.parse_args()
    
    run_evaluation(args.model, args.tp, args.force_base)