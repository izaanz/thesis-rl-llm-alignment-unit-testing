# 📊 Benchmark Suite for LLM Unit Test Generation

This folder contains the evaluation scripts used to benchmark the models (SFT, PPO, DPO) against industry standards. The suite covers **Functional Correctness**, **Code Coverage**, and **Mutation Analysis**.

---

## 🛠️ Setup & Requirements

Ensure you are in the root directory of the project before running these scripts.

```bash
# Install benchmark-specific dependencies
pip install evalplus cosmic-ray pytest-cov
```

> Note: Some scripts (like `fix_testeval.py`) rely on the shared utilities in `data_utils.py` located in this folder.

---

## 1️⃣ HumanEval+ (Functional Correctness)

This benchmark evaluates the model's ability to write functionally correct Python code. We use EvalPlus to run an extended set of test cases.

**Script:** `eval_humaneval.py`  

**Usage:**
```bash
# Run on your DPO model (with Tensor Parallelism = 2 GPUs)
python benchmarks/eval_humaneval.py \
    --model "path/to/your/merged_model" \
    --tp 2 \
    --force-base
```

**Arguments:**

| Argument     | Description |
|-------------|-------------|
| `--model`   | Path to the merged model or Hugging Face ID. |
| `--tp`      | Number of GPUs to split the model across (Tensor Parallelism). |
| `--force-base` | Critical: Strips the chat template to force "Code Completion" mode. Often boosts scores. |

---

## 2️⃣ TestEval (Statement & Branch Coverage)

This benchmark measures the quality of the generated tests by calculating how much of the canonical solution is executed (Coverage).

### Step A: Generate & Fix Tests
First, generate the raw predictions using your model (using vLLM or unsloth). Then, run the cleaner script to remove "Import Hallucinations" (e.g., `import solution`) that crash the evaluator.

```bash
# Fix the raw .jsonl output from your model
python benchmarks/fix_testeval.py --file "predictions/raw_model_output.jsonl"
```

**Output:** `predictions/raw_model_output_FIXED.jsonl`

### Step B: Run Parallel Evaluation
Run the fixed file through the coverage engine.

```bash
python benchmarks/run_testeval.py \
    --file "predictions/raw_model_output_FIXED.jsonl" \
    --workers 8
```

**Metrics Reported:**
- **Executable Correctness:** % of test suites that run without crashing.  
- **Line Coverage:** % of code lines executed.  
- **Branch Coverage:** % of control flow branches (if/else) executed.  

---

## 3️⃣ UnLeaked TestBench (ULT) - Mutation Analysis

This is the most rigorous benchmark. It tests **Generalization** (on unseen code) and **Robustness** (using Mutation Testing).

### Step A: Generate Tests
Use the model to write tests for the ULT dataset.

```bash
python benchmarks/ult/gen_tests.py \
    --model "path/to/merged_model" \
    --data "data/ULT.jsonl" \
    --output "predictions/ult_dpo.jsonl"
```

### Step B: Run Mutation Testing (Cosmic Ray)
This script injects synthetic bugs (mutants) into the code to see if your tests catch them.

```bash
python benchmarks/ult/run_mutation.py \
    --predictions "predictions/ult_dpo.jsonl" \
    --output "predictions/ult_dpo_results.jsonl"
```

**Interpretation:**
- **Survival Rate:** Lower is better (means you killed the bugs).  
- **Mutation Score:** Higher is better (100% means you caught every bug).  

---

## 🧩 Helper Utilities

- **`merge_adapter.py`**: Merges your LoRA adapter (from PPO/DPO training) into the base model so it can be loaded by vLLM for benchmarking.  

```bash
python benchmarks/merge_adapter.py --base "Qwen/..." --adapter "sft_output" --output "merged_model"
```

- **`data_utils.py`**: Shared functions for reading/writing `.jsonl` files.
