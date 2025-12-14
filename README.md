# thesis-rl-llm-alignment-unit-testing
# A Comparative Analysis of RL (PPO) and DPO for Automated Unit Test Generation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Bachelor Thesis** | **B.Sc. Applied Artificial Intelligence** > **Author:** Izaan Zubair  
> **Supervisor:** Dr. Nghia Duong-Trung  
> **Institution:** IU University of Applied Sciences  

## 📄 Abstract
This research investigates whether reinforcement learning alignment techniques can overcome the limitations of **Supervised Fine-Tuning (SFT)** for automated unit test generation. Current LLMs often produce "shallow tests"—syntactically valid but functionally weak code that fails to detect edge-case defects (the "Happy Path" bias).

This project implements a three-phase methodology using **Qwen2.5-Coder-7B-Instruct**:
1.  **SFT:** Baseline adaptation using **Structured Chain-of-Thought (SCoT)** prompting.
2.  **PPO:** Online Reinforcement Learning with a novel **Robust Reward Engine** that evaluates executability, coverage, and mutation scores.
3.  **DPO:** Offline Direct Preference Optimization using a mined dataset of "Elite" vs. "Weak" test pairs.

**Findings:** The DPO-aligned model achieved **97.95% execution correctness** and **97.65% branch coverage**, significantly outperforming both SFT (70.79%) and PPO (92.95%) while requiring 3x less training time.

---

## 🏗️ Methodology & Architecture

The training pipeline consists of three distinct phases designed to shift the model from *imitation* (SFT) to *adversarial testing* (RL).

![Training Pipeline](https://github.com/izaanz/thesis-rl-llm-alignment-unit-testing/blob/main/image/pipeline.png)

### The Robust Reward Engine
A custom reward function acts as the ground-truth oracle for the RL phases. It assigns a scalar reward ($R \in [0, 1]$) based on a "Staircase" hierarchy:

| Level | Status | Criteria | Reward |
| :--- | :--- | :--- | :--- |
| **1. Broken** | ❌ Crash | Code fails to compile or run. | `0.0 - 0.05` |
| **2. Zombie** | 💀 No-Logic | Code runs but contains no assertions. | `0.05` |
| **3. Trying** | ⚠️ Partial | Some tests pass, but others fail. | `0.15 - 0.45` |
| **4. Passing** | 🟢 Weak | 100% Pass, but low coverage/mutation score. | `~0.50` |
| **5. Elite** | 🏆 **Robust** | **100% Pass + High Coverage + Kills Mutants.** | `0.80 - 1.0` |

---
The training pipeline transforms the model from a passive code generator into an adversarial tester through three distinct phases.

### Phase 1: Supervised Fine-Tuning (SFT)
The baseline model is adapted using **Structured Chain-of-Thought (SCoT)** prompting. This forces the model to plan its testing strategy (Sequence -> Branch -> Loop -> Plan) before generating code, reducing the rate of "Zombie Tests" (code that runs but tests nothing).

### Phase 2: PPO with Staircase Reward
We introduce a **Staircase Reward Function** ($R \in [0, 1]$) to solve the sparse reward problem in code generation. The reward is calculated hierarchically:

* **Level 1 (Syntax):** 0.05 points for valid Python syntax (even if it crashes).
* **Level 2 (Logic):** +0.05 points if assertions (`assert`, `pytest.raises`) are present.
* **Level 3 (Correctness):** Up to +0.45 points based on the pass rate against the canonical solution.
* **Level 4 (Quality):** The final 0.50 points are unlocked *only* if the test suite achieves high **Statement Coverage** and kills injected **Mutants** (synthetic bugs).

### Phase 3: Direct Preference Optimization (DPO)
We mine the trajectory of the SFT model to create a static dataset of 1,190 preference pairs. A pair $(y_w, y_l)$ is selected only if:
* **Winner ($y_w$):** Is an "Elite" candidate (100% Pass + >50% Mutation Score).
* **Loser ($y_l$):** Is a "Weak" candidate (100% Pass but 0% Mutation Score).
* **Margin:** The reward gap is significant ($R_w - R_l > 0.4$).

## 📂 Repository Structure

```text
thesis-rl-llm-alignment-unit-testing/
├── benchmarks/           # Evaluation scripts (TestEval, HumanEval+, ULT)
│   ├── fix_testeval.py   # Cleans generated code for evaluation
│   ├── merge_adapter.py  # Helper to merge LoRA adapters
│   └── ...
├── data/                 # Local dataset storage (KodCode-Light-RL)
├── training/             # Training pipelines for SFT, PPO, and DPO
│   ├── 1_train_sft.py    # Phase 1: SFT Baseline
│   ├── 2_mine_dpo.py     # Phase 3a: Mining "Elite" pairs
│   ├── 3_train_dpo.py    # Phase 3b: DPO Training
│   └── 4_train_ppo.py    # Phase 2: PPO Training
├── utils/                # Core logic (Reward Engine, SCoT formatting)
├── requirements.txt      # Project dependencies
└── README.md             # This file
```
## 🚀 Installation & Usage
### 1. Setup Environment
```bash
# Clone the repository
git clone https://github.com/izaanz/thesis-rl-llm-alignment-unit-testing.git
cd thesis-rl-llm-alignment-unit-testing

# Install dependencies
pip install -r requirements.txt
```
### 2. Training Workflow
The training scripts are numbered to follow the thesis methodology.
#### Phase 1: SFT Baseline
```bash 
python training/1_train_sft.py
```

#### Phase 2: PPO Alignment (Optional comparison)
```bash
python training/4_train_ppo.py
```
####Phase 3: DPO Alignment (Recommended)

```bash
# Step 3a: Mine "Elite" preference pairs using the SFT model
python training/2_mine_dpo_pairs.py

# Step 3b: Train the DPO model on the mined dataset
python training/3_train_dpo.py
```
## 📊 Results & Benchmarks
### 1. Functional Correctness (TestEval)
DPO achieved near-perfect execution correctness, solving the Zombie Test problem prevalent in SFT.
| Model           | Syntax Correctness | Execution Correctness |
|-----------------|--------------------|-----------------------|
| Qwen2.5-Base    | **99.80%**             | 68.50%                |
| SFT (Baseline)  | 98.11%             | 70.79%                |
| PPO             | 98.50%             | 92.95%                |
| DPO             | 99.55%             | **97.95%**                |

### 2. Test Quality (Coverage)
While SFT plateaus at ~72% coverage (Happy Path bias), DPO effectively saturates branch coverage.
**Table 8. TestEval Average Line and Branch Coverage**

| Model                 | Avg Line Cov | Avg Branch Cov |
|-----------------------|--------------|----------------|
| Qwen2.5-Coder-7B-Ins  | 72.50%       | 70.00%        |
| SFT                   | 74.76%       | 72.14%        |
| PPO                   | 92.56%       | 91.65%        |
| DPO                   | 98.56%       | 97.65%        |

While Qwen2.5 and SFT plateau around ~70–75% coverage, reflecting a Happy Path bias, PPO shows a substantial improvement in both line and branch coverage, and DPO further saturates branch coverage.
### 3. Generalization (UnLeaked TestBench - ULT)
| Model                | Pass@1 (ULT) |
|----------------------|--------------|
| Qwen2.5-Coder-7B-Ins | 12.20%       |
| SFT                  | 12.90%       |
| PPO                  | 15.40%       |

On the adversarial ULT benchmark with unseen data, DPO outperformed Qwen2.5, SFT, and PPO, surpassing even larger open-source models.

## 🔍 Qualitative Analysis
The thesis highlights how alignment changes the nature of generated tests. Below is a comparison using a buggy Fibonacci function (returns 0 for negative inputs instead of raising ValueError).
| Model | Status | Behavior | Example Test |
|------|--------|----------|--------------|
| **SFT (Baseline)** | 🟢 Passing (Weak) | Writes “Happy Path” tests only. Misses the bug entirely. | `assert fib(5) == 5` |
| **PPO (Reward Hacking)** | 🔴 Flawed | Learns the bug and enforces incorrect behavior to maximize reward. | `# PPO expects 0!`<br>`assert fib(-1) == 0` |
| **DPO (Robust)** | 🏆 Elite | Detects the bug and enforces correct logic. | `with pytest.raises(ValueError):`<br>&nbsp;&nbsp;`fib(-1)` |

## 💻 Computational Efficiency
DPO proved to be the most efficient alignment strategy, avoiding the memory overhead of PPO.
| Method | Peak VRAM | Training Time | Sample Efficiency |
|--------|-----------|---------------|-------------------|
| SFT    | 24.5 GB   | 1.5 hrs       | 10k static samples |
| PPO    | 31.1 GB   | 12.0 hrs      | 2.5k generated rollouts |
| DPO    | 26.9 GB   | 4.2 hrs       | 1.1k mined pairs |

## Conclusion
DPO offers a scalable, efficient alternative to PPO for domain-specific alignment, achieving superior results with 35% less training time and significantly lower memory overhead.
