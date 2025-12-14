import re
import os
import sys
import uuid
import shutil
import subprocess
import numpy as np
from tabulate import tabulate

# --- Note: This regex defines the mutants we inject to see if the LLM catches them.
# If the LLM's test passes on these mutants, it gets a lower reward (it's a weak test).
REPLACEMENTS = [("==", "!="), ("!=", "=="), ("<", ">="), (">", "<="), (" and ", " or "), ("+", "-")]

def inject_mutants(code_str):
    """Injects synthetic bugs (operator swaps) to test if the unit test catches them."""
    mutants = []
    for old, new in REPLACEMENTS:
        if old in code_str:
            mutants.append(code_str.replace(old, new, 1))
            if len(mutants) >= 2: break
    return mutants

def execute_code_safely(code_body, test_body, entry_point_hint, temp_dir="./temp_exec"):
    """
    Executes code in isolation. 
    Returns: A dictionary of stats used for reward calculation.
    """
    run_id = str(uuid.uuid4())[:8]
    run_dir = f"{temp_dir}/{run_id}"
    os.makedirs(run_dir, exist_ok=True)
    
    func_file = f"{run_dir}/solution.py"
    test_file = f"{run_dir}/test_sol.py"
    cov_file = f"{run_dir}/.coverage"
    
    # Default Stats
    stats = {
        "passed": False, "cov": 0.0, "mut": -1.0, "ratio": 0.0,
        "p_count": 0, "f_count": 0, "msg": "Init", "loc": len(test_body.split('\n'))
    }
    
    try:
        # 1. Dynamic Function Name Extraction
        # Note: Using raw string for regex to avoid escape char issues
        match_name = re.search(r"def\s+(\w+)\s*\(", code_body)
        entry_point = match_name.group(1) if match_name else entry_point_hint

        # 2. Write Files
        with open(func_file, "w") as f: f.write(code_body)
        # Adding sys.path.append to make sure it finds the solution file
        test_content = f"import pytest\nimport sys\nsys.path.append('.')\nfrom solution import {entry_point}\n\n{test_body}"
        with open(test_file, "w") as f: f.write(test_content)

        env = os.environ.copy()
        env["COVERAGE_FILE"] = cov_file
        
        # 3. Run Pytest (Capture Output)
        cmd = ["coverage", "run", f"--include={func_file}", "-m", "pytest", test_file, "-q", "--tb=no", "--disable-warnings"]
        proc = subprocess.run(cmd, cwd=run_dir, env=env, capture_output=True, text=True, timeout=5)
        
        # 4. Parse Results
        p_match = re.search(r"(\d+)\s+passed", proc.stdout)
        f_match = re.search(r"(\d+)\s+failed", proc.stdout)
        
        p_count = int(p_match.group(1)) if p_match else 0
        f_count = int(f_match.group(1)) if f_match else 0
        
        # Fallback for "All passed" scenarios without explicit text (happens sometimes with small tests)
        if proc.returncode == 0 and p_count == 0:
            p_count = proc.stdout.count('.') if '.' in proc.stdout else 1
            f_count = 0

        total = p_count + f_count
        pass_ratio = (p_count / total) if total > 0 else 0.0
        passed_fully = (proc.returncode == 0) or (f_count == 0 and total > 0)
        if passed_fully: pass_ratio = 1.0

        stats.update({"passed": passed_fully, "p_count": p_count, "f_count": f_count, "ratio": pass_ratio})

        # 5. Coverage Calculation
        if pass_ratio > 0:
            cov_proc = subprocess.run(["coverage", "report", f"--data-file={cov_file}", func_file], cwd=run_dir, capture_output=True, text=True)
            # Regex to find the coverage percentage for solution.py
            match = re.search(r"solution\.py\s+\d+\s+\d+\s+(\d+)%", cov_proc.stdout)
            stats["cov"] = int(match.group(1))/100.0 if match else 0.0
        
        # 6. Mutation Testing (Only if tests pass and coverage > 50%)
        # This is the "Adversarial" part of the thesis
        if passed_fully and stats["cov"] > 0.5:
            stats["mut"] = 0.0
            mutants = inject_mutants(code_body)
            killed = 0
            if mutants:
                for mut_code in mutants:
                    with open(func_file, "w") as f: f.write(mut_code)
                    # Test should FAIL on buggy code (exit code != 0 means killed)
                    m_proc = subprocess.run([sys.executable, "-m", "pytest", test_file, "-q"], cwd=run_dir, capture_output=True, timeout=3)
                    if m_proc.returncode != 0: killed += 1
                stats["mut"] = killed / len(mutants)

        msg = "Success" if passed_fully else f"Partial"
        if proc.returncode != 0 and pass_ratio == 0: msg = "Syntax/Err"
        stats["msg"] = msg
        
        return stats

    except Exception as e: 
        stats["msg"] = "Sys Error"
        return stats
    finally: 
        # Clean up the temp folder so we don't crash memory (this was a nuisance to fix -_-)
        shutil.rmtree(run_dir, ignore_errors=True)

def check_structure(code_str):
    """Checks for ANY verification logic (Assert, Raises, Unittest)."""
    # Using raw strings for regex to ensure python treats backslashes correctly
    has_assert = bool(re.search(r"\bassert\b", code_str))
    has_raises = bool(re.search(r"pytest\.(raises|warns|deprecated_call)", code_str))
    has_unittest = bool(re.search(r"self\.assert[a-zA-Z]+", code_str))
    has_raise = bool(re.search(r"\braise\s+[a-zA-Z]+Error", code_str))
    return (has_assert or has_raises or has_unittest or has_raise)

def calculate_staircase_reward(stats, code_body):
    """
    The heart of the thesis methodology. 
    Scales reward from 0.0 to 1.0 based on:
    Syntax -> Assertions -> Pass Rate -> Coverage -> Mutation Score.
    """
    has_verification_logic = check_structure(code_body)
    
    # LEVEL 1: BROKEN
    if not stats['passed'] and stats['p_count'] == 0:
        if stats['msg'] == "Syntax/Err": return 0.0, "❌ Syntax"
        return 0.05, "❌ Crash" 

    # LEVEL 2: ZOMBIE (Runs but tests nothing)
    if not has_verification_logic: return 0.05, "💀 NO-LOGIC"

    # LEVEL 3: TRYING (Fail but has logic)
    if not stats['passed']:
        partial_score = 0.15 + (stats['ratio'] * 0.3)
        return partial_score, f"⚠️ {stats['ratio']*100:.0f}% Pass"

    # LEVEL 4: PASSING
    base_score = 0.5
    cov_bonus = stats['cov'] * 0.15
    mut_bonus = stats['mut'] * 0.35 # Heavy weighting on mutation for thesis goal
    
    total = base_score + cov_bonus + mut_bonus
    msg = "✅ PASS"
    if stats['mut'] > 0.8: msg = "🏆 ELITE"
    elif stats['mut'] > 0.5: msg = "⭐ STRONG"
    elif stats['mut'] == 0.0: msg = "👻 WEAK"
    
    return min(total, 1.0), msg

def print_thesis_status(step, log_data, phase="PPO"):
    """Prints a nice table for the logs."""
    print(f"\n{'='*20} 🎓 {phase} STEP {step} REPORT {'='*20}")
    headers = ["ID", "Status", "Tests", "Cov %", "Mut %", "LoC", "Score", "Msg"]
    table = []
    for item in log_data:
        status = "✅" if item['passed'] else "❌"
        if item['score'] > 0.85: status = "🏆"
        elif item['score'] < 0.1: status = "💀"
        
        test_disp = f"{item['p_count']}✅ {item['f_count']}❌"
        mut_str = f"{item['mut']*100:.0f}%" if item['mut'] >= 0 else "-"
        
        table.append([
            item['id'], status, test_disp, f"{item['cov']*100:.0f}%", 
            mut_str, item['loc'], f"⭐ {item['score']:.2f}", item['msg'][:15]
        ])
    print(tabulate(table, headers=headers, tablefmt="fancy_grid"))
    print(f"📊 Batch Avg Score: {np.mean([x['score'] for x in log_data]):.4f}")
    print(f"{'='*60}\n")