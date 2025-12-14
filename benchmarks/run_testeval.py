import os
import subprocess
import json
import signal
import random
import shutil
import multiprocessing
import argparse
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy
from data_utils import read_jsonl

# Student Note: Setting a seed ensures that if we run this twice, 
# the "random selection" of test cases for Pass@k remains consistent.
random.seed(42)

def process_single_item(args):
    """
    Worker function: Runs one generated test suite against the problem code.
    Runs in its own isolated temporary directory to prevent file conflicts.
    """
    i, data, ks = args
    
    task_num = data.get('task_num', i)
    difficulty = data.get('difficulty', 'unknown')
    code = data['code']       # The problem (Fibonacci, etc.)
    test_cases = data['tests'] # The generated test suite
    
    # Create unique temp dir for this process
    temp_dir = f'temp_eval_{i}_{difficulty}'
    
    item_stats = {
        'total_cases': 0, 'total_syn_correct': 0, 'total_exec_correct': 0,
        'total_line_cov': 0, 'total_branch_cov': 0, 'exec_fails': [],
        'line_covs_at_k': {f'cov@{k}': 0.0 for k in ks},
        'branch_covs_at_k': {f'cov@{k}': 0.0 for k in ks},
    }

    try:
        os.makedirs(temp_dir, exist_ok=True)
        
        # 1. Write the Function Under Test
        # We rename it to 'under_test.py' so we can import it standardly
        with open(f'{temp_dir}/under_test.py', 'w') as f:
            f.write(code)
            
        test_import_header = f'from under_test import Solution\n'
        
        passed_tests = []
        
        for j, testcase in enumerate(test_cases):
            item_stats['total_cases'] += 1
            
            # --- Check 1: Syntax ---
            try:
                compile(testcase, '<string>', 'exec')
                item_stats['total_syn_correct'] += 1
            except:
                continue # Skip if it doesn't even compile

            # --- Check 2: Execution ---
            test_filename = f'test_{j}.py'
            
            # Combine header + generated code
            # Note: The fix_testeval.py script should have already handled imports, 
            # but we prepend the specific import for 'under_test' here.
            # However, if the model used 'from solution import...', our cleaner removed it,
            # so now we need to make sure the function is available.
            # Ideally, we inject 'from under_test import *' to cover most bases.
            full_test_code = "import pytest\nfrom under_test import *\n" + testcase
            
            with open(f'{temp_dir}/{test_filename}', 'w') as f:
                f.write(full_test_code)
            
            cwd = os.getcwd()
            os.chdir(temp_dir)
            
            try:
                # Run Pytest (Quiet mode)
                # Return code 0 = All Pass, 1 = Tests Failed (but ran), 2/4 = Error
                cmd = ['pytest', test_filename, '-q', '--disable-warnings']
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
                
                if result.returncode in [0, 1]:
                    item_stats['total_exec_correct'] += 1
                    passed_tests.append(test_filename)
                else:
                    err = result.stderr.decode() or "Crash"
                    item_stats['exec_fails'].append({'task': task_num, 'err': err[:100]})
                    
            except subprocess.TimeoutExpired:
                 item_stats['exec_fails'].append({'task': task_num, 'err': "Timeout"})
            except Exception as e:
                 item_stats['exec_fails'].append({'task': task_num, 'err': str(e)})
            finally:
                os.chdir(cwd)

        # --- Check 3: Coverage (Only if executable) ---
        if passed_tests:
            os.chdir(temp_dir)
            try:
                # We calculate coverage on 'under_test.py'
                base_cov_cmd = ['pytest', '--cov=under_test', '--cov-branch', '--cov-report=json:coverage.json']
                
                # Run all passed tests at once to get Total Coverage
                subprocess.run(base_cov_cmd + passed_tests, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
                
                if os.path.exists('coverage.json'):
                    with open('coverage.json', 'r') as f:
                        cov = json.load(f)
                        
                    # Calculate Stats
                    t_stmts = cov['totals']['num_statements']
                    c_stmts = cov['totals']['covered_lines']
                    item_stats['total_line_cov'] = c_stmts / t_stmts if t_stmts > 0 else 0
                    
                    t_br = cov['totals']['num_branches']
                    c_br = cov['totals']['covered_branches']
                    item_stats['total_branch_cov'] = c_br / t_br if t_br > 0 else 0
                    
            except Exception:
                pass 
            finally:
                os.chdir(cwd)

    except Exception as e:
        print(f"Error on Task {task_num}: {e}")
    finally:
        # Cleanup temp folder (crucial for parallel processing!)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            
    return item_stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Fixed predictions file (.jsonl)")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    args = parser.parse_args()
    
    print(f"🚀 Starting TestEval on: {args.file}")
    data = read_jsonl(args.file)
    
    # Ks for Pass@K (1, 2, 5 are standard)
    KS = [1, 2, 5] 
    
    # Prepare arguments for map
    tasks = [(i, d, KS) for i, d in enumerate(data)]
    
    print(f"   Processing {len(tasks)} tasks with {args.workers} workers...")
    
    # Parallel Execution
    results = []
    with multiprocessing.Pool(args.workers) as pool:
        for res in tqdm(pool.imap_unordered(process_single_item, tasks), total=len(tasks)):
            results.append(res)
            
    # Aggregation
    print("\n📊 Aggregating Results...")
    total_cases = sum(r['total_cases'] for r in results)
    exec_correct = sum(r['total_exec_correct'] for r in results)
    
    avg_line_cov = sum(r['total_line_cov'] for r in results) / len(results) if results else 0
    avg_branch_cov = sum(r['total_branch_cov'] for r in results) / len(results) if results else 0

    print(f"{'='*40}")
    print(f"EXECUTABLE CORRECTNESS: {exec_correct / total_cases * 100:.2f}%")
    print(f"AVG LINE COVERAGE:      {avg_line_cov * 100:.2f}%")
    print(f"AVG BRANCH COVERAGE:    {avg_branch_cov * 100:.2f}%")
    print(f"{'='*40}")

if __name__ == "__main__":
    main()