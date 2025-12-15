import re

def clean_imports_and_fix_logic(example):
    """
    Removes specific 'solution' imports, handles aliasing, and injects 
    standard library imports to ensure tests are runnable.
    Student Note: This was necessary because the model often forgot imports or used relative imports 
    that break in the sandbox.
    """
    test_code = example['test']
    
    # A. Handle "from solution import func as alias" OR "from solution import func"
    import_pattern = r"^from solution import (\w+)(?:\s+as\s+(\w+))?"
    
    match = re.search(import_pattern, test_code, flags=re.MULTILINE)
    
    if match:
        imported_func = match.group(1)
        alias = match.group(2)
        
        # Remove the import line entirely
        test_code = re.sub(import_pattern, "", test_code, flags=re.MULTILINE).strip()
        
        # CRITICAL: If an alias was used (e.g., 'import two_sum as ts'),
        # we replace all instances of 'ts' with 'two_sum' in the code body.
        if alias and alias != imported_func:
            test_code = re.sub(r"\b" + re.escape(alias) + r"\b", imported_func, test_code)
            
    # B. Remove generic "import solution" or "from solution import *"
    test_code = re.sub(r"^import solution\s*$", "", test_code, flags=re.MULTILINE)
    test_code = re.sub(r"^from solution import \*", "", test_code, flags=re.MULTILINE)

    # C. Inject "The Kitchen Sink" of Standard Imports
    common_imports = [
        "import pytest",
        "from typing import *",
        "import math",
        "import collections",
        "import heapq",
        "import itertools",
        "import random",
        "import sys",
        "import re",
        "import bisect"
    ]
    
    header = "\n".join(common_imports)
    example['test'] = f"{header}\n\n{test_code}"
    return example

def format_scot(example):
    """
    Wraps the code and the cleaned test suite into the SCoT prompt format.
    Ensures the model outputs reasoning steps before the code.
    """
    code = example['solution'].replace("```python", "").replace("```", "").strip()
    target = example['test'] # This is now the CLEANED version
    
    # The exact system prompt used in the Thesis
    prompt = (
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
    return {
        "prompt": prompt,
        "completion": target,
        "text": f"{prompt}```python\n{target}\n```<|im_end|>",
        "entry_point": example.get('entry_point', 'solution'),
        "code_body": code
    }
