"""
apply_optimized_signature.py
Reads optimized_program.json and updates signatures.py automatically
"""

import os
import json
import re

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OPTIMIZED_JSON = os.path.join(project_root, 'data', 'optimization_results.json')
SIGNATURES_FILE = os.path.join(project_root, 'src', 'signatures.py')


def apply():
    # ── Load optimized signature from results ─────────────────────────────────
    with open(OPTIMIZED_JSON, 'r', encoding='utf-8') as f:
        results = json.load(f)

    new_sig = results["optimized_signature"]
    old_sig = results["original_signature"]

    print(f"OLD: {old_sig}")
    print(f"NEW: {new_sig}")

    # ── Read current signatures.py ────────────────────────────────────────────
    with open(SIGNATURES_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    # ── Replace the docstring inside HelpSteer2Signature ─────────────────────
    # Matches: """...""" inside the class
    pattern = r'(class HelpSteer2Signature\(dspy\.Signature\):\s+""")(.*?)(""")'
    replacement = rf'\g<1>{new_sig}\g<3>'

    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

    if new_content == content:
        print("WARNING: Pattern not matched — signatures.py was NOT updated.")
        print("Check the class name or docstring format.")
        return

    # ── Write back ────────────────────────────────────────────────────────────
    with open(SIGNATURES_FILE, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print(f"\nSignatures.py updated successfully!")
    print(f"\nNew HelpSteer2Signature docstring:\n  '{new_sig}'")


if __name__ == "__main__":
    apply()