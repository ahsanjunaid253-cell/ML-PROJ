import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))     # folder this file lives in


def run(args):
    print("\n>>> running:", " ".join(args))
    result = subprocess.run([sys.executable] + args, cwd=ROOT)
    if result.returncode != 0:
        print("Step failed, stopping.")
        sys.exit(1)


run(["training/pre_processed.py"])
run(["training/train.py"])
run(["-m", "uvicorn", "backend.main:app", "--reload"])   # keeps running until Ctrl+C
