"""
Run benchmark evaluation comparing Baseline RAG vs Search-R1 Mini.
Usage: python scripts/run_eval.py [data_path] [limit]
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.evaluation.benchmark import run_full_benchmark

if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else None
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    run_full_benchmark(data_path, limit)
