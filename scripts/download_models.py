"""
Pre-download embedding and reranker models for offline use.
Usage: python scripts/download_models.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Downloading BGE embedding model...")
from sentence_transformers import SentenceTransformer
SentenceTransformer("BAAI/bge-small-en-v1.5")
print("Done.")

print("Downloading BGE reranker model...")
from sentence_transformers import CrossEncoder
CrossEncoder("BAAI/bge-reranker-base")
print("Done.")

print("All models downloaded successfully.")
