"""
Pre-download every model the pipeline needs, into $SEARCH_ZERO_ROOT/models.

Covers:
  - Qwen2.5-7B-Instruct   (SFT / GRPO / eval base model)
  - BGE small en v1.5     (embedder)
  - BGE reranker base     (cross-encoder reranker)

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --skip-llm      # only BGE pair
    python scripts/download_models.py --llm Qwen/Qwen2.5-3B-Instruct

Respects HF_ENDPOINT, so `HF_ENDPOINT=https://hf-mirror.com` works from CN.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.config import get_config

# repo_id -> local directory name under $SEARCH_ZERO_ROOT/models
LLM_REPO = "Qwen/Qwen2.5-7B-Instruct"
EMBEDDER_REPO = "BAAI/bge-small-en-v1.5"
RERANKER_REPO = "BAAI/bge-reranker-base"


def download(repo_id: str, dest: str) -> bool:
    """Fetch a repo into a plain local directory (no symlinks, offline-ready)."""
    if os.path.exists(os.path.join(dest, "config.json")):
        print(f"  already present: {dest}")
        return True

    os.makedirs(dest, exist_ok=True)
    print(f"  {repo_id} -> {dest}")
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=repo_id, local_dir=dest)
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        print(f"  retry manually: huggingface-cli download {repo_id} --local-dir {dest}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", default=LLM_REPO, help="base model repo id")
    parser.add_argument("--skip-llm", action="store_true",
                        help="only download the BGE embedder + reranker")
    args = parser.parse_args()

    cfg = get_config()
    print(f"Models root: {cfg.models_dir}")
    os.makedirs(cfg.models_dir, exist_ok=True)

    ok = True

    if not args.skip_llm:
        name = args.llm.rsplit("/", 1)[-1]
        print("Downloading base model...")
        ok &= download(args.llm, os.path.join(cfg.models_dir, name))

    print("Downloading BGE embedding model...")
    ok &= download(EMBEDDER_REPO, os.path.join(cfg.models_dir, "bge-small-en-v1.5"))

    print("Downloading BGE reranker model...")
    ok &= download(RERANKER_REPO, os.path.join(cfg.models_dir, "bge-reranker-base"))

    if not ok:
        print("\nSome downloads failed.")
        sys.exit(1)

    print("\nAll models downloaded successfully.")
    print("\nPoint .env at the local copies so nothing re-downloads at runtime:")
    print(f"  BASE_MODEL={cfg.base_model}")
    print(f"  EMBEDDING_MODEL={os.path.join(cfg.models_dir, 'bge-small-en-v1.5')}")
    print(f"  RERANKER_MODEL={os.path.join(cfg.models_dir, 'bge-reranker-base')}")


if __name__ == "__main__":
    main()
