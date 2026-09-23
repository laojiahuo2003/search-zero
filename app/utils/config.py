"""
Centralized configuration via environment variables.

Every filesystem path the pipeline touches derives from SEARCH_ZERO_ROOT, so a
single env var moves models + data + outputs onto another disk:

    SEARCH_ZERO_ROOT=/mnt/workspace
    ├── models/    download_models.py output (Qwen, BGE embedder, BGE reranker)
    ├── data/      wiki index, HotpotQA, SFT trajectories
    └── outputs/   SFT and GRPO checkpoints

Unset -> the repo root, which is the historical layout.
"""
import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


def _env(key: str, default: str) -> str:
    """os.getenv, but an empty/whitespace value falls back to the default.

    `.env` files are full of commented-out placeholders; treating "" as unset
    keeps `SEARCH_ZERO_ROOT=` from silently collapsing every path to "".
    """
    val = os.getenv(key)
    return val.strip() if val and val.strip() else default


@dataclass
class Config:
    # --- LLM ---
    llm_api_key: str = _env("LLM_API_KEY", "sk-xxx")
    llm_base_url: str = _env("LLM_BASE_URL", "https://api.openai.com/v1")
    llm_model: str = _env("LLM_MODEL", "gpt-4o-mini")
    llm_temperature: float = float(_env("LLM_TEMPERATURE", "0.0"))

    # --- Search ---
    search_provider: str = _env("SEARCH_PROVIDER", "duckduckgo")  # duckduckgo | tavily
    tavily_api_key: Optional[str] = os.getenv("TAVILY_API_KEY") or None
    search_max_results: int = int(_env("SEARCH_MAX_RESULTS", "5"))

    # --- Paths ---
    # Single root for every downloaded / derived artifact. Point it at a big
    # disk (e.g. /mnt/workspace) to keep models + data + outputs out of the
    # repo. Unset -> repo root, which is the historical layout.
    root_dir: str = _env(
        "SEARCH_ZERO_ROOT",
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    )
    models_dir: str = os.path.join(root_dir, "models")
    data_dir: str = os.path.join(root_dir, "data")
    outputs_dir: str = os.path.join(root_dir, "outputs")
    wiki_cache_dir: str = os.path.join(root_dir, "wiki_cache")

    # Base model for SFT / GRPO / eval. Override with BASE_MODEL to use a
    # different size (Qwen2.5-3B-Instruct, Qwen2.5-1.5B-Instruct, ...).
    base_model: str = _env(
        "BASE_MODEL", os.path.join(models_dir, "Qwen2.5-7B-Instruct")
    )

    # --- Embedding ---
    # Defaults live under $SEARCH_ZERO_ROOT/models (see download_models.py).
    # Set to a HF repo id (BAAI/bge-small-en-v1.5) to use the HF cache instead.
    embedding_model: str = _env(
        "EMBEDDING_MODEL", os.path.join(models_dir, "bge-small-en-v1.5")
    )

    # --- Reranker ---
    reranker_model: str = _env(
        "RERANKER_MODEL", os.path.join(models_dir, "bge-reranker-base")
    )

    # --- Agent ---
    agent_max_steps: int = int(_env("AGENT_MAX_STEPS", "5"))
    agent_top_k_retrieval: int = int(_env("AGENT_TOP_K", "3"))

    # --- Server ---
    host: str = _env("HOST", "0.0.0.0")
    port: int = int(_env("PORT", "8000"))

    # --- Tracking (SwanLab) ---
    # api_key is only needed for mode=online. Without one the run stays local.
    swanlab_api_key: Optional[str] = os.getenv("SWANLAB_API_KEY") or None
    swanlab_project: str = _env("SWANLAB_PROJECT", "search-zero")
    swanlab_workspace: Optional[str] = os.getenv("SWANLAB_WORKSPACE") or None
    # None -> resolve_mode() picks online when a key exists, else local.
    swanlab_mode: Optional[str] = os.getenv("SWANLAB_MODE") or None
    swanlab_logdir: str = _env("SWANLAB_LOGDIR", "swanlog")
    swanlab_experiment: Optional[str] = os.getenv("SWANLAB_EXP_NAME") or None

    # --- Derived artifact paths — these are what the training scripts read. ---
    wiki_index_path: str = os.path.join(data_dir, "wiki_index.json")
    hotpotqa_dev_path: str = os.path.join(data_dir, "hotpotqa_dev.json")
    hotpotqa_train_path: str = os.path.join(data_dir, "hotpotqa_train_500.json")
    hotpotqa_eval_path: str = os.path.join(data_dir, "hotpotqa_eval_100.json")
    sft_data_dir: str = os.path.join(data_dir, "sft")
    sft_trajectories_path: str = os.path.join(sft_data_dir, "sft_trajectories.jsonl")
    sft_filtered_path: str = os.path.join(
        sft_data_dir, "sft_trajectories_filtered.jsonl"
    )
    sft_checkpoint: str = os.path.join(outputs_dir, "search_r1_sft")
    grpo_output_dir: str = os.path.join(outputs_dir, "search_r1_grpo_search")
    faiss_index_path: str = os.path.join(data_dir, "faiss_index")


_config: Optional[Config] = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config
