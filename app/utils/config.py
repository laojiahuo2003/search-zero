"""
Centralized configuration via environment variables.
"""
import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # --- LLM ---
    llm_api_key: str = os.getenv("LLM_API_KEY", "sk-xxx")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))

    # --- Search ---
    search_provider: str = os.getenv("SEARCH_PROVIDER", "duckduckgo")  # duckduckgo | tavily
    tavily_api_key: Optional[str] = os.getenv("TAVILY_API_KEY", None)
    search_max_results: int = int(os.getenv("SEARCH_MAX_RESULTS", "5"))

    # --- Embedding ---
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    # --- Reranker ---
    reranker_model: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")

    # --- Agent ---
    agent_max_steps: int = int(os.getenv("AGENT_MAX_STEPS", "5"))
    agent_top_k_retrieval: int = int(os.getenv("AGENT_TOP_K", "3"))

    # --- Server ---
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))

    # --- Tracking (SwanLab) ---
    # api_key is only needed for mode=online. Without one the run stays local.
    swanlab_api_key: Optional[str] = os.getenv("SWANLAB_API_KEY", None)
    swanlab_project: str = os.getenv("SWANLAB_PROJECT", "search-zero")
    swanlab_workspace: Optional[str] = os.getenv("SWANLAB_WORKSPACE", None)
    # None -> resolve_mode() picks online when a key exists, else local.
    swanlab_mode: Optional[str] = os.getenv("SWANLAB_MODE", None)
    swanlab_logdir: str = os.getenv("SWANLAB_LOGDIR", "swanlog")
    swanlab_experiment: Optional[str] = os.getenv("SWANLAB_EXP_NAME", None)

    # --- Paths ---
    data_dir: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
    faiss_index_path: str = os.path.join(data_dir, "faiss_index")


_config: Optional[Config] = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config
