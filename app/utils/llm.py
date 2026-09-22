"""
Unified LLM client supporting any OpenAI-compatible API.
Works with OpenAI, Qwen, DeepSeek, vLLM, etc.
"""
from openai import OpenAI
from app.utils.config import get_config


class LLMClient:
    """Thin wrapper around OpenAI-compatible chat completions."""

    def __init__(self):
        cfg = get_config()
        self.client = OpenAI(api_key=cfg.llm_api_key, base_url=cfg.llm_base_url)
        self.model = cfg.llm_model
        self.temperature = cfg.llm_temperature

    def generate(self, messages: list[dict], stop: list[str] | None = None, max_tokens: int = 2048) -> str:
        """Send a chat completion request and return the response text."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=max_tokens,
            stop=stop,
        )
        return response.choices[0].message.content or ""

    def generate_json(self, messages: list[dict], max_tokens: int = 2048) -> str:
        """Generate with JSON mode enabled."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or ""


_llm: LLMClient | None = None


def get_llm() -> LLMClient:
    global _llm
    if _llm is None:
        _llm = LLMClient()
    return _llm
