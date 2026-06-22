"""
LLM client for grounded generation via Groq API.

Groq hosts open-weight models (Llama 3.3 70B, Llama 3.1 8B) with
very fast inference. Free tier: 30 req/min, 6k tokens/min for 70B.

Key concept — temperature:
  Controls randomness. temperature=0 → deterministic (picks the most
  likely token every time). For regulatory text, we want conservative,
  reproducible answers — no creative paraphrasing of legal rules.
"""

from typing import Protocol

from dotenv import load_dotenv

load_dotenv()


class LLMClient(Protocol):
    """Any object with a complete() method works as an LLM client."""

    def complete(self, system: str, user: str) -> str: ...

    @property
    def model_name(self) -> str: ...


class GroqClient:
    """Groq-hosted LLM (Llama 3.3 70B or 3.1 8B)."""

    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        from groq import Groq

        self._client = Groq()
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    def complete(self, system: str, user: str) -> str:
        result = self.complete_with_usage(system, user)
        return result["content"]

    def complete_with_usage(self, system: str, user: str) -> dict:
        """Return content + token usage for cost tracking."""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
            max_tokens=1024,
        )
        usage = response.usage
        return {
            "content": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
            },
            "model": self._model,
        }
