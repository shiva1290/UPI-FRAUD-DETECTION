"""
LLM client abstraction (DIP: depend on abstraction, not Groq directly).
OCP: add new providers by implementing this interface.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class LLMClient(ABC):
    """Abstract LLM client. Implement for different providers (Groq, OpenAI, etc.)."""

    @abstractmethod
    def complete(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Send messages and return response content."""
        pass


class GroqLLMClient(LLMClient):
    """Groq API implementation."""

    def __init__(self, api_key: str, model: str = 'llama-3.3-70b-versatile'):
        import httpx
        from groq import Groq
        self.model = model
        try:
            self._client = Groq(api_key=api_key, http_client=httpx.Client())
        except TypeError:
            try:
                self._client = Groq(api_key=api_key, proxies=None)
            except TypeError:
                self._client = Groq(api_key=api_key)

    def complete(self, messages: List[Dict[str, str]], temperature: float = 0.1, **kwargs) -> str:
        completion = self._client.chat.completions.create(
            messages=messages,
            model=self.model,
            temperature=temperature,
            response_format={"type": "json_object"},
            **kwargs
        )
        return completion.choices[0].message.content
