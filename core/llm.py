from abc import ABC, abstractmethod
from typing import Any

class BaseLLM(ABC):
    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        max_tokens: int = 512,
        stop: list[str] | None = None,
        response_format: Any | None = None,
        stream: bool = False,
    ) -> Any:
        pass
