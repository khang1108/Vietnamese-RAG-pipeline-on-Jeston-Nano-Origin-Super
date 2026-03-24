from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
from typing import List

class BaseLLM(ABC):
    @abstractmethod
    def generate(self, chunks: List[str]) -> str:
        pass