from abc import ABC, abstractmethod
from core.logger import logger

class TemplatePrompt(ABC):
    @abstractmethod
    def create_prompt(self, query: str, context: str):
        pass