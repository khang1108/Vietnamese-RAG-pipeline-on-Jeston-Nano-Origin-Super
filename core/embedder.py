from abc import ABC, abstractmethod
from typing import List, Any

class BaseEmbedder(ABC):
    @abstractmethod
    def embed_text(self, text) -> List[Any]:
        '''
        This funtion is used to embed the input text into embedding vector
        '''
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[Any]]:
        pass
    
    def query_rewrite(self, str):
        pass