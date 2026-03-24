from abc import ABC, abstractmethod
from typing import List, Dict, Any

import numpy as np

class BaseVectorDB(ABC):
    @abstractmethod
    def add_texts(self, texts: List[str], embeddings: np.ndarray):
        '''
        This function is used to add texts to Vector Database
        '''
        pass 

    @abstractmethod
    def search(self, query: np.ndarray, k: int = 4) -> List[str]:
        '''
        This function is used to search 'k' candidates that are similarity to the query.
        '''
        pass

    @abstractmethod
    def save(self, filepath: str):
        pass

    @abstractmethod
    def load(self, filepath: str):
        pass