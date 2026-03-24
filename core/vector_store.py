from abc import ABC, abstractmethod
from typing import List
from core.chunk import Chunk, RetrievedChunk

import numpy as np

class BaseVectorDB(ABC):
    @abstractmethod
    def add_chunks(self, chunks: List[Chunk], embeddings: np.ndarray):
        '''
        This function is used to add chunks to Vector Database
        '''
        pass 

    @abstractmethod
    def search(self, query: np.ndarray, k: int = 4) -> List[RetrievedChunk]:
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
