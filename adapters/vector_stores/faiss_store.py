from core.vector_store import BaseVectorDB
from core.logger import logger
from typing import List, Any
from pathlib import Path

import os
import faiss
import numpy as np

'''
FAISS Parameter of IndexIVFFlat:
    - 'nlist': in range of 4sqrt(N) to 6sqrt(N)
    - 'nprobe': Try to choose the best. 1, 2, 4, 8, 16,...
'''

class FAISSVectorDB(BaseVectorDB):
    def __init__(self, dimension: int, chunks: List[Any]):
        self.dimension = dimension

        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunks = chunks
        self.current_id = 0

    def add_texts(self, texts, embeddings):
        '''
        Add a batch of texts to the index storage.

        Parameters:
            texts (List[str]): A list of texts
            embeddings (np.ndarray): A list of embeddings
        '''
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        num_vectors = embeddings.shape[0] # To get the number of vector. (n, dim)
        self.index.add(embeddings)

        for i, text in enumerate(texts):
            self.chunks[self.current_id + i] = text

        self.current_id += num_vectors
        logger.success(f"[VectorDB] Successfully add {num_vectors} chunks to VectorDB. Current total of vectors: {self.index.ntotal}")

    def search(self, query, k = 4):
        '''
        Use similarity search to find 'top_k' candidates from index storage.
        
        Parameters:
            query (str): A query 
            k (int): Top-k candidates are most similarity to the query
        '''
        if query.dtype != np.float32:
            query = query.astype(np.float32)

        dist, indices = self.index.search(query, k)

        results = []
        for j, doc_id in enumerate(indices[0]):
            if doc_id != -1:
                text = self.chunks[doc_id]
                dist = dist[0][j]

                results.append((text, dist))

        return results
    
    def save(self, filepath: str):
        '''
        Save 'index' to local storage.

        Args:
            filepath (str): Path to local in which 'index' file is saved.
        '''
        path_dir = Path(filepath)
        path_dir.mkdir(exist_ok=True)

        faiss.write_index(self.index, f"{filepath}/local.index")
        logger.success(f"[VectorDB] Successfully save index to local in {filepath}/local.index")

    def load(self, filepath: str):
        '''
        Load index from local storage.

        Parameters:
            filepath (str): Path to your local index file.
        '''

        if(os.path.exists(filepath) == False):
            logger.error("[VectorDB] File is not existed")
            return
        
        self.index = faiss.read_index(filepath)
        logger.success(f"[VectorDB] Successfully loaded {self.index.ntotal} from the local storage.")