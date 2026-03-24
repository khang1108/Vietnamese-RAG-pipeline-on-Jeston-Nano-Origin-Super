from core.vector_store import BaseVectorDB
from core.logger import logger
from typing import Any
from pathlib import Path
from dataclasses import asdict
from core.chunk import Chunk, RetrievedChunk

import json
import faiss
import numpy as np

'''
FAISS Parameter of IndexIVFFlat:
    - 'nlist': in range of 4sqrt(N) to 6sqrt(N)
    - 'nprobe': Try to choose the best. 1, 2, 4, 8, 16,...
'''

class FAISSVectorDB(BaseVectorDB):
    INDEX_FILENAME = "local.index"
    CHUNKS_FILENAME = "chunks.json"

    def __init__(self, dimension: int, chunks: list[Chunk] | None = None):
        self.dimension = dimension

        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunks: list[Chunk] = list(chunks) if chunks is not None else []

    def add_chunks(self, chunks: list[Chunk], embeddings: np.ndarray):
        '''
        Add a batch of chunks to the index storage.

        Parameters:
            chunks (list[Chunk]): A list of chunk objects.
            embeddings (np.ndarray): A batch of embeddings with shape (n, dim).
        '''
        if len(chunks) == 0:
            logger.warning("[VectorDB] No chunks provided. Skip indexing.")
            return

        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        if embeddings.ndim != 2:
            raise ValueError(
                f"Embeddings must be a 2D array of shape (n, dim), got {embeddings.shape}."
            )

        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                "The number of embeddings must match the number of chunks."
            )

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}, got {embeddings.shape[1]}."
            )

        num_vectors = embeddings.shape[0]
        self.index.add(embeddings)
        self.chunks.extend(chunks)
        logger.success(
            f"[VectorDB] Successfully added {num_vectors} chunks. Current total: {self.index.ntotal}"
        )

    def search(self, query: np.ndarray, k: int = 4) -> list[RetrievedChunk]:
        '''
        Use similarity search to find the top-k nearest chunks.
        
        Parameters:
            query (np.ndarray): Query embedding.
            k (int): Number of nearest candidates.
        '''
        if self.index.ntotal == 0:
            logger.warning("[VectorDB] Search skipped because the index is empty.")
            return []

        if query.ndim == 1:
            query = query.reshape(1, -1)

        if query.dtype != np.float32:
            query = query.astype(np.float32)

        if query.shape[1] != self.dimension:
            raise ValueError(
                f"Query dimension mismatch: expected {self.dimension}, got {query.shape[1]}."
            )

        distances, indices = self.index.search(query, k)

        results: list[RetrievedChunk] = []
        for j, doc_id in enumerate(indices[0]):
            if doc_id != -1:
                results.append(
                    RetrievedChunk(
                        chunk=self.chunks[int(doc_id)],
                        score=float(distances[0][j]),
                    )
                )

        return results
    
    def save(self, filepath: str):
        '''
        Save 'index' to local storage.

        Args:
            filepath (str): Path to local in which 'index' file is saved.
        '''
        path_dir = Path(filepath)
        path_dir.mkdir(parents=True, exist_ok=True)

        index_path = path_dir / self.INDEX_FILENAME
        chunks_path = path_dir / self.CHUNKS_FILENAME

        faiss.write_index(self.index, str(index_path))
        with chunks_path.open("w", encoding="utf-8") as file:
            json.dump([asdict(chunk) for chunk in self.chunks], file, ensure_ascii=False, indent=2)

        logger.success(
            f"[VectorDB] Successfully saved index to {index_path} and metadata to {chunks_path}"
        )

    def load(self, filepath: str):
        '''
        Load index from local storage.

        Parameters:
            filepath (str): Path to your local index file.
        '''
        path_dir = Path(filepath)
        index_path = path_dir / self.INDEX_FILENAME
        chunks_path = path_dir / self.CHUNKS_FILENAME

        if not index_path.exists():
            raise FileNotFoundError(f"Index file does not exist: {index_path}")

        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunk metadata file does not exist: {chunks_path}")

        self.index = faiss.read_index(str(index_path))

        with chunks_path.open("r", encoding="utf-8") as file:
            chunk_payloads: list[dict[str, Any]] = json.load(file)

        self.chunks = [Chunk(**payload) for payload in chunk_payloads]

        if self.index.ntotal != len(self.chunks):
            raise ValueError(
                "Loaded index size does not match the number of stored chunks."
            )

        logger.success(
            f"[VectorDB] Successfully loaded {self.index.ntotal} vectors from {index_path}"
        )
