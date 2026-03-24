from core.embedder import BaseEmbedder
from core.vector_store import BaseVectorDB

from core.logger import logger

import time

class RAGPipeline:
    def __init__(self, embedder: BaseEmbedder, vectordb: BaseVectorDB):
        self.embedder = embedder
        self.vectordb = vectordb

    def retrieve(self, question: str, top_k: int = 2):
        logger.info(f"[Info] User asked: {question[:50]}...")

        start_time = time.time()

        try:
            query_vec = self.embedder.embed_text(question)
            results = self.vectordb.search(query_vec, top_k)
            end_time = time.time()
            
            logger.success(f"[RAG] Successfully retrieved {len(results)} / {top_k} in {(end_time - start_time):.4f}")
        except Exception as e:
            logger.error(f"[RAG] Get an unknown error: {e}")
    
        return [res[0] for res in results] 
    