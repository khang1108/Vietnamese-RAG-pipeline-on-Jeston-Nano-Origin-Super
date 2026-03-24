from core.embedder import BaseEmbedder
from core.vector_store import BaseVectorDB
from core.chunk import Chunk
from core.logger import logger

class IndexingService:
    def __init__(self, embedder: BaseEmbedder, vector_store: BaseVectorDB):
        self.embedder = embedder
        self.vector_store = vector_store

    def index_chunks(self, chunks: list[Chunk], savepath: str):
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedder.embed_texts(texts)
        logger.success(f"[Indexing] Successfully indexed {len(embeddings)} chunks")
        self.vector_store.add_chunks(chunks=chunks, embeddings=embeddings)
        
        self.vector_store.save(savepath)
        logger.success(f"[Indexing] Saved index to {savepath}")
