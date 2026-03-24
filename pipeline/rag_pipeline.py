from core.embedder import BaseEmbedder
from core.vector_store import BaseVectorDB
from core.llm import BaseLLM
from core.prompt_builder import TemplatePrompt
from core.chunk import RetrievedChunk
from core.logger import logger

import time

class RAGPipeline:
    def __init__(self, embedder: BaseEmbedder, vectordb: BaseVectorDB, llm: BaseLLM, prompt_builder: TemplatePrompt):
        self.embedder = embedder
        self.vectordb = vectordb
        self.llm = llm
        self.prompt_builder = prompt_builder

    def retrieve(self, question: str, top_k: int = 2) -> list[RetrievedChunk]:
        logger.info(f"[Info] User asked: {question[:50]}...")

        start_time = time.time()

        try:
            query_vec = self.embedder.embed_text(question)
            results = self.vectordb.search(query_vec, top_k)
            end_time = time.time()
            
            logger.success(f"[RAG] Successfully retrieved {len(results)} / {top_k} in {(end_time - start_time):.4f}")
        except Exception as e:
            logger.error(f"[RAG] Get an unknown error: {e}")
            raise

        return results
    
    def generate(self, query: str, retrieved_chunks: list[RetrievedChunk]) -> str:
        context_text = "\n\n".join(item.chunk.content for item in retrieved_chunks)
        messages = self.prompt_builder.create_prompt(query, context_text)

        try:
            response = self.llm.generate(messages, temperature=0.2, top_k=40, top_p=0.9)
            logger.success(f"[RAG] Successfully generated response for user's query")
        except Exception as e:
            logger.error(f"[RAG] Got an unknown error: {e}")
            raise
        return response["choices"][0]["message"]["content"]
    
    def run(self, query: str, top_k: int = 3):
        retrieved_chunks = self.retrieve(query, top_k=top_k)
        answer = self.generate(query, retrieved_chunks)

        return {
            "query": query,
            "contexts": retrieved_chunks,
            "answer": answer
        }
