import json
from pathlib import Path

from adapters.embedders.VNSbert import VNSbert
from adapters.llms.qwenllm import QwenLLM
from adapters.prompts.prompt import GeneralPrompt
from adapters.vector_stores.faiss_store import FAISSVectorDB
from core.chunk import Chunk
from core.logger import logger
from pipeline.rag_pipeline import RAGPipeline
from services.indexing_service import IndexingService

MODEL_PATH = './models/qwen2.5-3b-instruct-q5_k_m.gguf'
RAW_CHUNKS_PATH = './data/chunks.json'
INDEX_DIR = './storage/faiss'


def load_chunks_from_json(filepath: str) -> list[Chunk]:
    with open(filepath, "r", encoding="utf-8") as file:
        payload = json.load(file)

    chunks: list[Chunk] = []
    for index, item in enumerate(payload):
        chunks.append(
            Chunk(
                doc_id=item["doc_id"],
                chunk_id=item.get("chunk_id", f'{item["doc_id"]}_{index}'),
                content=item["content"],
                chunk_index=item.get("chunk_index", index),
                source=item.get("source"),
                title=item.get("title"),
                metadata=item.get("metadata"),
            )
        )

    return chunks

def get_or_create_vector_db(embedder: VNSbert) -> FAISSVectorDB:
    vector_db = FAISSVectorDB(dimension=embedder.dimension)
    index_path = Path(INDEX_DIR) / FAISSVectorDB.INDEX_FILENAME

    if index_path.exists():
        logger.info(f"[App] Found existing FAISS index at {INDEX_DIR}. Loading...")
        vector_db.load(INDEX_DIR)
        return vector_db

    logger.info("[App] No index found. Building a new index from data/chunks.json...")
    chunks = load_chunks_from_json(RAW_CHUNKS_PATH)

    indexing_service = IndexingService(embedder=embedder, vector_store=vector_db)
    indexing_service.index_chunks(chunks=chunks, savepath=INDEX_DIR)
    return vector_db

def build_pipeline() -> RAGPipeline:
    embedder = VNSbert()
    vector_db = get_or_create_vector_db(embedder)

    prompt_builder = GeneralPrompt()
    llm = QwenLLM(
        model_path=MODEL_PATH,
        n_ctx=8192,
        n_gpu_layers=-1,
        n_batch=512,
        chat_format='chatml',  
        verbose=False,
    )

    return RAGPipeline(
        embedder=embedder,
        vectordb=vector_db,
        llm=llm,
        prompt_builder=prompt_builder,
    )


def main() -> None:
    logger.info("[App] Starting Vietnamese RAG pipeline...")
    rag = build_pipeline()

    while True:
        query = input("\nUser: ").strip()

        if not query:
            continue

        if query.lower() in {"exit", "quit", "q"}:
            print("Bye.")
            break

        result = rag.run(query=query, top_k=3)

        print("\nAssistant:")
        print(result["answer"])

        print("\nRetrieved Chunks:")
        for idx, item in enumerate(result["contexts"], start=1):
            title = item.chunk.title or item.chunk.doc_id
            print(f"{idx}. score={item.score:.4f} | {title}")


if __name__ == "__main__":
    main()
