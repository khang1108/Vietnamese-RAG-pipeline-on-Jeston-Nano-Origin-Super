from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    content: str
    chunk_index: Optional[int] = None

    source: Optional[str] = None
    title: Optional[str] = None

    metadata: Optional[Dict] = None

@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float