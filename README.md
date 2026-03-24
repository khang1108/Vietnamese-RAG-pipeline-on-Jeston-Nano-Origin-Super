# Vietnamese RAG Pipeline

A modular **Retrieval-Augmented Generation (RAG) pipelin**e for **Vietnamese educational and mental health assistance** on Jetson edge devices.

## Overview
This project focuses on building a **plug-in based RAG system** where embedding models, vector databases, and local LLMs can be swapped independently.

## Repository Architecture

```
Vietnamese_RAG
├── adapters/
│   ├── embedders/
│   ├── llms/
│   ├── prompts/
│   └── vector_stores/
├── config/
├── core/
│   ├── embedder.py
│   ├── llm.py
│   ├── logger.py
│   ├── prompt_builder.py
│   └── vector_store.py
├── pipeline/
│   └── rag_pipeline.py
└── main.py
```