# Sanskrit RAG Pipeline

A production-ready **Retrieval-Augmented Generation (RAG)** system designed for Sanskrit document question-answering, built with LangChain, HuggingFace, and ChromaDB.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Requirements](#requirements)
4. [Setup & Installation](#setup--installation)
5. [Configuration](#configuration)
6. [Running the Pipeline](#running-the-pipeline)
7. [Running Tests](#running-tests)
8. [Architecture Summary](#architecture-summary)

---

## Overview

This system ingests Sanskrit PDF documents, chunks and embeds them using a multilingual sentence-transformer model, stores them in a ChromaDB vector database, and answers natural-language queries by retrieving relevant context and passing it to a HuggingFace-hosted LLM.

**Key design decisions for Sanskrit:**
- Embedding model `paraphrase-multilingual-MiniLM-L12-v2` supports Devanagari script natively.
- `chunk_size=500 / chunk_overlap=50` preserves shloka and verse boundaries.
- `search_type="similarity"` with `k=3` retrieves the top 3 most relevant passages per query.

---

## Project Structure

```
rag_project_pritesh/
├── main.py  # Main entry point
|___app.py                    
├── requirements.txt
├── README.md
├── info.txt
├── data/
│   └── raw/                      # Place your Sanskrit PDF files here
└── src/
    ├── config.py                 # All settings (LLM, chunking, paths)
    ├── utils/
    │   └── logger.py             # Centralised logging
    ├── data_ingestion/
    │   └── load_documents.py     # PDF loader (PyPDFLoader)
    ├── preprocessing/
    │   └── text_splitter.py      # RecursiveCharacterTextSplitter
    ├── embeddings/
    │   └── vectore_store.py      # ChromaDB build & load
    ├── retrieval/
    │   └── retrival.py           # Similarity retriever
    ├── llm/
    │   └── llm_model.py          # HuggingFace LLM (ChatHuggingFace)
    ├── pipeline/
    │   └── rag_pipeline.py       # End-to-end RAG chain
    
```

---

## Requirements

- Python 3.10+
- A HuggingFace API token with access to the model endpoint

---

## Setup & Installation

### 1. Clone / unzip the project

```bash
unzip rag_pipeline_v2.zip
cd rag_project2
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create a `.env` file

```env
HUGGINGFACEHUB_API_TOKEN=hf_your_token_here
```

All other settings have sensible defaults (see [Configuration](#configuration)).

### 5. Add your Sanskrit PDFs

Copy your PDF files into `data/raw/`:

```bash
cp /path/to/your/sanskrit_docs/*.pdf data/raw/
```

---

## Configuration

All settings live in `src/config.py` and can be overridden via `.env`:

| Variable | Default | Description |
|---|---|---|
| `HUGGINGFACEHUB_API_TOKEN` | _(required)_ | HuggingFace API key |
| `LLM_REPO_ID` | `openai/gpt-oss-120b:groq` | LLM endpoint |
| `LLM_MAX_NEW_TOKENS` | `400` | Max tokens in response |
| `LLM_TEMPERATURE` | `0.3` | Sampling temperature |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Sentence transformer |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `RETRIEVAL_K` | `3` | Top-K passages retrieved |
| `DATA_FOLDER` | `data/raw` | Folder with source PDFs |
| `VECTOR_DB_DIR` | `vector_db` | ChromaDB persistence path |

---

## Running the Pipeline

### Step 1 — Ingest documents (run once)

```bash
python app.py --ingest
```

Loads all PDFs from `data/raw/`, splits them into chunks, embeds them, and saves the ChromaDB index to `vector_db/`.

### Step 2a — Single query

```bash
python app.py --query "What is the meaning of Dharma in the Bhagavad Gita?"
```

### Step 2b — Interactive mode

```bash
python app.py
```

Starts a prompt loop. Type `exit` or `quit` to stop.

---


## Architecture Summary

```
[data/raw PDFs]
      │
      ▼
DocumentLoader          load_documents.py
      │
      ▼
TextSplitter            text_splitter.py   chunk_size=500, overlap=50
      │
      ▼
VectorStore.create()    vectore_store.py   HuggingFace embeddings → ChromaDB
      │
      ▼  (persisted to vector_db/)
      │
[query time]
      │
      ▼
VectorStore.load()  →  Retriever (k=3)  →  RAG Chain  →  LLMModel  →  Answer
```

---

## Notes on Sanskrit Processing

- **Script**: Devanagari is fully supported by the chosen multilingual embedding model.
- **Chunking**: `RecursiveCharacterTextSplitter` respects paragraph and sentence boundaries, which maps well to shloka structure.
- **PyPDFLoader**: Handles Devanagari text extraction correctly when PDFs are text-based (not scanned images). For scanned Sanskrit manuscripts, an OCR step would be needed before ingestion.
