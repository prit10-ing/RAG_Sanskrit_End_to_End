"""
config.py - Final working configuration for Sanskrit RAG Pipeline
"""

import logging
import os
from dotenv import load_dotenv

load_dotenv()

# -- Logging -------------------------------------------------------------------
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# -- Data paths ----------------------------------------------------------------
DATA_FOLDER = os.getenv("DATA_FOLDER", "data/raw")
VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", "vector_db")

# -- LLM -----------------------------------------------------------------------
LLM_REPO_ID = os.getenv("LLM_REPO_ID", "openai/gpt-oss-120b:groq")
LLM_TASK = "text-generation"
LLM_MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", 512))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.2))

# -- Embeddings ----------------------------------------------------------------
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

# -- Text splitting ------------------------------------------------------------
# FIX: Smaller chunks = more precise retrieval for cross-language queries.
# With chunk_size=1500, one chunk could contain two different stories mixed
# together, confusing the LLM. Smaller chunks = one story idea per chunk.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))

# -- Retrieval -----------------------------------------------------------------
# FIX: k=7 covers more of the document for cross-language (English->Sanskrit)
# queries where the embedding similarity score is naturally lower.
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", 7))
RETRIEVAL_SEARCH_TYPE = "similarity"

# -- Prompt --------------------------------------------------------------------
PROMPT_TEMPLATE = """You are an expert assistant for Sanskrit documents.
The documents contain Sanskrit stories with some English translations.

Use the context chunks below to answer the question.

LANGUAGE RULE — strictly follow this:
- Question in Sanskrit/Devanagari script → answer in Sanskrit
- Question in English → answer in English  
- Question in Hindi → answer in Hindi

ANSWERING RULES:
- Read ALL context chunks carefully before answering.
- For story questions: narrate the story found in the chunks.
- For "when does God help" type questions: the answer is that God helps only
  when a person makes their own effort first (प्रयत्न).
- For broad questions (moral, topic, summary): synthesize across all chunks.
- Answer naturally. Do not copy Sanskrit text verbatim — explain it clearly.
- Write ONLY the answer. No preamble, no repeating the question.
- Say "I could not find the answer." ONLY IF none of the chunks relate at all.

CONTEXT:
{context}

Question: {question}

Answer:"""