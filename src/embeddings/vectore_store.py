# """
# embeddings/vectore_store.py
# ============================
# Builds and loads a Chroma vector store with HuggingFace embeddings.
# Your original model name and Chroma settings are preserved exactly.
# """

# import logging
# from langchain_chroma import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_core.documents import Document
# from src.config import EMBEDDING_MODEL, VECTOR_DB_DIR

# logger = logging.getLogger(__name__)


# class VectorStore:
#     """Manages creation and loading of the Chroma vector store."""

#     def __init__(
#         self,
#         embedding_model_name: str = EMBEDDING_MODEL,
#         persist_directory: str = VECTOR_DB_DIR,
#     ):
#         self.persist_directory = persist_directory
#         self._db: Chroma | None = None

#         logger.info("Loading embedding model: %s", embedding_model_name)
#         try:
#             self._embedding_model = HuggingFaceEmbeddings(
#                 model_name=embedding_model_name
#             )
#             logger.info("Embedding model loaded.")
#         except Exception as e:
#             logger.error("Failed to load embedding model: %s", e)
#             raise

#     # ── public ─────────────────────────────────────────────────────────────────

#     def create(self, chunks: list[Document]) -> Chroma:
#         """Build a new Chroma DB from document chunks and persist it to disk."""
#         if not chunks:
#             raise ValueError("Cannot build vector store from an empty chunk list.")

#         logger.info(
#             "Building Chroma vector store from %d chunk(s)…", len(chunks)
#         )
#         try:
#             self._db = Chroma.from_documents(
#                 documents=chunks,
#                 embedding=self._embedding_model,
#                 persist_directory=self.persist_directory,
#             )
#             logger.info(
#                 "Vector store saved to '%s'.", self.persist_directory
#             )
#             return self._db
#         except Exception as e:
#             logger.error("Failed to create vector store: %s", e)
#             raise

#     def load(self) -> Chroma:
#         """Load an existing Chroma DB from disk."""
#         if self._db is not None:
#             logger.debug("Returning cached vector store.")
#             return self._db

#         logger.info(
#             "Loading Chroma vector store from '%s'…", self.persist_directory
#         )
#         try:
#             self._db = Chroma(
#                 persist_directory=self.persist_directory,
#                 embedding_function=self._embedding_model,
#             )
#             logger.info("Vector store loaded successfully.")
#             return self._db
#         except Exception as e:
#             logger.error("Failed to load vector store: %s", e)
#             raise


# # ── module-level convenience functions (mirror original signatures) ────────────

# def create_vector_store(chunks: list[Document]) -> Chroma:
#     """Original function signature preserved for compatibility."""
#     return VectorStore().create(chunks)


# def load_vector_store() -> Chroma:
#     """Load the persisted vector store from the default directory."""
#     return VectorStore().load()


"""
Vector Store using ChromaDB + HuggingFace Embeddings
"""

import logging

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from src.config import EMBEDDING_MODEL, VECTOR_DB_DIR

# logger
logger = logging.getLogger(__name__)


class VectorStore:

    def __init__(
        self,
        embedding_model=EMBEDDING_MODEL,
        persist_directory=VECTOR_DB_DIR,
    ):

        self.persist_directory = persist_directory
        self.db = None

        try:
            logger.info(
                f"Loading embedding model: {embedding_model}"
            )

            self.embedding_model = HuggingFaceEmbeddings(
                model_name=embedding_model
            )

            logger.info("Embedding model loaded successfully")

        except Exception as e:

            logger.error(
                f"Error loading embedding model: {e}"
            )

            raise

    # create vector database
    def create(self, chunks):

        if len(chunks) == 0:
            raise ValueError("Chunks list is empty")

        try:
            logger.info(
                f"Creating vector store with {len(chunks)} chunks"
            )

            self.db = Chroma.from_documents(
                documents=chunks,
                embedding=self.embedding_model,
                persist_directory=self.persist_directory,
            )

            logger.info(
                f"Vector store saved in: {self.persist_directory}"
            )

            return self.db

        except Exception as e:

            logger.error(
                f"Error creating vector store: {e}"
            )

            raise

    # load existing vector database
    def load(self):

        try:
            logger.info(
                f"Loading vector store from: {self.persist_directory}"
            )

            self.db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_model,
            )

            logger.info("Vector store loaded successfully")

            return self.db

        except Exception as e:

            logger.error(
                f"Error loading vector store: {e}"
            )

            raise


# compatibility functions
def create_vector_store(chunks):

    vector_store = VectorStore()

    return vector_store.create(chunks)


def load_vector_store():

    vector_store = VectorStore()

    return vector_store.load()