"""
Text Splitter
Splits documents into smaller chunks
"""

import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP

# logger
logger = logging.getLogger(__name__)


class TextSplitter:

    def __init__(
        self,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    ):

        # validation
        if chunk_size <= 0:
            raise ValueError("Chunk size must be greater than 0")

        if chunk_overlap < 0:
            raise ValueError("Chunk overlap cannot be negative")

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "Chunk overlap must be smaller than chunk size"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        try:
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )

            logger.info(
                f"Text splitter initialized "
                f"(chunk_size={chunk_size}, "
                f"chunk_overlap={chunk_overlap})"
            )

        except Exception as e:

            logger.error(
                f"Error initializing text splitter: {e}"
            )

            raise

    # split documents into chunks
    def split(self, documents):

        # check documents empty or not
        if len(documents) == 0:

            logger.warning("Documents list is empty")

            return []

        try:
            logger.info(
                f"Splitting {len(documents)} documents"
            )

            chunks = self.splitter.split_documents(
                documents
            )

            logger.info(
                f"Created {len(chunks)} chunks"
            )

            return chunks

        except Exception as e:

            logger.error(
                f"Error splitting documents: {e}"
            )

            raise


# compatibility function
def split_documents(documents):

    splitter = TextSplitter()

    return splitter.split(documents)