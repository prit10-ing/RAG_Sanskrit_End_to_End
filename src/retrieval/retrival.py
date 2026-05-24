"""
Retriever - Final working version
"""

import logging
from src.config import RETRIEVAL_K, RETRIEVAL_SEARCH_TYPE

logger = logging.getLogger(__name__)


class Retriever:

    def __init__(self, vector_db, search_type=RETRIEVAL_SEARCH_TYPE, k=RETRIEVAL_K):

        if vector_db is None:
            raise ValueError("Vector database cannot be None")

        if k <= 0:
            raise ValueError("k must be greater than 0")

        self.vector_db = vector_db
        self.search_type = search_type
        self.k = k

    def get_retriever(self):

        try:
            logger.info(f"Creating retriever (search_type={self.search_type}, k={self.k})")

            # MMR requires fetch_k; similarity just needs k
            if self.search_type == "mmr":
                search_kwargs = {"k": self.k, "fetch_k": self.k * 4}
            else:
                search_kwargs = {"k": self.k}

            retriever = self.vector_db.as_retriever(
                search_type=self.search_type,
                search_kwargs=search_kwargs,
            )

            logger.info("Retriever created successfully")
            return retriever

        except Exception as e:
            logger.error(f"Error creating retriever: {e}")
            raise


def create_retriever(vector_db):
    return Retriever(vector_db).get_retriever()