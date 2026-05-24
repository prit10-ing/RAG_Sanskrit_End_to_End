
"""
PDF Document Loader
Loads all PDF files from a folder using PyPDFLoader
"""

import os
import logging
from langchain_community.document_loaders import PyPDFLoader

# logger
logger = logging.getLogger(__name__)


class DocumentLoader:

    def __init__(self, folder_path):

        if not folder_path:
            raise ValueError("Folder path cannot be empty")

        self.folder_path = folder_path

    def load(self):

        documents = []

        # check folder exists or not
        if not os.path.exists(self.folder_path):
            raise FileNotFoundError(
                f"Folder not found: {self.folder_path}"
            )

        # check path is folder or not
        if not os.path.isdir(self.folder_path):
            raise NotADirectoryError(
                f"This is not a folder: {self.folder_path}"
            )

        # get all pdf files
        pdf_files = [
            file for file in os.listdir(self.folder_path)
            if file.endswith(".pdf")
        ]

        logger.info(f"Found {len(pdf_files)} PDF files")

        # if no pdf found
        if len(pdf_files) == 0:
            logger.warning("No PDF files found")
            return []

        # load all pdf files
        for file in pdf_files:

            file_path = os.path.join(self.folder_path, file)

            try:
                logger.info(f"Loading file: {file}")

                loader = PyPDFLoader(file_path)

                docs = loader.load()

                documents.extend(docs)

                logger.info(
                    f"Loaded {len(docs)} pages from {file}"
                )

            except Exception as e:

                logger.error(
                    f"Error loading {file}: {e}"
                )

        logger.info(
            f"Total documents loaded: {len(documents)}"
        )

        return documents


# function for compatibility
def load_documents(folder_path):

    loader = DocumentLoader(folder_path)

    return loader.load()