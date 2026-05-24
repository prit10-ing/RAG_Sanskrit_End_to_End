# """
# data_ingestion/load_documents.py
# =================================
# Loads PDF files from a folder using PyPDFLoader.
# Your original logic is preserved exactly inside DocumentLoader.load().
# """

# import logging
# import os
# from langchain_community.document_loaders.pdf import PyPDFLoader
# from langchain_core.documents import Document

# logger = logging.getLogger(__name__)


# class DocumentLoader:
#     """Loads all PDF documents from a given folder."""

#     def __init__(self, folder_path: str):
#         if not folder_path:
#             raise ValueError("folder_path must not be empty.")
#         self.folder_path = folder_path

#     # ── public ─────────────────────────────────────────────────────────────────

#     def load(self) -> list[Document]:
#         """
#         Walk folder_path and load every .pdf file found.
#         Returns a flat list of LangChain Document objects.
#         """
#         self._validate_folder()
#         pdf_files = self._find_pdfs()

#         if not pdf_files:
#             logger.warning("No PDF files found in '%s'.", self.folder_path)
#             return []

#         documents = []
#         for file in pdf_files:
#             docs = self._load_single(file)
#             documents.extend(docs)

#         logger.info(
#             "Loaded %d page(s) from %d PDF file(s).",
#             len(documents),
#             len(pdf_files),
#         )
#         return documents

#     # ── private ────────────────────────────────────────────────────────────────

#     def _validate_folder(self) -> None:
#         if not os.path.exists(self.folder_path):
#             raise FileNotFoundError(
#                 f"Data folder not found: '{self.folder_path}'"
#             )
#         if not os.path.isdir(self.folder_path):
#             raise NotADirectoryError(
#                 f"Path is not a directory: '{self.folder_path}'"
#             )

#     def _find_pdfs(self) -> list[str]:
#         """Return sorted list of .pdf filenames inside folder_path."""
#         files = sorted(
#             f for f in os.listdir(self.folder_path) if f.lower().endswith(".pdf")
#         )
#         logger.info("Found %d PDF file(s) in '%s'.", len(files), self.folder_path)
#         return files

#     def _load_single(self, filename: str) -> list[Document]:
#         """Load one PDF file; skip on error so one bad file won't crash the run."""
#         file_path = os.path.join(self.folder_path, filename)
#         logger.debug("Loading: %s", file_path)
#         try:
#             loader = PyPDFLoader(file_path)
#             pdf_docs = loader.load()
#             logger.debug("  → %d page(s) from '%s'.", len(pdf_docs), filename)
#             return pdf_docs
#         except Exception as e:
#             logger.error("Failed to load '%s': %s — skipping.", filename, e)
#             return []


# # ── module-level convenience function (mirrors original signature) ──────────────

# def load_documents(folder_path: str) -> list[Document]:
#     """Original function signature preserved for compatibility."""
#     return DocumentLoader(folder_path).load()




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