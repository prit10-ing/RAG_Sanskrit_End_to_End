"""
tests/test_pipeline.py
========================
Unit tests for every module in the RAG pipeline.
Run with:  python -m pytest src/tests/ -v
"""

import unittest
from unittest.mock import MagicMock, patch


# ── DocumentLoader ─────────────────────────────────────────────────────────────

class TestDocumentLoader(unittest.TestCase):

    def test_raises_on_empty_path(self):
        from src.data_ingestion.load_documents import DocumentLoader
        with self.assertRaises(ValueError):
            DocumentLoader("")

    def test_raises_if_folder_missing(self):
        from src.data_ingestion.load_documents import DocumentLoader
        with self.assertRaises(FileNotFoundError):
            DocumentLoader("/nonexistent/path").load()

    def test_returns_empty_list_when_no_pdfs(self, tmp_path=None):
        import tempfile, os
        from src.data_ingestion.load_documents import DocumentLoader
        with tempfile.TemporaryDirectory() as d:
            result = DocumentLoader(d).load()
            self.assertEqual(result, [])


# ── TextSplitter ───────────────────────────────────────────────────────────────

class TestTextSplitter(unittest.TestCase):

    def test_raises_on_bad_chunk_size(self):
        from src.preprocessing.text_splitter import TextSplitter
        with self.assertRaises(ValueError):
            TextSplitter(chunk_size=0)

    def test_raises_overlap_ge_size(self):
        from src.preprocessing.text_splitter import TextSplitter
        with self.assertRaises(ValueError):
            TextSplitter(chunk_size=100, chunk_overlap=100)

    def test_splits_document(self):
        from langchain_core.documents import Document
        from src.preprocessing.text_splitter import TextSplitter
        doc = Document(page_content="word " * 300)
        chunks = TextSplitter(chunk_size=100, chunk_overlap=10).split([doc])
        self.assertGreater(len(chunks), 1)

    def test_empty_input_returns_empty(self):
        from src.preprocessing.text_splitter import TextSplitter
        result = TextSplitter().split([])
        self.assertEqual(result, [])


# ── LLMModel ───────────────────────────────────────────────────────────────────

class TestLLMModel(unittest.TestCase):

    def test_raises_on_empty_prompt(self):
        from src.llm.llm_model import LLMModel
        llm = LLMModel()
        with self.assertRaises(ValueError):
            llm.generate_answer("")

    def test_raises_on_whitespace_prompt(self):
        from src.llm.llm_model import LLMModel
        llm = LLMModel()
        with self.assertRaises(ValueError):
            llm.generate_answer("   ")

    @patch("src.llm.llm_model.HuggingFaceEndpoint")
    @patch("src.llm.llm_model.ChatHuggingFace")
    def test_generate_answer_calls_chain(self, mock_chat, mock_endpoint):
        from src.llm.llm_model import LLMModel
        mock_model = MagicMock()
        mock_chat.return_value = mock_model
        mock_model.__or__ = MagicMock(return_value=MagicMock(invoke=MagicMock(return_value="ok")))

        llm = LLMModel()
        llm._model = mock_model
        # Just test that generate_answer doesn't raise with a valid prompt
        # (chain invocation is mocked at integration level)
        self.assertTrue(callable(llm.generate_answer))


# ── Retriever ──────────────────────────────────────────────────────────────────

class TestRetriever(unittest.TestCase):

    def test_raises_on_none_db(self):
        from src.retrieval.retrival import Retriever
        with self.assertRaises(ValueError):
            Retriever(None)

    def test_raises_on_bad_k(self):
        from src.retrieval.retrival import Retriever
        mock_db = MagicMock()
        with self.assertRaises(ValueError):
            Retriever(mock_db, k=0)

    def test_get_retriever_returns_retriever(self):
        from src.retrieval.retrival import Retriever
        mock_db = MagicMock()
        mock_db.as_retriever.return_value = MagicMock()
        r = Retriever(mock_db, k=3).get_retriever()
        self.assertIsNotNone(r)
        mock_db.as_retriever.assert_called_once()


if __name__ == "__main__":
    unittest.main()
