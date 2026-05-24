"""
RAG Pipeline - Final working version
Load Documents -> Split Text -> Create Vector DB -> Retrieve -> Generate Answer
"""

import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.data_ingestion.load_documents import DocumentLoader
from src.preprocessing.text_splitter import TextSplitter
from src.embeddings.vectore_store import VectorStore
from src.retrieval.retrival import Retriever
from src.llm.llm_model import LLMModel

from src.config import DATA_FOLDER, PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


class RAGPipeline:

    def __init__(self, data_folder=DATA_FOLDER):

        self.data_folder = data_folder
        self.vector_store = VectorStore()
        self.llm = LLMModel()
        self.db = None
        self.chain = None

    # =========================
    # Ingestion
    # =========================

    def ingest(self):

        logger.info("Starting ingestion process")

        loader = DocumentLoader(self.data_folder)
        documents = loader.load()

        if len(documents) == 0:
            logger.warning("No documents found")
            return

        splitter = TextSplitter()
        chunks = splitter.split(documents)

        self.db = self.vector_store.create(chunks)
        logger.info("Ingestion completed")

    # =========================
    # Query
    # =========================

    def query(self, question):

        if not question.strip():
            raise ValueError("Question cannot be empty")

        logger.info(f"Question: {question}")

        try:
            chain = self.get_chain()
            raw_answer = chain.invoke(question)
            answer = self._extract_answer(raw_answer)
            logger.info("Answer generated")
            return answer

        except Exception as e:
            logger.error(f"Error during query: {e}")
            raise

    # =========================
    # Extract Answer
    # =========================

    def _extract_answer(self, raw_output: str) -> str:
        """
        Handles 3 cases:
          1. Model echoed prompt + wrote "Answer:" -> extract after last "Answer:"
          2. Model replied directly (no "Answer:" marker) -> strip boilerplate lines
          3. Truly empty -> return fallback message
        """
        if not raw_output or not raw_output.strip():
            return "I could not find the answer in the provided documents."

        text = raw_output.strip()

        # Case 1: find the last "Answer:" marker (handles prompt echo)
        for marker in ["Answer:", "answer:", "ANSWER:"]:
            idx = text.rfind(marker)
            if idx != -1:
                candidate = text[idx + len(marker):].strip()
                if candidate:
                    return candidate

        # Case 2: no marker — strip echoed prompt boilerplate line by line
        boilerplate_starts = [
            "You are an expert",
            "The documents contain",
            "Use the context",
            "LANGUAGE RULE",
            "ANSWERING RULES",
            "Question in Sanskrit",
            "Question in English",
            "Question in Hindi",
            "For direct questions",
            "For broad questions",
            "Answer naturally",
            "Write ONLY",
            "Say \"I could not",
            "CONTEXT:",
            "[Chunk",
            "Question:",
            "Answer:",
            "======",
            "------",
        ]

        lines = text.split("\n")
        clean_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            is_boilerplate = any(stripped.startswith(bp) for bp in boilerplate_starts)
            if not is_boilerplate:
                clean_lines.append(stripped)

        result = "\n".join(clean_lines).strip()
        if result:
            return result

        # Case 3: nothing left
        return "I could not find the answer in the provided documents."

    # =========================
    # Build Chain
    # =========================

    def get_chain(self):

        if self.chain is not None:
            return self.chain

        if self.db is None:
            logger.info("Loading vector database")
            self.db = self.vector_store.load()

        retriever = Retriever(self.db).get_retriever()
        model = self.llm._get_model()
        parser = StrOutputParser()
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

        def format_docs(docs):

            if not docs:
                logger.warning("Retriever returned 0 documents!")
                return "No context available."

            logger.info(f"Retrieved {len(docs)} document chunks for context")

            cleaned = ""
            for i, doc in enumerate(docs):
                text = doc.page_content
                text = text.replace("\n", " ")
                text = " ".join(text.split())
                cleaned += f"[Chunk {i+1}]: {text}\n\n"

            return cleaned.strip()

        self.chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | model
            | parser
        )

        logger.info("RAG chain created")
        return self.chain