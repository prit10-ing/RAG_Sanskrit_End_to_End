"""
llm/llm_model.py
=================
HuggingFace LLM wrapped in a LangChain chat interface.
"""

import logging
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from src.config import (
    LLM_REPO_ID,
    LLM_TASK,
    LLM_MAX_NEW_TOKENS,
    LLM_TEMPERATURE,
)

load_dotenv()
logger = logging.getLogger(__name__)


class LLMModel:
    """
    Wraps HuggingFaceEndpoint + ChatHuggingFace.
    The model is loaded lazily on the first call to generate_answer().
    """

    def __init__(
        self,
        repo_id: str = LLM_REPO_ID,
        task: str = LLM_TASK,
        max_new_tokens: int = LLM_MAX_NEW_TOKENS,
        temperature: float = LLM_TEMPERATURE,
    ):
        self.repo_id = repo_id
        self.task = task
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._model: ChatHuggingFace | None = None
        self._parser = StrOutputParser()

    # ── public ─────────────────────────────────────────────────────────────────

    def generate_answer(self, prompt: str) -> str:
        """Run the chain and return a plain-text answer."""
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must not be empty.")

        logger.info("Sending prompt to LLM (length=%d chars).", len(prompt))
        try:
            model = self._get_model()
            chain = model | self._parser
            result = chain.invoke(prompt)
            logger.info("LLM response received.")
            return result
        except Exception as e:
            logger.error("LLM generation failed: %s", e)
            raise

    # ── private ────────────────────────────────────────────────────────────────

    def _get_model(self) -> ChatHuggingFace:
        """Lazy-load model on first use."""
        if self._model is None:
            logger.info("Initialising LLM endpoint: %s", self.repo_id)
            try:
                llm = HuggingFaceEndpoint(
                    repo_id=self.repo_id,
                    task=self.task,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                )
                self._model = ChatHuggingFace(llm=llm)
                logger.info("LLM ready.")
            except Exception as e:
                logger.error("Failed to initialise LLM: %s", e)
                raise
        return self._model


# ── module-level helpers ───────────────────────────────────────────────────────

_default_llm = LLMModel()


def genrate_answer(promt: str) -> str:
    """Original function — preserved spelling and signature."""
    return _default_llm.generate_answer(promt)