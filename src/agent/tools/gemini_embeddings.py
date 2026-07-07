"""Gemini embedding adapter for LangChain vector stores."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import List

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-embedding-001"
DEFAULT_OUTPUT_DIM = 3072
BATCH_SIZE = int(os.environ.get("GEMINI_EMBED_BATCH", "32"))
_RETRY_EXC = (genai_errors.APIError, ConnectionError, TimeoutError)


class GeminiEmbeddings(Embeddings):
    """LangChain Embeddings implementation for Google AI Studio Gemini."""

    def __init__(self, model: str | None = None, output_dim: int = DEFAULT_OUTPUT_DIM):
        api_key = os.environ["GEMINI_API_KEY"]
        self.model = model or os.environ.get("EMBEDDING_MODEL", DEFAULT_MODEL)
        self.output_dim = output_dim
        self._client = genai.Client(api_key=api_key)

    def _embed_batch_once(self, texts: List[str], task_type: str) -> List[List[float]]:
        resp = self._client.models.embed_content(
            model=self.model,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=self.output_dim,
            ),
        )
        return [e.values for e in resp.embeddings]

    async def _aembed_batch_once(self, texts: List[str], task_type: str) -> List[List[float]]:
        resp = await self._client.aio.models.embed_content(
            model=self.model,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=self.output_dim,
            ),
        )
        return [e.values for e in resp.embeddings]

    def _embed_batch(self, texts: List[str], task_type: str) -> List[List[float]]:
        delay = 2.0
        for attempt in range(6):
            try:
                return self._embed_batch_once(texts, task_type)
            except _RETRY_EXC:
                if attempt == 5:
                    raise
                logger.warning("Gemini embedding request failed; retrying.")
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
        return []

    async def _aembed_batch(self, texts: List[str], task_type: str) -> List[List[float]]:
        delay = 2.0
        for attempt in range(6):
            try:
                return await self._aembed_batch_once(texts, task_type)
            except _RETRY_EXC:
                if attempt == 5:
                    raise
                logger.warning("Gemini embedding request failed; retrying.")
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30.0)
        return []

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            out.extend(self._embed_batch(texts[i:i + BATCH_SIZE], "RETRIEVAL_DOCUMENT"))
        return out

    def embed_query(self, text: str) -> List[float]:
        return self._embed_batch([text], "RETRIEVAL_QUERY")[0]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            out.extend(await self._aembed_batch(texts[i:i + BATCH_SIZE], "RETRIEVAL_DOCUMENT"))
        return out

    async def aembed_query(self, text: str) -> List[float]:
        return (await self._aembed_batch([text], "RETRIEVAL_QUERY"))[0]
