"""
Embedding provider abstraction. Kept separate from vector_store.py so the
embedding backend (Ollama locally, or a cloud provider later) can be swapped
without touching retrieval logic — this is the same seam pattern used for
the VLM provider in ingestion/multimodal_detector.py.
"""

from __future__ import annotations

import logging
from typing import Protocol

import requests

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text, same order."""
        ...


class OllamaEmbeddingProvider:
    """Real embedding provider — uses nomic-embed-text served locally via
    Ollama. This is the path you'll actually use once `ollama pull
    nomic-embed-text` has completed."""

    def __init__(self, base_url: str, model: str, timeout_s: int = 60):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=self.timeout_s,
            )
            response.raise_for_status()
            vectors.append(response.json()["embedding"])
        return vectors