"""Local embedding model (sentence-transformers) behind a small async interface.

Embeddings are computed locally so a phone call never waits on a network round
trip to an embedding API, and no patient text leaves the machine for retrieval.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from app.core.logging import get_logger

log = get_logger(__name__)


class Embedder(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def detect_device(preferred: str | None = None) -> str:
    if preferred:
        return preferred
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class SentenceTransformerEmbedder:
    """Cosine-normalised embeddings; BGE-style instruction applied to queries only."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str | None = None,
        batch_size: int = 32,
        query_instruction: str = "",
        model_factory: Callable[[str, str], Any] | None = None,
    ) -> None:
        self.model_name = model_name
        self._batch_size = batch_size
        self._query_instruction = query_instruction
        self._device = detect_device(device) if model_factory is None else (device or "cpu")
        self._model_factory = model_factory or self._default_factory
        self._model: Any | None = None
        self._dimension: int | None = None

    @staticmethod
    def _default_factory(model_name: str, device: str) -> Any:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name, device=device)

    @property
    def model(self) -> Any:
        if self._model is None:
            log.info("embedder.loading", model=self.model_name, device=self._device)
            self._model = self._model_factory(self.model_name, self._device)
        return self._model

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            dim = self.model.get_sentence_embedding_dimension()
            if dim is None:  # pragma: no cover - defensive; all supported models report it
                dim = len(self.embed_query("dimension probe"))
            self._dimension = int(dim)
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode(
            self._query_instruction + text,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [float(x) for x in vector.tolist()]


async def aembed_documents(embedder: Embedder, texts: Sequence[str]) -> list[list[float]]:
    """Run the (CPU/GPU-bound) encode off the event loop."""
    return await asyncio.to_thread(embedder.embed_documents, texts)


async def aembed_query(embedder: Embedder, text: str) -> list[float]:
    return await asyncio.to_thread(embedder.embed_query, text)
