"""The bi-encoder and the cross-encoder of the medicines catalogue.

Which models, and what they are for, were settled by measuring them on the whole
catalogue (docs/rag.md, "Medicines catalogue"). In short:

* A bi-encoder embeds a name by its meaning, and a misheard brand name has none. On names
  said correctly it finds what the spelling-and-sound search finds. On misheard ones it
  finds far fewer. So it is a second leg: it adds the few names it alone finds.
  ``BAAI/bge-small-en-v1.5`` did better than the two medical bi-encoders tried.
* A cross-encoder, put in charge of the order, ranked misheard names worse than the
  search does. So it only breaks ties among names the rules already hold equal: which
  of a brand's products are listed first.

Neither decides which medicine a caller is told about. app/medicines/lookup.py does, by
its rules.

Both models run through sentence-transformers. Where that is not installed
``build_encoders`` returns None and the lookup works by spelling and sound alone.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence

from app.core.config import Settings
from app.core.logging import get_logger
from app.medicines.catalog import Medicine
from app.rag.embeddings import Embedder, SentenceTransformerEmbedder
from app.rag.reranking.cross_encoder import CrossEncoderReranker, Reranker

log = get_logger(__name__)

# Names are a few words long. Large batches are what make a quarter of a million quick.
_EMBED_BATCH = 256
_RERANK_BATCH = 128
_RERANK_MAX_TOKENS = 64


def passage(medicine: Medicine) -> str:
    """What the cross-encoder reads for a medicine: its name, and what is in it."""
    if medicine.composition and medicine.composition != medicine.name:
        return f"{medicine.name} ({medicine.composition})"
    return medicine.name


class NameEncoders:
    def __init__(self, embedder: Embedder, reranker: Reranker | None = None) -> None:
        self._embedder = embedder
        self._reranker = reranker
        # Loading a model takes seconds. Until both are in memory the lookup does without.
        self.ready = False

    @property
    def bi_encoder(self) -> str:
        return self._embedder.model_name

    @property
    def cross_encoder(self) -> str | None:
        return self._reranker.model_name if self._reranker else None

    @property
    def dimension(self) -> int:
        return self._embedder.dimension

    def warm(self) -> None:
        """Load both models and run each once. Blocking: call it off the event loop."""
        self._embedder.embed_query("paracetamol")
        if self._reranker:
            self._reranker.score("paracetamol", ["Paracetamol Tablet 500 mg"])
        self.ready = True
        log.info(
            "medicines.encoders_ready", bi_encoder=self.bi_encoder, cross_encoder=self.cross_encoder
        )

    def embed_names(self, names: Sequence[str]) -> list[list[float]]:
        """One vector for each catalogue name, as it is stored."""
        return self._embedder.embed_documents(names)

    def embed_heard(self, heard: str) -> list[float]:
        return self._embedder.embed_query(heard)

    def order(self, heard: str, medicines: Sequence[Medicine]) -> list[Medicine]:
        """The medicines, most relevant to what was heard first. Ties keep their order."""
        if not self._reranker or len(medicines) < 2:
            return list(medicines)
        scores = self._reranker.score(heard, [passage(m) for m in medicines])
        best_first = sorted(range(len(medicines)), key=lambda i: (-scores[i], i))
        return [medicines[i] for i in best_first]


def build_encoders(settings: Settings) -> NameEncoders | None:
    """The two models for this machine, or None where they are switched off or cannot run."""
    if not settings.medicines_encoders:
        return None
    if importlib.util.find_spec("sentence_transformers") is None:
        log.info("medicines.encoders_unavailable", reason="sentence-transformers is not installed")
        return None
    embedder = SentenceTransformerEmbedder(
        settings.medicines_bi_encoder,
        device=settings.embedding_device,
        batch_size=_EMBED_BATCH,
        query_instruction=settings.medicines_query_instruction,
    )
    reranker = (
        CrossEncoderReranker(
            settings.medicines_cross_encoder,
            device=settings.embedding_device,
            batch_size=_RERANK_BATCH,
            max_length=_RERANK_MAX_TOKENS,
        )
        if settings.medicines_cross_encoder
        else None
    )
    return NameEncoders(embedder, reranker)
