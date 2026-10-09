"""The medicines catalogue in Qdrant: one point for each medicine name.

Each point has the medicine itself as its payload, and up to two vectors:

``name``   sparse: the letters and the sound of the name (app/medicines/text.py). Always
           there. No model is needed to make one or to search by one.
``dense``  the name as a bi-encoder embeds it (app/medicines/encoders.py). There when the
           catalogue was indexed with a bi-encoder.

A name is searched for by both, and the two answers are not blended. Measured on the
whole catalogue, the sparse leg finds a misheard name far more often than a bi-encoder
does, and fusing the two by rank pushed right answers down. So the sparse leg's names
come first and the dense leg only adds the ones it alone found.
"""

from __future__ import annotations

import uuid
import warnings
from collections.abc import Callable, Sequence

from qdrant_client import AsyncQdrantClient, models

from app.core.logging import get_logger
from app.medicines.catalog import Medicine
from app.medicines.text import features

log = get_logger(__name__)

SPARSE = "name"
DENSE = "dense"
# A medicine id always maps to the same point, so indexing again replaces, never doubles.
_POINT_NAMESPACE = uuid.UUID("6f0d3c1e-5b7a-4c1e-9a55-0c11e7a00002")
_UPSERT_BATCH = 1000
_UPSERT_BATCH_DENSE = 500  # each point carries a few kilobytes of vector

ProgressFn = Callable[[int, int], None]


class NoDenseVectorsError(RuntimeError):
    """The collection was made without dense vectors, and they cannot be added to it."""


def point_id(medicine_id: str) -> str:
    return str(uuid.uuid5(_POINT_NAMESPACE, medicine_id))


def _vector(text: str) -> models.SparseVector:
    found = features(text)
    return models.SparseVector(indices=list(found), values=list(found.values()))


class MedicineStore:
    def __init__(self, client: AsyncQdrantClient, collection: str) -> None:
        self.client = client
        self.collection = collection
        self._dense: bool | None = None  # not asked yet

    async def exists(self) -> bool:
        return await self.client.collection_exists(self.collection)

    async def has_dense(self) -> bool:
        """Whether the names in this collection carry a bi-encoder's vector."""
        if self._dense is None:
            info = await self.client.get_collection(self.collection)
            vectors = info.config.params.vectors
            self._dense = isinstance(vectors, dict) and DENSE in vectors
        return self._dense

    async def ensure_collection(
        self, *, recreate: bool = False, dimension: int | None = None
    ) -> None:
        """Make the collection if it is not there. ``dimension`` adds the dense vector."""
        if await self.exists():
            if not recreate:
                if dimension and not await self.has_dense():
                    raise NoDenseVectorsError(
                        f"'{self.collection}' has no dense vectors. Index it again with --recreate."
                    )
                return
            await self.client.delete_collection(self.collection)
        dense: dict[str, models.VectorParams] = {}
        if dimension:
            # The full vectors stay on disk and a one-byte copy of each is searched in memory:
            # a quarter of a million 384-number vectors is 388 MB as written, 97 MB like this.
            dense[DENSE] = models.VectorParams(
                size=dimension,
                distance=models.Distance.COSINE,
                on_disk=True,
                quantization_config=models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8, quantile=0.99, always_ram=True
                    )
                ),
            )
        await self.client.create_collection(
            self.collection,
            vectors_config=dense,
            # IDF: Qdrant weighs each letter group by how rare it is in the catalogue.
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        self._dense = bool(dimension)
        with warnings.catch_warnings():
            # The embedded local index has no payload indexes (filters still work, unindexed).
            warnings.filterwarnings("ignore", message="Payload indexes have no effect")
            await self.client.create_payload_index(
                self.collection, "source", field_schema=models.PayloadSchemaType.KEYWORD
            )
        log.info("medicines.collection_created", collection=self.collection, dense=dimension)

    async def count(self) -> int:
        return (await self.client.count(self.collection, exact=True)).count

    async def upsert(
        self,
        medicines: Sequence[Medicine],
        *,
        dense: Sequence[Sequence[float]] | None = None,
        progress: ProgressFn | None = None,
    ) -> None:
        """Write medicines, with ``dense`` holding one bi-encoder vector for each if given."""
        if dense is not None and len(dense) != len(medicines):
            raise ValueError("medicines and dense vectors must be the same length")
        size = _UPSERT_BATCH if dense is None else _UPSERT_BATCH_DENSE
        for start in range(0, len(medicines), size):
            points = []
            for offset, medicine in enumerate(medicines[start : start + size]):
                vector: dict[str, models.SparseVector | list[float]] = {
                    SPARSE: _vector(medicine.name)
                }
                if dense is not None:
                    vector[DENSE] = list(dense[start + offset])
                points.append(
                    models.PointStruct(
                        id=point_id(medicine.id),
                        vector=vector,
                        payload=medicine.model_dump(exclude_none=True),
                    )
                )
            await self.client.upsert(self.collection, points=points)
            if progress:
                progress(min(start + size, len(medicines)), len(medicines))

    async def search(
        self,
        heard: str,
        *,
        limit: int = 64,
        dense: Sequence[float] | None = None,
        dense_limit: int = 16,
    ) -> list[Medicine]:
        """The catalogue names closest to what was heard, nearest first.

        ``dense`` is what was heard as the bi-encoder embeds it. Its nearest names are
        asked for in the same request, and those the sparse leg did not find are added
        after the sparse leg's own.
        """
        vector = _vector(heard)
        requests: list[models.QueryRequest] = []
        if vector.indices:
            requests.append(
                models.QueryRequest(query=vector, using=SPARSE, limit=limit, with_payload=True)
            )
        if dense is not None and dense_limit > 0:
            requests.append(
                models.QueryRequest(
                    query=list(dense), using=DENSE, limit=dense_limit, with_payload=True
                )
            )
        if not requests:
            return []
        responses = await self.client.query_batch_points(self.collection, requests=requests)
        found: dict[str, Medicine] = {}
        for response in responses:
            for point in response.points:
                if point.payload and str(point.id) not in found:
                    found[str(point.id)] = Medicine.model_validate(point.payload)
        return list(found.values())
