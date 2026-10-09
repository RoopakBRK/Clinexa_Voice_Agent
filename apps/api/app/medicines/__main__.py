"""Builds and tries the medicines catalogue.

    python -m app.medicines build                  read the three files, print what was found
    python -m app.medicines index [--recreate]     put the catalogue in Qdrant
    python -m app.medicines find "glycomate 500"   what Clinexa finds for a name

``index`` and ``find`` use the Qdrant in QDRANT_URL. With ``--local`` they use the embedded
index under data/indexes instead, which is for trying things out: it reads every name for
every lookup, so pass ``--limit`` to keep it small.

Where sentence-transformers is installed, ``index`` also embeds every name with the
bi-encoder (about a minute on a laptop) and ``find`` uses both encoders, as the server
would. ``--no-encoders`` leaves them out.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections import Counter

from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.medicines.catalog import SOURCE_NAMES, Medicine, load_catalog
from app.medicines.encoders import NameEncoders, build_encoders
from app.medicines.lookup import MedicineLookup
from app.medicines.store import MedicineStore
from app.rag.retrieval.qdrant_store import build_client


def _catalog(limit: int | None) -> list[Medicine]:
    started = time.perf_counter()
    medicines = load_catalog(get_settings().data_dir)
    by_source = Counter(medicine.source for medicine in medicines)
    print(f"Read {len(medicines):,} medicines in {time.perf_counter() - started:.0f}s:")
    for source, name in SOURCE_NAMES.items():
        print(f"  {by_source[source]:>8,}  {name}")
    return medicines[:limit] if limit else medicines


def _store(local: bool) -> MedicineStore:
    settings = get_settings()
    if not local and not settings.qdrant_url:
        sys.exit("QDRANT_URL is not set. Set it, or pass --local to use the embedded index.")
    return MedicineStore(build_client(settings, force_local=local), settings.medicines_collection)


def _encoders(args: argparse.Namespace) -> NameEncoders | None:
    return None if args.no_encoders else build_encoders(get_settings())


# How many names are embedded and written at a time.
_EMBED_AT_ONCE = 5000


async def _index(args: argparse.Namespace) -> None:
    store = _store(args.local)
    await store.exists()  # fail now, not after reading the files, if Qdrant cannot be reached
    medicines = _catalog(args.limit)
    if not medicines:
        # Nothing to put in. Above all, do not empty a collection that is in use.
        sys.exit(f"No medicines found in {get_settings().data_dir}. Nothing was changed.")
    encoders = _encoders(args)
    if encoders:
        print(f"Bi-encoder: {encoders.bi_encoder} ({encoders.dimension} dimensions)")
    else:
        print("Bi-encoder: none. Names will be found by spelling and sound alone.")
    await store.ensure_collection(
        recreate=args.recreate, dimension=encoders.dimension if encoders else None
    )
    started = time.perf_counter()
    embedding = 0.0

    def progress(done: int, total: int) -> None:
        if done % 20_000 == 0 or done == total:
            print(f"  {done:>8,} of {total:,}", flush=True)

    if encoders is None:
        await store.upsert(medicines, progress=progress)
    else:
        for at in range(0, len(medicines), _EMBED_AT_ONCE):
            batch = medicines[at : at + _EMBED_AT_ONCE]
            began = time.perf_counter()
            vectors = await asyncio.to_thread(encoders.embed_names, [m.name for m in batch])
            embedding += time.perf_counter() - began
            await store.upsert(batch, dense=vectors)
            progress(min(at + _EMBED_AT_ONCE, len(medicines)), len(medicines))
    took = time.perf_counter() - started
    print(
        f"'{store.collection}' now holds {await store.count():,} medicines "
        f"({took:.0f}s" + (f", {embedding:.0f}s of it embedding" if encoders else "") + ")."
    )
    await store.client.close()


async def _find(args: argparse.Namespace) -> None:
    store = _store(args.local)
    if not await store.exists():
        sys.exit(f"There is no '{store.collection}' collection yet. Run: make medicines")
    encoders = _encoders(args)
    if encoders:
        await asyncio.to_thread(encoders.warm)
        dense = "and its dense vectors" if await store.has_dense() else "(no dense vectors stored)"
        print(
            f"Using {encoders.bi_encoder} {dense}, and {encoders.cross_encoder or 'no cross-encoder'}."
        )
    # No hurry here: show what the lookup finds, however long the first call takes.
    lookup = MedicineLookup(store, encoders=encoders, timeout_s=30.0)
    for heard in args.names:
        started = time.perf_counter()
        match = await lookup.find(heard)
        took = (time.perf_counter() - started) * 1000
        if match is None:
            sys.exit(f'"{heard}": the catalogue could not be asked.')
        print(f'"{heard}" -> {match.status} ({took:.0f} ms)')
        for field in ("name", "strength", "unit", "composition", "source"):
            if (value := getattr(match, field)) is not None:
                print(f"    {field}: {value}")
        if match.choices:
            print(f"    choices: {'; '.join(match.choices)}")
    await lookup.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m app.medicines", description=__doc__.split("\n")[0]
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build", help="read the three files and print what was found")
    index = commands.add_parser("index", help="put the catalogue in Qdrant")
    index.add_argument("--recreate", action="store_true", help="empty the collection first")
    find = commands.add_parser("find", help="what Clinexa finds for a name")
    find.add_argument("names", nargs="+")
    for command in (index, find):
        command.add_argument("--local", action="store_true", help="use the embedded index")
        command.add_argument(
            "--no-encoders",
            action="store_true",
            help="leave the bi-encoder and the cross-encoder out",
        )
    index.add_argument("--limit", type=int, help="index only the first N medicines")
    args = parser.parse_args()

    configure_logging("WARNING", json_logs=False)
    if args.command == "build":
        _catalog(None)
        return
    try:
        asyncio.run(_index(args) if args.command == "index" else _find(args))
    except (ResponseHandlingException, UnexpectedResponse, OSError) as exc:
        host = (get_settings().qdrant_url or "").split("//")[-1]
        sys.exit(
            f"Could not reach Qdrant at {host or 'the embedded index'}: {type(exc).__name__}.\n"
            "  On Qdrant Cloud, open cloud.qdrant.io and check the cluster is Running: one that is\n"
            "  suspended or deleted resets connections. Then check QDRANT_URL and QDRANT_API_KEY."
        )


if __name__ == "__main__":
    main()
