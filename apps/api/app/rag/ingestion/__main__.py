"""CLI: python -m app.rag.ingestion [--docs ID ...] [--workers N] [--force-extract]"""

from __future__ import annotations

import argparse

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.rag.ingestion.pipeline import run_ingestion


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunk the WHO knowledge base into data/processed/"
    )
    parser.add_argument(
        "--docs", nargs="*", help="doc_ids from data/manifests/documents.yaml (default: all)"
    )
    parser.add_argument(
        "--workers", type=int, help="extraction processes (default: CPU count - 1, max 8)"
    )
    parser.add_argument("--force-extract", action="store_true", help="ignore the extraction cache")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level, json_logs=False)
    report = run_ingestion(
        settings, doc_ids=args.docs, workers=args.workers, force_extract=args.force_extract
    )
    print(
        f"\n{report.total_chunks} chunks ({report.total_retrievable} retrievable) in {report.duration_s}s"
    )
    for d in report.documents:
        print(
            f"  {d.doc_id:40} pages={d.pages:<4} chunks={d.chunks:<5} "
            f"tokens p50={d.tokens_p50:.0f} p95={d.tokens_p95:.0f} max={d.tokens_max}"
        )
    print(f"\nReport: {settings.data_dir / 'processed' / 'ingestion_report.md'}")
    print(f"Samples: {settings.data_dir / 'processed' / 'samples.md'}")


if __name__ == "__main__":
    main()
