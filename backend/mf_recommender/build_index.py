"""Build (or rebuild) the FAISS index:  python -m mf_recommender.build_index [--force]"""

from __future__ import annotations

import argparse
import logging

from .config import Settings
from .data import fund_documents, load_funds
from .embeddings import build_embeddings
from .vectorstore import load_or_build_vectorstore


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Rebuild even if the saved index is current")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    settings = Settings.from_env()
    docs = fund_documents(load_funds(settings.data_path))
    store = load_or_build_vectorstore(
        docs, build_embeddings(settings.embedding_model), settings.index_dir, settings.embedding_model,
        force_rebuild=args.force,
    )
    print(f"Index ready: {store.index.ntotal} funds -> {settings.index_dir}")


if __name__ == "__main__":
    main()
