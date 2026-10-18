"""Build, persist and reload the LangChain FAISS vector store."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

log = logging.getLogger(__name__)
MANIFEST = "manifest.json"


def corpus_fingerprint(docs: list[Document], embedding_model: str) -> str:
    h = hashlib.sha256(embedding_model.encode())
    for doc in docs:
        h.update(f"{doc.id}\x1f{doc.page_content}\x1e".encode())
    return h.hexdigest()


def build_vectorstore(docs: list[Document], embeddings: Embeddings) -> FAISS:
    # Embeddings are unit length, so maximum inner product == cosine similarity
    # (higher is better). The old notebook used raw L2 distance and *added* it to the
    # score, which rewarded the least similar funds.
    return FAISS.from_documents(docs, embeddings, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)


def load_or_build_vectorstore(
    docs: list[Document],
    embeddings: Embeddings,
    index_dir: str | Path,
    embedding_model: str,
    *,
    force_rebuild: bool = False,
) -> FAISS:
    """Reuse the saved index if it was built from the same documents and model, else rebuild."""
    index_dir = Path(index_dir)
    fingerprint = corpus_fingerprint(docs, embedding_model)
    manifest_path = index_dir / MANIFEST

    if not force_rebuild and manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("fingerprint") == fingerprint:
                log.info("Loading FAISS index from %s", index_dir)
                # The docstore is pickled; only load indexes this project wrote itself.
                return FAISS.load_local(
                    str(index_dir),
                    embeddings,
                    allow_dangerous_deserialization=True,
                    distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
                )
            log.info("Fund data or embedding model changed; rebuilding the index")
        except Exception:  # corrupt or partial index: rebuild
            log.exception("Could not load the saved index; rebuilding")

    log.info("Building FAISS index for %d funds with %s", len(docs), embedding_model)
    store = build_vectorstore(docs, embeddings)
    index_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(index_dir))
    manifest_path.write_text(
        json.dumps({"fingerprint": fingerprint, "embedding_model": embedding_model, "documents": len(docs)}, indent=2)
    )
    return store
