"""
One-off ingestion script: reads all SOP .txt files from data/sops/ and
indexes them into the vector store. Run this once after adding/changing
SOP documents (or whenever you want to rebuild the index from scratch).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from config import CONFIG, SOP_DIR
from knowledge.embedder import OllamaEmbeddingProvider
from knowledge.vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_sop_file(path: Path) -> dict:
    """Extracts SOP ID and Title from the header lines of the file, uses
    the full text as the document body for embedding."""
    text = path.read_text(encoding="utf-8")

    id_match = re.search(r"SOP ID:\s*(.+)", text)
    title_match = re.search(r"Title:\s*(.+)", text)

    if not id_match or not title_match:
        raise ValueError(f"SOP file {path.name} missing required 'SOP ID:' or 'Title:' header")

    return {
        "sop_id": id_match.group(1).strip(),
        "title": title_match.group(1).strip(),
        "text": text,
    }


def load_sop_records(sop_dir: Path) -> list[dict]:
    files = sorted(sop_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt SOP files found in {sop_dir}")
    return [parse_sop_file(f) for f in files]


def main():
    records = load_sop_records(SOP_DIR)
    logger.info("Loaded %d SOP files from %s", len(records), SOP_DIR)

    embedding_provider = OllamaEmbeddingProvider(
        base_url=CONFIG.vlm.ollama_base_url,
        model=CONFIG.vector_store.embedding_model,
    )
    store = VectorStore(
        embedding_provider=embedding_provider,
        persist_dir=CONFIG.vector_store.persist_dir,
        collection_name=CONFIG.vector_store.collection_name,
    )
    store.index_sops(records)
    logger.info("Vector store now contains %d documents", store.count())


if __name__ == "__main__":
    main()