"""
Vector store: wraps ChromaDB (local, persistent, no external account needed)
for storing and retrieving SOP documents by semantic similarity.

Design choice worth documenting in your README: Chroma computes cosine
similarity internally when the collection is configured with
`hnsw:space: cosine` (its default is actually L2 — we set this explicitly
since my spec calls out cosine similarity specifically, and it materially
changes ranking behavior for text embeddings, which are typically compared
by direction rather than magnitude).
"""

from __future__ import annotations

import logging

import chromadb

from knowledge.embedder import EmbeddingProvider
from schemas.incident_schema import RetrievedSOP

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        persist_dir: str,
        collection_name: str = "security_sops",
    ):
        self.embedding_provider = embedding_provider
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def index_sops(self, sop_records: list[dict]) -> None:
        """
        sop_records: list of {"sop_id": str, "title": str, "text": str}
        Re-indexing is idempotent — Chroma upserts by ID.
        """
        ids = [r["sop_id"] for r in sop_records]
        documents = [r["text"] for r in sop_records]
        metadatas = [{"title": r["title"], "sop_id": r["sop_id"]} for r in sop_records]

        embeddings = self.embedding_provider.embed(documents)

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info("Indexed %d SOP documents", len(sop_records))

    def query(self, incident_summary: str, top_k: int = 3) -> list[RetrievedSOP]:
        query_embedding = self.embedding_provider.embed([incident_summary])[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        retrieved: list[RetrievedSOP] = []
        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]  # cosine DISTANCE (0=identical, 2=opposite)

        for doc_id, doc_text, meta, distance in zip(ids, documents, metadatas, distances):
            similarity_score = 1.0 - distance  # convert distance -> similarity for readability
            retrieved.append(
                RetrievedSOP(
                    sop_id=meta["sop_id"],
                    title=meta["title"],
                    text=doc_text,
                    similarity_score=round(similarity_score, 4),
                )
            )

        return retrieved

    def count(self) -> int:
        return self.collection.count()