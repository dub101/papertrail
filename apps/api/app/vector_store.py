from __future__ import annotations

import uuid

from typing import Any, Dict, List

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from fastembed import TextEmbedding


class VectorStore:
    """
    Minimal Qdrant-backed vector store for arXiv papers.

    - Embeds text (abstract/title) using fastembed
    - Upserts vectors + paper payload into Qdrant
    - Creates indexes for authors/categories for later filtering
    """

    def __init__(self, qdrant_url: str, collection: str):
        self.client = QdrantClient(url=qdrant_url)
        self.collection = collection

        # fastembed model (downloads on first use if not cached)
        self.embedding_model = TextEmbedding()

        # Determine vector size safely (fastembed returns an iterator in many versions)
        sample_vec = self.embed_texts(["sample"])[0]
        self.vector_size = len(sample_vec)

        # Ensure Qdrant collection + indexes exist
        self.ensure_collection()
    
    def _qid(self, arxiv_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, arxiv_id))


    def ensure_collection(self) -> None:
        """
        Create collection if missing (DO NOT recreate, to avoid wiping data).
        Also create payload indexes for authors/categories for filtering later.
        """
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=qm.VectorParams(
                    size=self.vector_size,
                    distance=qm.Distance.COSINE,
                ),
            )

        # Create payload indexes (safe to call repeatedly; ignore "already exists" errors)
        self._ensure_payload_index("authors", qm.PayloadSchemaType.KEYWORD)
        self._ensure_payload_index("categories", qm.PayloadSchemaType.KEYWORD)

    def _ensure_payload_index(self, field_name: str, field_type: qm.PayloadSchemaType) -> None:
        try:
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name=field_name,
                field_schema=qm.FieldSchema(type=field_type, array=True),
            )
        except Exception:
            # Index probably already exists (or server/version difference). Good enough for MVP.
            pass

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Convert a list of texts to a list of vectors.

        fastembed often returns an iterator/generator of numpy arrays, so we materialize it
        and convert vectors to plain Python lists for Qdrant.
        """
        vectors = list(self.embedding_model.embed(texts))

        out: List[List[float]] = []
        for v in vectors:
            # v is usually a numpy array; convert to list[float]
            if hasattr(v, "tolist"):
                out.append(v.tolist())
            else:
                out.append(list(v))
        return out

    def upsert_papers(self, papers: List[Dict[str, Any]]) -> None:
        """
        Upsert a batch of papers into Qdrant.

        - ID: uses paper['arxiv_id'] (string IDs are supported)
        - Vector: embedding of abstract (fallback title)
        - Payload: stores entire paper dict
        """
        if not papers:
            return

        texts_to_embed: List[str] = []
        ids: List[str] = []
        payloads: List[Dict[str, Any]] = []

        for p in papers:
            arxiv_id = p.get("arxiv_id")
            if not arxiv_id:
                # skip malformed input
                continue

            text = (p.get("abstract") or "").strip()
            if not text:
                text = (p.get("title") or "").strip()

            ids.append(self._qid(str(arxiv_id)))
            texts_to_embed.append(text)
            payloads.append(p)

        if not ids:
            return

        vectors = self.embed_texts(texts_to_embed)

        points: List[qm.PointStruct] = []
        for pid, vec, payload in zip(ids, vectors, payloads):
            points.append(
                qm.PointStruct(
                    id=pid,
                    vector=vec,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection,
            points=points,
        )

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        query_vector = self.embed_texts([query])[0]

        resp = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,      # dense vector -> nearest neighbor search
            limit=limit,
            with_payload=True,
        )

        # resp.points is a list of records; each record has .payload
        return [p.payload for p in resp.points if p.payload is not None]


