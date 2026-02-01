import os
import json
import uuid
import time

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from apps.api.app.arxiv_client import search_arxiv
from apps.api.app.vector_store import VectorStore

app = FastAPI(title="paper-researcher")

_vector_store = None


def get_vector_store():
    global _vector_store
    if _vector_store is not None:
        return _vector_store

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    try:
        _vector_store = VectorStore(qdrant_url=qdrant_url, collection="papers")
        return _vector_store
    except Exception as e:
        print(f"[qdrant init error] {e}")
        _vector_store = None
        return None


class ChatRequest(BaseModel):
    conversation_id: str | None = None
    message: str = Field(..., min_length=1)
    top_n: int = Field(5, ge=1, le=20)


def sse(event: str, data: dict) -> str:
    return f"event: {event}\n" + f"data: {json.dumps(data)}\n\n"


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/chat")
def chat(req: ChatRequest):
    conv_id = req.conversation_id or str(uuid.uuid4())

    def format_papers_as_stream(papers: list[dict], source_label: str):
        """
        Helper to stream the same grounded bullets format from either Qdrant or arXiv.
        source_label is just used in the intro line.
        """
        yield sse("delta", {"text": f"Found {len(papers)} papers ({source_label}) for '{req.message}'.\n"})
        time.sleep(0.2)

        for paper in papers:
            first_author = paper["authors"][0] if paper.get("authors") else "Unknown"
            categories = ", ".join((paper.get("categories") or [])[:2]) or "Uncategorized"
            abstract = paper.get("abstract") or ""
            first_sentence = (abstract.split(". ")[0] + ".") if abstract else "No abstract available."
            bullet_line = f"- {paper.get('title', '')} by {first_author} [{categories}]: {first_sentence}\n"
            yield sse("delta", {"text": bullet_line})
            time.sleep(0.2)

        sources = [{"arxiv_id": p.get("arxiv_id", ""), "url": p.get("entry_url", "")} for p in papers]
        yield sse("sources", {"items": sources})
        yield sse("done", {"conversation_id": conv_id})

    def event_stream():
        yield sse("status", {"stage": "start", "message": "Starting..."})
        time.sleep(0.2)

        # 1) Try Qdrant first (semantic retrieval)
        vs = get_vector_store()
        if vs is not None:
            try:
                yield sse("status", {"stage": "retrieval", "message": "Searching cached papers in Qdrant..."})
                cached = vs.search(req.message, limit=req.top_n)  # returns list[payload dict]
                if cached:
                    yield sse("status", {"stage": "retrieval", "message": f"Using {len(cached)} cached papers"})
                    yield from format_papers_as_stream(cached, source_label="Qdrant cache")
                    return
                else:
                    yield sse("status", {"stage": "retrieval", "message": "No cache hits, falling back to arXiv"})
            except Exception as e:
                print(f"[qdrant search error] {e}")
                yield sse("status", {"stage": "retrieval", "message": "Cache search failed, falling back to arXiv"})

        else:
            yield sse("status", {"stage": "retrieval", "message": "Qdrant unavailable, falling back to arXiv"})

        # 2) Fall back to arXiv search
        yield sse("status", {"stage": "arxiv_search", "message": "Searching arXiv..."})
        time.sleep(0.2)

        results = search_arxiv(req.message, req.top_n)
        if not results:
            yield sse("error", {"code": "ARXIV_EMPTY", "message": "No results found"})
            return

        # 3) Best-effort cache results in Qdrant
        vs2 = get_vector_store()
        if vs2 is None:
            yield sse("status", {"stage": "qdrant", "message": "Qdrant unavailable, skipping cache"})
        else:
            try:
                yield sse("status", {"stage": "qdrant", "message": f"Caching {len(results)} papers in Qdrant..."})
                vs2.upsert_papers(results)
            except Exception as e:
                print(f"[qdrant upsert error] {e}")
                yield sse("status", {"stage": "qdrant", "message": "Failed to cache in Qdrant, continuing"})

        # 4) Stream response from arXiv results
        yield from format_papers_as_stream(results, source_label="arXiv")
        return

    return StreamingResponse(event_stream(), media_type="text/event-stream")

