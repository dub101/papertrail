import feedparser
import re
from typing import List, Dict
from urllib.parse import urlencode

def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def search_arxiv(query: str, top_n: int = 5) -> List[Dict]:
    base_url = "https://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": top_n,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    url = f"{base_url}?{urlencode(params)}"

    feed = feedparser.parse(url, request_headers={"User-Agent": "paper-researcher/0.1"})
    if not getattr(feed, "entries", None):
        return []

    results = []
    for entry in feed.entries:
        arxiv_id = entry.id.split("/abs/")[-1]
        title = normalize_whitespace(entry.title)
        authors = [a.name for a in getattr(entry, "authors", [])]
        categories = [t["term"] for t in getattr(entry, "tags", [])]
        abstract = normalize_whitespace(entry.summary)
        entry_url = entry.id

        pdf_url = next(
            (link.href for link in getattr(entry, "links", []) if getattr(link, "type", "") == "application/pdf"),
            entry.id.replace("/abs/", "/pdf/"),
        )

        results.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "authors": authors,
            "categories": categories,
            "abstract": abstract,
            "entry_url": entry_url,
            "pdf_url": pdf_url,
            "published": entry.get("published"),
            "updated": entry.get("updated"),
        })

    return results

