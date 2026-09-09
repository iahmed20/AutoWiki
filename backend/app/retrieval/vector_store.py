"""
Vector-based (embedding) retrieval, backed by Chroma.

This is what gets written into during the interview step (search results
and synthesized answers get embedded and stored), and what gets queried
during article writing (each section pulls only its most relevant chunks,
instead of every prompt getting the entire research transcript dumped
into it).
"""
import re
from pathlib import Path
from typing import Any, Dict, List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from app.config import settings

CHROMA_ROOT = Path(".chroma")
EMBEDDING_MODEL = "text-embedding-3-small"


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "default"


def get_vector_store(topic: str) -> Chroma:
    """One persisted Chroma collection per topic, so re-running the
    pipeline on the same topic can reuse and grow its index instead of
    starting cold each time."""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=settings.require_openai())
    persist_dir = CHROMA_ROOT / _slugify(topic)
    persist_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=_slugify(topic),
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )


def index_text(store: Chroma, text: str, metadata: Dict[str, Any]) -> None:
    """Embed and store a single chunk of retrieved or synthesized text."""
    if not text or not text.strip():
        return
    store.add_documents([Document(page_content=text, metadata=metadata)])


def retrieve(store: Chroma, query: str, k: int = 6, where: Dict[str, Any] | None = None) -> List[Document]:
    """Top-k most relevant chunks for a query, optionally filtered by
    metadata (e.g. {'topic': topic} to stay within one article's index)."""
    return store.similarity_search(query, k=k, filter=where)
