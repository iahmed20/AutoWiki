"""
Web search retrieval, backed by Google Programmable Search (CSE).
Exposed as a LangChain StructuredTool so the research agent in
orchestration/interview.py can call it directly.
"""
import re
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import requests
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.config import settings

GOOGLE_CSE_ENDPOINT = "https://www.googleapis.com/customsearch/v1"
_TRACKING_PARAMS_PREFIX = ("utm_",)
_TRACKING_PARAMS_EXACT = {"gclid", "fbclid"}


def canonicalize_url(url: str) -> str:
    """Strip tracking params and normalize a URL so duplicate hits
    (same page, different campaign params) collapse together."""
    try:
        parts = urlparse(url)
        netloc = parts.netloc.lower().replace(":80", "").replace(":443", "")
        query = [
            (k, v)
            for k, v in parse_qsl(parts.query)
            if not k.startswith(_TRACKING_PARAMS_PREFIX) and k not in _TRACKING_PARAMS_EXACT
        ]
        return urlunparse((parts.scheme, netloc, parts.path, parts.params, urlencode(query), parts.fragment))
    except Exception:
        return url


class GoogleSearchInput(BaseModel):
    query: str = Field(..., description="Web search query")
    k: int = Field(5, ge=1, le=10, description="Max results to return")
    safe: str = Field("active", description="SafeSearch mode: active|off")
    lr: Optional[str] = Field(None, description="Language restrict, e.g. 'lang_en'")


def google_search(query: str, k: int = 5, safe: str = "active", lr: Optional[str] = None) -> List[Dict[str, Any]]:
    """Run a Google CSE search and return normalized {title, url, snippet, source} hits."""
    api_key, cx = settings.require_google_cse()

    params: Dict[str, Any] = {
        "key": api_key,
        "cx": cx,
        "q": query,
        "num": min(k, 10),
        "safe": safe,
    }
    if lr:
        params["lr"] = lr

    resp = requests.get(GOOGLE_CSE_ENDPOINT, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    hits: List[Dict[str, Any]] = []
    for item in data.get("items", []):
        title = (item.get("title") or "").strip()
        url = canonicalize_url((item.get("link") or "").strip())
        snippet = re.sub(r"\s+", " ", (item.get("snippet") or "").strip())
        if not title or not url:
            continue
        hits.append({"title": title, "url": url, "snippet": snippet, "source": "google_cse"})

    return hits[:k]


google_search_tool = StructuredTool.from_function(
    func=google_search,
    name="web_search",
    description="Search the web. Returns a list of {title, url, snippet, source}.",
    args_schema=GoogleSearchInput,
)
