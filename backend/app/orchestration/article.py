"""
Writes the final article section by section from the refined outline.

Each section is grounded by querying the vector store for the chunks
most relevant to that section's heading, rather than stuffing every
section's prompt with the full research transcript — the actual reason
to use retrieval here instead of just concatenating everything the
interviews produced.

Note on what changed here: the original `final_article()` in the
prototype notebook had a hardcoded CRISPR outline baked into the
function body instead of using its own `outline` argument, so it
couldn't actually write about any topic other than CRISPR. This
version parses the real outline and writes each section from it.
"""
import re
from dataclasses import dataclass
from typing import List, Optional

from langchain_chroma import Chroma
from openai import OpenAI

from app.config import settings
from app.orchestration.interview import Interview
from app.retrieval.vector_store import retrieve

_HEADING_RE = re.compile(r"^(#{1,4})\s+(.*)$")
DEFAULT_TOP_K = 6


@dataclass
class Section:
    level: int  # 1 = #, 2 = ##, etc.
    title: str


def parse_outline(outline: str) -> List[Section]:
    sections = []
    for line in outline.splitlines():
        match = _HEADING_RE.match(line.strip())
        if match:
            sections.append(Section(level=len(match.group(1)), title=match.group(2).strip()))
    return sections


def write_article(
    topic: str,
    outline: str,
    interviews: List[Interview],
    store: Optional[Chroma] = None,
    top_k: int = DEFAULT_TOP_K,
) -> str:
    """Write the article by generating prose for each outline section.

    If `store` is given, each section is grounded by an embedding-based
    similarity search against that topic's indexed research (scoped with
    a metadata filter so one topic's index doesn't leak into another's
    article). If no store is given, falls back to the full interview
    transcripts, which is simpler but less targeted."""
    sections = parse_outline(outline)
    if not sections:
        raise ValueError("Outline had no parseable headings — check refine_outline() output.")

    fallback_sources = "\n\n".join(interview.as_transcript() for interview in interviews)
    client = OpenAI(api_key=settings.require_openai())

    article_parts = [f"# {topic}"]
    for section in sections:
        heading_md = "#" * (section.level + 1)  # nest one level under the title
        sources = _gather_section_sources(topic, section, store, top_k, fallback_sources)
        prose = _write_section(client, topic, section.title, sources)
        article_parts.append(f"{heading_md} {section.title}\n\n{prose}")

    return "\n\n".join(article_parts)


def _gather_section_sources(
    topic: str,
    section: Section,
    store: Optional[Chroma],
    top_k: int,
    fallback_sources: str,
) -> str:
    if store is None:
        return fallback_sources

    query = f"{topic}: {section.title}"
    hits = retrieve(store, query, k=top_k, where={"topic": topic})
    if not hits:
        return fallback_sources

    return "\n\n".join(f"[{hit.metadata.get('kind', 'source')}] {hit.page_content}" for hit in hits)


def _write_section(client: OpenAI, topic: str, section_title: str, sources: str) -> str:
    response = client.chat.completions.create(
        model=settings.strong_model,
        messages=[
            {
                "role": "system",
                "content": "You are a Wikipedia Article Writer. Write in a neutral, encyclopedic tone.",
            },
            {
                "role": "user",
                "content": (
                    f"Write the '{section_title}' section of a Wikipedia-style article about "
                    f"{topic}. Ground claims in the research below where relevant, and write "
                    "plainly — no meta-commentary about the task. Two to four paragraphs.\n\n"
                    f"Research:\n{sources}"
                ),
            },
        ],
    )
    return response.choices[0].message.content
