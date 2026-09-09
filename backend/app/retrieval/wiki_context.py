"""
Grounding-context retrieval: given a topic, find related Wikipedia
articles and pull their table-of-contents structure. This is what
orchestration/personas.py uses to figure out which expert perspectives
are relevant to a new article, rather than guessing from the topic
name alone.

TOC-scraping approach adapted from:
  Source: get_wiki_page_title_and_toc
  Author: shaoyijia
  Published: Apr 23, 2024 | Retrieved: Aug 18, 2025
  Availability: https://github.com/stanford-oval/storm/blob/main/knowledge_storm/storm_wiki/modules/persona_generator.py
  License: Stanford OVAL (see repo for details)
"""
from typing import List

import requests
import wikipediaapi
from bs4 import BeautifulSoup
from openai import OpenAI

from app.config import settings
from app.utils import parse_string_list

_EXCLUDED_SECTIONS = {"Contents", "See also", "Notes", "References", "External links"}


def _wiki_client() -> wikipediaapi.Wikipedia:
    return wikipediaapi.Wikipedia(user_agent=settings.user_agent, language="en")


def find_related_topics(topic: str, n: int = 10) -> List[str]:
    """Ask the model for related-but-distinct Wikipedia article titles
    to use as grounding context, deliberately excluding direct subtopics."""
    client = OpenAI(api_key=settings.require_openai())
    response = client.chat.completions.create(
        model=settings.strong_model,
        messages=[
            {"role": "system", "content": "You are a Wikipedia researcher."},
            {
                "role": "user",
                "content": (
                    f"Given the Wikipedia article {topic!r}, what are {n} other Wikipedia "
                    "article titles that are strongly related and would provide additional "
                    f"perspectives on {topic!r}? Exclude direct subtopics (like applications, "
                    "ethics, agriculture, medicine) and focus on distinct but relevant "
                    "scientific, technological, or social topics. Return only a valid JSON "
                    "array of strings."
                ),
            },
        ],
    )
    return parse_string_list(response.choices[0].message.content)


def get_table_of_contents(topics: List[str]) -> str:
    """Fetch and flatten the section headings of each related Wikipedia
    page into one indented outline string used as persona-generation input."""
    wiki = _wiki_client()
    toc_lines: List[str] = []

    for topic in topics:
        page = wiki.page(topic)
        if not page.exists():
            continue

        resp = requests.get(page.fullurl, timeout=15)
        soup = BeautifulSoup(resp.content, "html.parser")

        levels: List[int] = []
        for header in soup.find_all(["h2", "h3", "h4", "h5", "h6"]):
            level = int(header.name[1])
            title = header.text.replace("[edit]", "").strip().replace("\xa0", " ")
            if title in _EXCLUDED_SECTIONS:
                continue

            while levels and level <= levels[-1]:
                levels.pop()
            levels.append(level)

            indent = "  " * (len(levels) - 1)
            toc_lines.append(f"{indent}{title}")

    return "\n".join(toc_lines)



