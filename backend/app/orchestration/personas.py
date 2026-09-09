"""
Generates the set of expert 'perspectives' that will each interview
the research agent about the topic (the STORM multi-perspective step).
"""
from typing import List

from openai import OpenAI

from app.config import settings
from app.retrieval.wiki_context import find_related_topics, get_table_of_contents
from app.utils import parse_string_list

DEFAULT_PERSPECTIVES = ["Basic Fact Researcher"]


def _chat(prompt: str, model: str) -> str:
    client = OpenAI(api_key=settings.require_openai())
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a Wikipedia researcher."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


def draft_personas(topic: str, table_of_contents: str) -> List[str]:
    """First pass: infer candidate expert/stakeholder roles from the
    section structure of related articles."""
    prompt = (
        f"You are generating perspectives (personas) for writing a comprehensive "
        f"Wikipedia article about {topic}. Task: 1. Read across the section titles "
        "below and infer what types of experts or stakeholders would naturally speak "
        "to these topics. 2. Express each perspective as a role. 3. Remove duplicates "
        "or overly narrow personas. 4. Aim for diversity across the relevant fields. "
        "5. Return only a valid JSON array of strings.\n\n"
        f"Section titles:\n{table_of_contents}"
    )
    return parse_string_list(_chat(prompt, settings.strong_model))


def refine_personas(topic: str, draft: List[str], max_personas: int = 5) -> List[str]:
    """Second pass: cut the draft list down to a small, non-redundant set."""
    prompt = (
        f"You are curating perspectives (personas) to contribute to a Wikipedia "
        f"article about {topic}. Input: {draft}. Task: 1. Remove duplicates and "
        f"overly narrow roles. 2. Select only {max_personas} perspectives that, "
        "together, give broad coverage of the topic. 3. Return only a valid JSON "
        "array of strings."
    )
    return parse_string_list(_chat(prompt, settings.strong_model))


def generate_personas(topic: str, max_personas: int = 5) -> List[str]:
    """End-to-end persona generation for a topic, grounded in related
    Wikipedia articles when they exist, falling back to a generic
    fact-researcher persona when they don't."""
    related = find_related_topics(topic)
    toc = get_table_of_contents(related)

    if not toc.strip():
        return DEFAULT_PERSPECTIVES

    draft = draft_personas(topic, toc)
    refined = refine_personas(topic, draft, max_personas=max_personas)

    if "Basic Fact Researcher" not in refined:
        refined.append("Basic Fact Researcher")
    return refined
