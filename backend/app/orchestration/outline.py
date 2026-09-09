"""
Turns the persona interviews into a structured article outline —
first a rough draft from the topic alone, then a refined version
grounded in what the interviews actually surfaced.
"""
from typing import List

from openai import OpenAI

from app.config import settings
from app.orchestration.interview import Interview


def draft_outline(topic: str) -> str:
    """A quick first-pass outline from the model's own knowledge,
    before any research has happened. Used as a scaffold, not a
    final structure."""
    client = OpenAI(api_key=settings.require_openai())
    response = client.chat.completions.create(
        model=settings.fast_model,
        messages=[
            {"role": "system", "content": "You are a Wikipedia Article Writer and Editor."},
            {
                "role": "user",
                "content": (
                    f"Generate an outline of a hypothetical Wikipedia article on the topic: "
                    f"{topic}. Use # for headings, ## for subheadings, and ### for "
                    "sub-subheadings if necessary. Respond with only the outline."
                ),
            },
        ],
    )
    return response.choices[0].message.content


def refine_outline(topic: str, draft: str, interviews: List[Interview]) -> str:
    """Merge the draft outline with what the persona interviews actually
    surfaced, producing the outline the article will be written from."""
    transcripts = "\n\n".join(interview.as_transcript() for interview in interviews)

    client = OpenAI(api_key=settings.require_openai())
    response = client.chat.completions.create(
        model=settings.strong_model,
        messages=[
            {"role": "system", "content": "You are a Wikipedia Article Writer and Editor."},
            {
                "role": "user",
                "content": (
                    "Given a topic, a draft outline, and research conversations on the "
                    "topic, generate a comprehensive Wikipedia article outline. Use # for "
                    "headings, ## for subheadings, and ### for sub-subheadings only where "
                    "warranted by the research. Respond with only the outline.\n\n"
                    f"Topic: {topic}\n\nDraft outline:\n{draft}\n\nResearch conversations:\n{transcripts}"
                ),
            },
        ],
    )
    return response.choices[0].message.content
