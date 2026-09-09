"""
Generates a plain, no-retrieval article for the same topic, purely as
a comparison point for evaluation. This is what "GPT-4 alone" produces,
with no search agent, no personas, no outline refinement — the thing
the retrieval-augmented pipeline should be beating in metrics.py.
"""
from openai import OpenAI

from app.config import settings


def generate_baseline_article(topic: str) -> str:
    client = OpenAI(api_key=settings.require_openai())
    response = client.chat.completions.create(
        model=settings.strong_model,
        messages=[
            {"role": "system", "content": "You are a Wikipedia Article Writer."},
            {
                "role": "user",
                "content": (
                    f"Write a Wikipedia-style article about {topic} from your own "
                    "knowledge, with no external research. Use Markdown headings."
                ),
            },
        ],
    )
    return response.choices[0].message.content
