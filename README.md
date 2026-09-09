# AutoWiki

Generates a Wikipedia-style long-form article on a given topic, adapted from
Stanford OVAL's STORM algorithm (arXiv: https://arxiv.org/pdf/2402.14207).

## How it works

1. **Personas** — given a topic, find related Wikipedia articles and derive a
   small set of expert perspectives from their section structure (e.g.
   Molecular Biologist, Regulatory Policy Expert), falling back to a generic
   researcher persona if no related articles exist.
2. **Interviews** — each persona interviews a LangGraph ReAct agent that has
   access to a Google Custom Search tool. The agent decides when to search and
   what to search for; it isn't a fixed retrieve-then-answer pipeline. Every
   search result and synthesized answer is embedded (OpenAI
   `text-embedding-3-small`) and written into a per-topic Chroma collection.
3. **Outline** — a draft outline is generated from the topic alone, then
   refined against what the interviews actually surfaced.
4. **Article** — each outline section queries the vector store for its most
   relevant chunks (rather than every section's prompt getting the full
   research transcript), and is written from those.
5. **Evaluation** *(optional)* — a no-retrieval baseline article is generated
   for the same topic, and both it and the retrieval-augmented article are
   scored (ROUGE-1/2/L, BLEU) against a reference article you supply.

## Project layout

```
backend/app/
  retrieval/
    web_search.py       # Google CSE tool the agent can call
    wiki_context.py      # related-topic + TOC scraping for persona grounding
    vector_store.py       # Chroma wrapper: index + similarity search
  orchestration/
    personas.py            # expert perspective generation
    interview.py             # the ReAct agent interview loop; indexes as it goes
    outline.py                 # draft + refine the article outline
    article.py                  # per-section writing, grounded via vector retrieval
  evaluation/
    baseline.py                  # no-retrieval comparison article
    metrics.py                    # ROUGE/BLEU scoring
```

See `MIGRATION.md` for what moved from the original notebook/script and why.

## Usage

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env   # fill in your API keys

python -m app.main "Your topic" --out article.md
```

With evaluation against a reference:

```bash
python -m app.main "Your topic" --out article.md --baseline --reference reference.md
```

## Requirements

- OpenAI API key
- Google Programmable Search Engine (CSE) API key + search engine ID
