# AutoWiki

Generates a Wikipedia-style long-form article on a given topic, adapted from
Stanford OVAL's STORM algorithm (arXiv: https://arxiv.org/pdf/2402.14207).

Rather than asking an LLM to write from memory, AutoWiki simulates a research
process first — generating expert personas, running agent-driven interviews
that search the web, building a topic outline from what was actually found —
and only then writes the article, grounded in that research.

## How it works

1. **Personas** — given a topic, ask GPT for related Wikipedia articles, scrape
   their section structure, and derive a small set of expert perspectives from
   it (e.g. for "CRISPR": Molecular Biologist, Bioethicist, Regulatory Policy
   Expert), falling back to a generic "Basic Fact Researcher" persona if no
   related articles exist.
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

## Tech stack

| Purpose | Technology |
|---|---|
| LLM calls | OpenAI API — `gpt-4.1` for reasoning-heavy steps, `gpt-4o-mini` for cheap/high-volume ones (both configurable) |
| Embeddings | OpenAI `text-embedding-3-small` |
| Agent orchestration | LangGraph (`create_react_agent`) — the research agent that decides when to search |
| LLM/tool glue | LangChain (`ChatOpenAI`, `StructuredTool`) |
| Vector store | Chroma (`langchain-chroma`), persisted locally under `backend/.chroma/`, one collection per topic |
| Web search | Google Programmable Search Engine (Custom Search JSON API) |
| Persona grounding | `Wikipedia-API` + `BeautifulSoup` (scrapes related articles' table of contents) |
| Evaluation | `rouge-score`, `sacrebleu` |

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

## Setup

```bash
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in your API keys
```

Fill in `backend/.env`:

- `OPENAI_API_KEY` — required. The account behind it needs available credit;
  a `RateLimitError: insufficient_quota` means billing needs attention at
  platform.openai.com/settings/organization/billing.
- `GOOGLE_CSE_API_KEY` + `GOOGLE_CSE_CX` — required. Get these from a
  [Programmable Search Engine](https://programmablesearchengine.google.com/)
  (gives you `GOOGLE_CSE_CX`) and a matching key from
  [Google Cloud Console](https://console.cloud.google.com/) with the
  **Custom Search API** enabled for that project.
- `USER_AGENT` — used for Wikipedia scraping; the default is fine.
- `AUTOWIKI_FAST_MODEL` / `AUTOWIKI_STRONG_MODEL` — optional overrides for
  which OpenAI models are used where.

## Usage

```bash
python -m app.main "Your topic" --out article.md
```

Options:

| Flag | Default | Meaning |
|---|---|---|
| `--turns` | 5 | Interview questions asked per persona |
| `--max-personas` | 5 | Number of expert perspectives generated |
| `--baseline` | off | Also generate a no-retrieval comparison article |
| `--reference PATH` | — | Score both articles against a reference article you supply |

With evaluation against a reference:

```bash
python -m app.main "Your topic" --out article.md --baseline --reference reference.md
```

This prints a JSON report of ROUGE-1/2/L and BLEU for the baseline vs. the
retrieval-augmented article, both scored against your reference.

## Requirements

- OpenAI API key with available credit
- Google Programmable Search Engine (CSE) API key + search engine ID, with the
  Custom Search API enabled on the associated Google Cloud project

## Troubleshooting

- **`openai.RateLimitError: insufficient_quota` / `credit_balance_exhausted`**
  — the OpenAI account has no credit. Add credits and re-run.
- **`requests.exceptions.HTTPError: 403 Client Error: Forbidden` from
  `googleapis.com/customsearch/v1`** — Google is rejecting the CSE key/`cx`
  pair. Common causes: the Custom Search API isn't enabled on the key's
  Google Cloud project, the key has restrictions (e.g. HTTP referrer) that
  block server-side calls, or `cx` doesn't belong to that project.
- **Outline has no parseable headings** (`ValueError` from `article.py`) — the
  model didn't return Markdown headings for the outline; usually transient,
  re-run.
