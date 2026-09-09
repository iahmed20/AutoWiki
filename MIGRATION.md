# Applying this restructure to your repo

## What changed and why

- **`backend/controllers/`, `backend/models/`, `backend/src/`** — these were an
  Express/Node skeleton where every file was empty (0 bytes). Nothing in the
  actual project uses them. Delete them.
- **`backend/stormalgo.py`** and **`agent_prot.ipynb`** — this is where the real
  logic lived, spread across a standalone script and a notebook, with some
  duplicated logic (persona generation appears in both, slightly differently)
  and one real bug: `final_article()` in the notebook had a hardcoded CRISPR
  outline instead of using its own `outline` argument, so it could never
  actually write about anything else. Both are replaced by `backend/app/`.
- **`backend/app/`** — the new package, split into `retrieval/`,
  `orchestration/`, and `evaluation/`, described below.

## New structure

```
backend/
  app/
    config.py              # env vars (replaces google.colab.userdata)
    utils.py                # shared helpers
    retrieval/
      web_search.py          # Google CSE search tool (was notebook cell 7)
      wiki_context.py         # related-topic + TOC scraping (was stormalgo.py)
      vector_store.py          # Chroma wrapper for embedding-based retrieval — NEW
    orchestration/
      personas.py             # perspective generation
      interview.py             # the ReAct research agent + interview loop (was notebook cell 8)
      outline.py                # outline drafting + refinement (was notebook cells 9-10)
      article.py                 # section-by-section article writing (fixes the CRISPR bug)
    evaluation/
      baseline.py                 # no-retrieval comparison article — NEW
      metrics.py                   # real ROUGE/BLEU scoring — NEW
    main.py                          # CLI entrypoint (replaces blocking input())
  requirements.txt
  .env.example
```

## Steps

From the root of your `AutoWiki` checkout:

```bash
# 1. Remove the dead Node.js skeleton
git rm -r backend/controllers backend/models backend/src

# 2. Remove the old script and notebook (their logic now lives in backend/app/)
git rm backend/stormalgo.py agent_prot.ipynb

# 3. Copy in the new package (from wherever you extracted this delivery)
cp -r path/to/delivery/backend/app backend/app
cp path/to/delivery/backend/requirements.txt backend/requirements.txt
cp path/to/delivery/backend/.env.example backend/.env.example

# 4. Set up your own .env (never commit this)
cp backend/.env.example backend/.env
# then fill in OPENAI_API_KEY, GOOGLE_CSE_API_KEY, GOOGLE_CSE_CX

# 5. Install and smoke-test
cd backend
pip install -r requirements.txt
python -m app.main "Your test topic" --out test_article.md

# 6. Commit
cd ..
git add backend README.md .gitignore
git commit -m "Restructure backend into retrieval/orchestration/evaluation layers"
git push
```

Also add a `.gitignore` with at least:
```
.env
__pycache__/
*.pyc
.chroma/
```

## Getting a real evaluation number

Once the pipeline runs end to end, generate a baseline + retrieval-augmented
pair for a topic you can also get a solid reference article for (e.g. the
actual current Wikipedia article on that topic), then score both:

```bash
python -m app.main "CRISPR" --out crispr.md --baseline --reference crispr_wikipedia_actual.md
```

This prints a JSON report comparing ROUGE-1/2/L and BLEU for the baseline vs.
the retrieval-augmented article against that reference. Run it across a handful
of topics and average the results — that's what turns "cut hallucination by X%"
from a claim into something you can actually defend if asked about it.
