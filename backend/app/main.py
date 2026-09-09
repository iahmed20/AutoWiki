"""
 # Full pipeline, write article to a file python -m app.main "CRISPR" --out crispr_article.md
"""
import argparse
import json
import sys

from app.evaluation.baseline import generate_baseline_article
from app.evaluation.metrics import compare_to_baseline
from app.orchestration.article import write_article
from app.orchestration.interview import run_all_interviews
from app.orchestration.outline import draft_outline, refine_outline
from app.orchestration.personas import generate_personas
from app.retrieval.vector_store import get_vector_store


def generate_article(topic: str, turns: int = 5, max_personas: int = 5) -> str:
    store = get_vector_store(topic)
    personas = generate_personas(topic, max_personas=max_personas)
    interviews = run_all_interviews(topic, personas, turns=turns, store=store)
    draft = draft_outline(topic)
    outline = refine_outline(topic, draft, interviews)
    return write_article(topic, outline, interviews, store=store)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Wikipedia-style article.")
    parser.add_argument("topic", help="Article topic")
    parser.add_argument("--out", help="Write the article to this file (default: stdout)")
    parser.add_argument("--turns", type=int, default=5, help="Interview questions per persona")
    parser.add_argument("--max-personas", type=int, default=5)
    parser.add_argument("--baseline", action="store_true", help="Also generate a no-retrieval baseline article")
    parser.add_argument("--reference", help="Path to a reference article to score both outputs against")
    args = parser.parse_args()

    article = generate_article(args.topic, turns=args.turns, max_personas=args.max_personas)

    if args.out:
        with open(args.out, "w") as f:
            f.write(article)
    else:
        print(article)

    if args.baseline:
        baseline_article = generate_baseline_article(args.topic)
        if args.out:
            baseline_path = args.out.rsplit(".", 1)[0] + ".baseline.md"
            with open(baseline_path, "w") as f:
                f.write(baseline_article)

        if args.reference:
            with open(args.reference) as f:
                reference = f.read()
            report = compare_to_baseline(reference, baseline_article, article)
            print(json.dumps(report, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
