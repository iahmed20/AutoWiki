"""
Scoring for generated articles. This is new — the original project had
no evaluation code anywhere. It computes real ROUGE/BLEU scores; it does
not assume or hardcode any particular result. To get a defensible
"hallucination reduction" or quality number, run compare_to_baseline()
against a real reference article (e.g. the actual Wikipedia page) and
report the numbers this actually produces.
"""
from dataclasses import dataclass

import sacrebleu
from rouge_score import rouge_scorer

_SCORER = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


@dataclass
class ScoreReport:
    rouge1_f: float
    rouge2_f: float
    rougeL_f: float
    bleu: float

    def as_dict(self) -> dict:
        return {
            "rouge1_f": round(self.rouge1_f, 4),
            "rouge2_f": round(self.rouge2_f, 4),
            "rougeL_f": round(self.rougeL_f, 4),
            "bleu": round(self.bleu, 2),
        }


def score(reference: str, candidate: str) -> ScoreReport:
    """ROUGE-1/2/L F-measure and corpus BLEU of `candidate` against `reference`."""
    rouge = _SCORER.score(reference, candidate)
    bleu = sacrebleu.corpus_bleu([candidate], [[reference]])
    return ScoreReport(
        rouge1_f=rouge["rouge1"].fmeasure,
        rouge2_f=rouge["rouge2"].fmeasure,
        rougeL_f=rouge["rougeL"].fmeasure,
        bleu=bleu.score,
    )


def compare_to_baseline(reference: str, baseline_article: str, retrieval_article: str) -> dict:
    """Score both a no-retrieval baseline article and the retrieval-augmented
    article against the same reference, so the two are directly comparable."""
    return {
        "baseline": score(reference, baseline_article).as_dict(),
        "retrieval_augmented": score(reference, retrieval_article).as_dict(),
    }
