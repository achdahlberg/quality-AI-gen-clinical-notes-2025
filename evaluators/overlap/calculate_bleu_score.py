import sacrebleu
from promptflow.core import tool


@tool
def calculate_bleu_score(candidate_text: str, reference_text: str) -> float:
    """
    Calculate the BLEU score for the candidate text against the reference text.
    """
    bleu_score = sacrebleu.corpus_bleu([candidate_text], [[reference_text]])
    return bleu_score.score / 100
