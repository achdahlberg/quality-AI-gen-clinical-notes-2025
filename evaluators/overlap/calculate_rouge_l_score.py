from rouge_score import rouge_scorer
from promptflow.core import tool


@tool
def calculate_rouge_l_score(reference_text: str, candidate_text: str):
    """
    Calculate the ROUGE-L score between a reference and candidate text.
    Returns all ROUGE-L components after preprocessing.
    """

    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    # Compute ROUGE-L score
    scores = scorer.score(reference_text, candidate_text)

    # Extract ROUGE-L components
    # rouge_l_precision = scores['rougeL'].precision
    # rouge_l_recall = scores['rougeL'].recall

    rouge_l_score = scores["rougeL"].fmeasure

    return rouge_l_score
