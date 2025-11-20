from rouge_score import rouge_scorer


def calculate_rouge_all(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rouge3", "rouge4", "rougeL"], use_stemmer=True
    )
    scores = scorer.score(reference, hypothesis)

    return {
        "rouge_1": scores["rouge1"].fmeasure,
        "rouge_2": scores["rouge2"].fmeasure,
        "rouge_3": scores["rouge3"].fmeasure,
        "rouge_4": scores["rouge4"].fmeasure,
        "rouge_l": scores["rougeL"].fmeasure,
    }
