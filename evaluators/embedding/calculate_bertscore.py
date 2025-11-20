from bert_score import score


def calculate_bertscore(reference, hypothesis, lang="en"):
    P, R, F1 = score([hypothesis], [reference], lang=lang, rescale_with_baseline=True)
    return F1[0].item()  # Return the F1 score as a float
