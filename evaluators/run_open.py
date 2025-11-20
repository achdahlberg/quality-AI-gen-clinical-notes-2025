import os

import pandas as pd
from calculate_bleu_score import calculate_bleu_score
from calculate_cosine_similarity import cosine_similarity
from calculate_rouge_all import calculate_rouge_all
from evaluators.tools.dot_product import (
    calculate_dot_product,
)
from evaluators.tools.euclidean_distance import (
    calculate_euclidean_distance,
)
from evaluators.tools.l1_distance import (
    calculate_l1_distance,
)
from langchain_ollama import OllamaEmbeddings

# Base directory
BASE_DIR = "./cases"
OUTPUT_PATH = "./results_model X.csv"

# Configure Ollama Embeddings
OLLAMA_MODEL = "snowflake-arctic-embed2"  # For example
OLLAMA_URL = "X"  # Default Ollama local server

embedding = OllamaEmbeddings(model=OLLAMA_MODEL, base_url=OLLAMA_URL)


# Utility to load text from a file
def read_text(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")
        return ""


# Storage
results = []

# Loop over languages
for lang_folder in ["english_cases", "finnish_cases", "swedish_cases"]:
    lang_path = os.path.join(BASE_DIR, lang_folder)
    LANGUAGE = lang_folder.replace("_cases", "")

    for case_dir in os.listdir(lang_path):
        case_path = os.path.join(lang_path, case_dir)
        if not os.path.isdir(case_path):
            continue

        original_dir = os.path.join(case_path, "original")
        if not os.path.exists(original_dir):
            print(f"No 'original' folder in {case_dir}")
            continue

        original_files = [f for f in os.listdir(original_dir) if f.endswith(".csv")]
        if not original_files:
            print(f"No CSV found in {original_dir}")
            continue

        original_file_path = os.path.join(original_dir, original_files[0])
        original_text = read_text(original_file_path)
        if not original_text:
            continue

        # Add self-to-self metrics
        try:
            cosine = cosine_similarity(embedding, original_text, original_text)
            bleu = calculate_bleu_score(original_text, original_text)
            rouge_scores = calculate_rouge_all(original_text, original_text)
            euclidean = calculate_euclidean_distance(
                embedding, original_text, original_text
            )
            l1 = calculate_l1_distance(embedding, original_text, original_text)
            dot_product = calculate_dot_product(embedding, original_text, original_text)

            results.append(
                {
                    "case": case_dir,
                    "language": LANGUAGE,
                    "type": "original",
                    "file": original_files[0],
                    "cosine_similarity": cosine,
                    "bleu_score": bleu,
                    "rouge_1": rouge_scores["rouge_1"],
                    "rouge_2": rouge_scores["rouge_2"],
                    "rouge_3": rouge_scores["rouge_3"],
                    "rouge_4": rouge_scores["rouge_4"],
                    "rouge_l_score": rouge_scores["rouge_l"],
                    "euclidean_distance": euclidean,
                    "l1_distance": l1,
                    "dot_product": dot_product,
                }
            )
            print(f"{case_dir} / ORIGINAL processed.")
        except Exception as e:
            print(f"Error processing ORIGINAL in {case_dir}: {e}")

        # Process variations
        for variation in ["delete", "modify", "synonyms"]:
            variation_path = os.path.join(case_path, variation)
            if not os.path.exists(variation_path):
                continue

            for file in os.listdir(variation_path):
                if not file.endswith(".csv"):
                    continue

                mod_file_path = os.path.join(variation_path, file)
                mod_text = read_text(mod_file_path)
                if not mod_text:
                    continue

                try:
                    cosine = cosine_similarity(embedding, original_text, mod_text)
                    bleu = calculate_bleu_score(original_text, mod_text)
                    rouge_scores = calculate_rouge_all(original_text, mod_text)
                    euclidean = calculate_euclidean_distance(
                        embedding, original_text, mod_text
                    )
                    l1 = calculate_l1_distance(embedding, original_text, mod_text)
                    dot_product = calculate_dot_product(
                        embedding, original_text, mod_text
                    )

                    results.append(
                        {
                            "case": case_dir,
                            "language": LANGUAGE,
                            "type": variation,
                            "file": file,
                            "cosine_similarity": cosine,
                            "bleu_score": bleu,
                            "rouge_1": rouge_scores["rouge_1"],
                            "rouge_2": rouge_scores["rouge_2"],
                            "rouge_3": rouge_scores["rouge_3"],
                            "rouge_4": rouge_scores["rouge_4"],
                            "rouge_l_score": rouge_scores["rouge_l"],
                            "euclidean_distance": euclidean,
                            "l1_distance": l1,
                            "dot_product": dot_product,
                        }
                    )
                    print(f"{case_dir} / {file} processed.")
                except Exception as e:
                    print(f"Error processing {file}: {e}")

# Save results
df = pd.DataFrame(results)
df.to_csv(OUTPUT_PATH, index=False)
print(f"\n All results saved to: {OUTPUT_PATH}")
