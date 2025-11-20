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
from langchain_openai import AzureOpenAIEmbeddings

# 📁 Base directory
BASE_DIR = "./cases"

# 🔧 Embedding Model Configuration
# Choose which embedding model to use by uncommenting one of the options below:

# Option 1: text-embedding-ada-002 (older, reliable)
# EM_MODEL = "text-embedding-ada-002"
# EM_API_VERSION = "2023-05-15"
# OUTPUT_PATH = "./results_ada_002.csv"

# Option 2: text-embedding-3-large (newer, higher quality)
EM_MODEL = "text-embedding-3-large"  # For example
EM_API_VERSION = "2024-12-01-preview"  # Corresponding API version
OUTPUT_PATH = "./modelX.csv"

embedding = AzureOpenAIEmbeddings(
    deployment=EM_MODEL,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_type="azure",
    openai_api_version=EM_API_VERSION,
)


# Utility to load text from a file
def read_text(file_path):
    """Read text from a file and return it as a string."""
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

        # Load original file
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

        # Add self-to-self metrics (for Δ baseline)
        try:
            cosine = cosine_similarity(
                embeddings=embedding, text1=original_text, text2=original_text
            )
            bleu = calculate_bleu_score(original_text, original_text)
            rouge_scores = calculate_rouge_all(original_text, original_text)
            euclidean = calculate_euclidean_distance(
                embeddings=embedding, text1=original_text, text2=original_text
            )
            l1 = calculate_l1_distance(
                embeddings=embedding, text1=original_text, text2=original_text
            )
            dot_product = calculate_dot_product(
                embeddings=embedding, text1=original_text, text2=original_text
            )

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
                    cosine = cosine_similarity(
                        embeddings=embedding, text1=original_text, text2=mod_text
                    )
                    bleu = calculate_bleu_score(original_text, mod_text)
                    rouge_scores = calculate_rouge_all(original_text, mod_text)
                    euclidean = calculate_euclidean_distance(
                        embeddings=embedding, text1=original_text, text2=mod_text
                    )
                    l1 = calculate_l1_distance(
                        embeddings=embedding, text1=original_text, text2=mod_text
                    )

                    dot_product = calculate_dot_product(
                        embeddings=embedding, text1=original_text, text2=mod_text
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

# Save everything
df = pd.DataFrame(results)
df.to_csv(OUTPUT_PATH, index=False)
print(f"\nAll results saved to: {OUTPUT_PATH}")
