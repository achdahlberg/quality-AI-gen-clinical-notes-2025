"Checking from csv the missing ones and re-running only them"

import os

import pandas as pd
from jinja2 import Environment, FileSystemLoader
from langchain.schema import HumanMessage
from langchain_ollama import ChatOllama

# 📁 Configuration
BASE_DIR = "X"  # Add your configurations
RESULTS_CSV = "X/results"
TEMPLATE_DIR = "/X/LLM_evaluator"
LLM_MODEL = "gpt-oss:20b"  # Change to your model

template_env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
template = template_env.get_template("chatgpt.jinja2")


llm = ChatOllama(
    model=LLM_MODEL,
    temperature="0",
    num_predict="4000",
    base_url="Y",  # Change to your Ollama server URL
)


def read_text(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")
        return ""


def find_original_csv(case_id: str, language: str) -> str | None:
    """Return the first .csv in the original folder for this case and language."""
    original_dir = os.path.join(BASE_DIR, f"{language}_cases", case_id, "original")
    if not os.path.isdir(original_dir):
        print(f"Missing original dir: {original_dir}")
        return None
    files = [f for f in os.listdir(original_dir) if f.endswith(".csv")]
    if not files:
        print(f"No original CSV in {original_dir}")
        return None
    return os.path.join(original_dir, files[0])


def main():
    if not os.path.exists(RESULTS_CSV):
        raise FileNotFoundError(f"Results file not found: {RESULTS_CSV}")

    # Keep everything as strings to avoid NaN coercion surprises
    df = pd.read_csv(RESULTS_CSV, dtype=str)

    required_cols = {"case", "language", "type", "file", "llm_evaluation"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Results CSV missing required columns: {missing_cols}")

    # Empty means NaN or blank after stripping
    mask_missing = df["llm_evaluation"].isna() | df["llm_evaluation"].astype(
        str
    ).str.strip().eq("")

    if not mask_missing.any():
        print("Everything already has an answer. Nothing to do.")
        return

    to_run = df[mask_missing].copy()
    print(f"▶Re-evaluating {len(to_run)} item(s) without answers...")

    for idx, row in to_run.iterrows():
        language = str(row["language"]).strip()
        case_id = str(row["case"]).strip()
        variation = str(row["type"]).strip()
        file_name = str(row["file"]).strip()

        original_path = find_original_csv(case_id, language)
        variation_path = os.path.join(
            BASE_DIR, f"{language}_cases", case_id, variation, file_name
        )

        original_text = read_text(original_path) if original_path else ""
        variation_text = read_text(variation_path)

        if not original_text or not variation_text:
            print(f"Skipping {case_id} / {file_name} due to missing text.")
            continue

        prompt = template.render(
            original=original_text,
            variation=variation_text,
            language=language,
            variation_type=variation,
            case_id=case_id,
        )

        try:
            response = llm([HumanMessage(content=prompt)]).content
            df.at[idx, "llm_evaluation"] = response
            print(f"Re-evaluated: {case_id} / {file_name}")
        except Exception as e:
            print(f"LLM error for {case_id} / {file_name}: {e}")

    df.to_csv(RESULTS_CSV, index=False)
    print(f"\n Updated results saved to: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
