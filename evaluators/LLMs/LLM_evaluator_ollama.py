import os

import pandas as pd
from jinja2 import Environment, FileSystemLoader
from langchain.schema import HumanMessage
from langchain_ollama import ChatOllama

# 📁 Configuration
BASE_DIR = "/X/cases"
OUTPUT_PATH = "/X/results_modelY.csv"
TEMPLATE_DIR = "/X/LLM_evaluator"
LLM_MODEL = "gpt-oss:20b"  # Change to your model


template_env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
template = template_env.get_template("chatgpt.jinja2")


# llm = AzureChatOpenAI(
#    deployment_name="gpt-4",
#    openai_api_key=os.getenv("OPENAI_API_KEY"),
#    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#    openai_api_version="2024-02-01",
# )

llm = ChatOllama(
    model=LLM_MODEL,
    temperature="0",
    num_predict="5000",
    base_url="Y",  # Change to your Ollama server URL
)


def read_text(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")
        return ""


results = []

for lang_folder in ["english_cases", "finnish_cases", "swedish_cases"]:
    lang_path = os.path.join(BASE_DIR, lang_folder)
    LANGUAGE = lang_folder.replace("_cases", "")

    for case_dir in os.listdir(lang_path):
        case_path = os.path.join(lang_path, case_dir)
        if not os.path.isdir(case_path):
            continue

        original_dir = os.path.join(case_path, "original")
        if not os.path.exists(original_dir):
            continue

        original_files = [f for f in os.listdir(original_dir) if f.endswith(".csv")]
        if not original_files:
            continue

        original_file_path = os.path.join(original_dir, original_files[0])
        original_text = read_text(original_file_path)
        if not original_text:
            continue

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

                prompt = template.render(
                    original=original_text,
                    variation=mod_text,
                    language=LANGUAGE,
                    variation_type=variation,
                    case_id=case_dir,
                )

                try:
                    response = llm([HumanMessage(content=prompt)]).content
                    results.append(
                        {
                            "case": case_dir,
                            "language": LANGUAGE,
                            "type": variation,
                            "file": file,
                            "llm_evaluation": response,
                        }
                    )
                    print(f"{case_dir} / {file} evaluated by LLM.")
                except Exception as e:
                    print(f"Error with LLM evaluation for {file}: {e}")

# 💾 Save results
df = pd.DataFrame(results)
df.to_csv(OUTPUT_PATH, index=False)
print(f"\n LLM evaluation results saved to: {OUTPUT_PATH}")
