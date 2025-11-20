import os

import pandas as pd
from jinja2 import Environment, FileSystemLoader
from langchain.schema import HumanMessage
from langchain_openai import AzureChatOpenAI

# Optional Azure AI Inference SDK (for DeepSeek)
try:
    from azure.ai.inference import ChatCompletionsClient  # type: ignore
    from azure.ai.inference.models import (
        SystemMessage as DSSystemMessage,  # type: ignore
    )
    from azure.ai.inference.models import UserMessage as DSUserMessage  # type: ignore
    from azure.core.credentials import AzureKeyCredential  # type: ignore

    HAS_INFERENCE = True
except ImportError:
    HAS_INFERENCE = False


# ----------------------------------------------------------------------------
# 📁 Configuration
# ----------------------------------------------------------------------------

# 📁 Configuration
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cases"))
OUTPUT_PATH = "../results_llm_o4-mini.csv"
# Point to the directory where this script resides to reliably locate the template
TEMPLATE_DIR = os.path.dirname(__file__)
LLM_MODEL = "o4-mini"

# Template directory (same folder as this script)
TEMPLATE_DIR = os.path.dirname(__file__)

# DeepSeek model via Azure AI Inference (required)
DEEPSEEK_CONFIG = {
    "name": "DeepSeek-R1-0528",
    "endpoint": "https://<resource-name>.cognitiveservices.azure.com/",  # Change to your endpoint
    "api_version": "2024-05-01-preview",
}

# Evaluate only the DeepSeek model
MODELS = [
    {
        "name": DEEPSEEK_CONFIG["name"],
        "deployment": DEEPSEEK_CONFIG["name"],
        "client_type": "inference",
    }
]

# Default Azure OpenAI endpoint (fallback if environment variable missing)
DEFAULT_AZURE_ENDPOINT = "https://<resource-name>.cognitiveservices.azure.com/"

# ----------------------------------------------------------------------------
# 🔧 Template setup
# ----------------------------------------------------------------------------

# Use correct filename (Linux is case-sensitive)
template_env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
template = template_env.get_template("Metaprompted.jinja2")

# ----------------------------------------------------------------------------
# 🧰 Helper functions
# ----------------------------------------------------------------------------


def create_llm_client(model_cfg):
    """Return an object with a .invoke(prompt:str)->str interface for the given model config."""
    if model_cfg.get("client_type") == "inference":
        if not HAS_INFERENCE:
            raise RuntimeError(
                "azure-ai-inference SDK not installed. Run `pip install azure-ai-inference`."
            )

        client = ChatCompletionsClient(
            endpoint=DEEPSEEK_CONFIG["endpoint"],
            credential=AzureKeyCredential(os.getenv("OPENAI_API_KEY")),
            api_version=DEEPSEEK_CONFIG["api_version"],
        )

        # Wrap in simple callable to match LangChain interface
        def _invoke(prompt: str):
            resp = client.complete(
                messages=[
                    DSSystemMessage(content="You are a helpful assistant."),
                    DSUserMessage(content=prompt),
                ],
                max_tokens=2048,
                model=model_cfg["deployment"],
            )
            return resp.choices[0].message.content

        return _invoke

    # Default: Azure OpenAI via LangChain
    azure_llm = AzureChatOpenAI(
        deployment_name=model_cfg["deployment"],
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", DEFAULT_AZURE_ENDPOINT),
        openai_api_version=model_cfg["api_version"],
    )

    def _invoke(prompt: str):
        return azure_llm([HumanMessage(content=prompt)]).content

    return _invoke


def evaluate_model(model_cfg):
    """Run evaluation for a single model and write CSV."""
    model_name = model_cfg["name"]
    print(f"\n🚀 Evaluating model: {model_name}")

    invoke_fn = create_llm_client(model_cfg)

    results = []

    # Iterate through language folders
    for lang_folder in ["english_cases", "finnish_cases", "swedish_cases"]:
        lang_path = os.path.join(BASE_DIR, lang_folder)
        language = lang_folder.replace("_cases", "")

        if not os.path.isdir(lang_path):
            print(f"⚠️  Missing folder: {lang_path}. Skipping.")
            continue

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

            original_text = read_text(os.path.join(original_dir, original_files[0]))
            if not original_text:
                continue

            for variation in ["delete", "modify", "synonyms"]:
                variation_path = os.path.join(case_path, variation)
                if not os.path.exists(variation_path):
                    continue

                for file in os.listdir(variation_path):
                    if not file.endswith(".csv"):
                        continue

                    mod_text = read_text(os.path.join(variation_path, file))
                    if not mod_text:
                        continue

                    prompt = template.render(
                        original=original_text,
                        variation=mod_text,
                        language=language,
                        variation_type=variation,
                        case_id=case_dir,
                    )

                    try:
                        response = invoke_fn(prompt)
                        results.append(
                            {
                                "case": case_dir,
                                "language": language,
                                "type": variation,
                                "file": file,
                                "llm_evaluation": response,
                            }
                        )
                        print(f"✅ {case_dir}/{file} evaluated.")
                    except Exception as e:
                        print(f"❌ {case_dir}/{file} error: {e}")

    # Save CSV
    if results:
        df = pd.DataFrame(results)
        output_file = f"./results_{model_name}_metaprompted.csv"
        df.to_csv(output_file, index=False)
        print(f"💾 Saved results to {output_file}\n")
    else:
        print(f"⚠️  No results for model {model_name}. Nothing saved.\n")


# -----------------------------------------------------------------------------
# 📚 Utility Functions
# -----------------------------------------------------------------------------


def read_text(file_path: str) -> str:
    """Read text from a file and return it as a string (strip trailing newlines)."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        print(f"❌ Failed to read {file_path}: {e}")
        return ""


# ----------------------------------------------------------------------------
# 🚀 Main execution
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    for cfg in MODELS:
        evaluate_model(cfg)
    print("\n✅ All model evaluations completed.")
