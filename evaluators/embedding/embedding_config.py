"""
Embedding Model Configuration Utility
Use this script to easily switch between different embedding models in run.py
"""

import re

# Available embedding model configurations
EMBEDDING_CONFIGS = {
    "ada-002": {
        "model": "text-embedding-ada-002",
        "api_version": "2023-05-15",
        "output_file": "./results_ada_002.csv",
        "description": "OpenAI Ada-002 (older, reliable, good performance)",
    },
    "3-large": {
        "model": "text-embedding-3-large",
        "api_version": "2024-12-01-preview",
        "output_file": "./results_3_large.csv",
        "description": "OpenAI text-embedding-3-large (newer, highest quality)",
    },
}


def read_run_py():
    """Read the current run.py file."""
    with open("run.py", "r") as f:
        return f.read()


def write_run_py(content):
    """Write content to run.py file."""
    with open("run.py", "w") as f:
        f.write(content)


def get_current_config():
    """Detect which embedding model is currently configured."""
    content = read_run_py()

    # Look for active (uncommented) EM_MODEL line
    for line in content.split("\n"):
        if line.strip().startswith("EM_MODEL = ") and not line.strip().startswith("#"):
            for key, config in EMBEDDING_CONFIGS.items():
                if config["model"] in line:
                    return key
    return None


def switch_embedding_model(target_key):
    """Switch to a specific embedding model."""
    if target_key not in EMBEDDING_CONFIGS:
        print(f"❌ Unknown embedding model: {target_key}")
        print(f"Available options: {', '.join(EMBEDDING_CONFIGS.keys())}")
        return False

    content = read_run_py()
    target_config = EMBEDDING_CONFIGS[target_key]

    # Build the new configuration section
    new_config = f"""# 🔧 Embedding Model Configuration
# Choose which embedding model to use by uncommenting one of the options below:

# Option 1: text-embedding-ada-002 (older, reliable)
{"" if target_key == "ada-002" else "# "}EM_MODEL = "text-embedding-ada-002"
{"" if target_key == "ada-002" else "# "}EM_API_VERSION = "2023-05-15"
{"" if target_key == "ada-002" else "# "}OUTPUT_PATH = "./results_ada_002.csv"

# Option 2: text-embedding-3-large (newer, higher quality)
{"" if target_key == "3-large" else "# "}EM_MODEL = "text-embedding-3-large"
{"" if target_key == "3-large" else "# "}EM_API_VERSION = "2024-12-01-preview"
{"" if target_key == "3-large" else "# "}OUTPUT_PATH = "./results_3_large.csv\""""

    # Replace the configuration section
    pattern = r"# 🔧 Embedding Model Configuration.*?(?=\n\nembedding = AzureOpenAIEmbeddings)"
    new_content = re.sub(pattern, new_config, content, flags=re.DOTALL)

    write_run_py(new_content)

    print(f"✅ Switched to {target_config['model']}")
    print(f"   API Version: {target_config['api_version']}")
    print(f"   Output file: {target_config['output_file']}")
    print(f"   Description: {target_config['description']}")

    return True


def main():
    """Main function to handle command line arguments."""
    import sys

    current = get_current_config()

    print("🔧 Embedding Model Configuration Utility")
    print("=" * 50)

    if current:
        current_config = EMBEDDING_CONFIGS[current]
        print(f"📍 Current model: {current_config['model']}")
        print(f"   {current_config['description']}")
    else:
        print("❓ Could not detect current embedding model")

    print("\n📋 Available embedding models:")
    for key, config in EMBEDDING_CONFIGS.items():
        status = "✅ ACTIVE" if key == current else "  "
        print(f"   {status} {key}: {config['description']}")

    if len(sys.argv) == 2:
        target = sys.argv[1]
        print(f"\n🔄 Switching to: {target}")
        success = switch_embedding_model(target)
        if success:
            print("\n✨ Ready to run: python run.py")
    else:
        print("\n💡 Usage: python embedding_config.py <model_key>")
        print("   Example: python embedding_config.py ada-002")
        print("   Example: python embedding_config.py 3-large")


if __name__ == "__main__":
    main()
