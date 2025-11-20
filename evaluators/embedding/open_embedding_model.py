from promptflow.core import tool
from langchain_core.embeddings.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

@tool
def create_open_embedding_model(server_url: str, model_name: str) -> Embeddings:
    """
    # GA Open Embedding Model

    Creates an embedding model for a given open model to run on Ollama.
    This embedding model is needed for `GA Vectors / vectors.py` when using open language models.

    :param: server_url (str): Server URL for the model. If using a model installed on the same computer, the url is `http://127.0.0.1:11434`.
    :param: model_name (str): The name of the embedding model (e.g., `bge-m3`).

    :return: The `Embeddings` object used for `GA Vectors / vectors.py`.
    :rtype: Embeddings
    """
    return OllamaEmbeddings(model=model_name, base_url=server_url)