import numpy as np
from langchain_openai import AzureOpenAIEmbeddings
from numpy.linalg import norm
from promptflow.core import tool

# gives the cosine similarity between the embedding vectors of two texts
# the output value is between -1 and 1, where
# -1 = vectors are pointing in the opposite directions
#  0 = vectors are orthogonal
#  1 = vectors are pointing in the same direction


@tool
def cosine_similarity(
    embeddings: AzureOpenAIEmbeddings, text1: str, text2: str
) -> float:
    """
    Calculate cosine similarity between two texts using their embeddings.

    Args:
        embeddings: AzureOpenAIEmbeddings instance
        text1: First text to compare
        text2: Second text to compare

    Returns:
        float: Cosine similarity score between -1 and 1
    """
    # Get embeddings for both texts
    vector1 = embeddings.embed_query(text1)
    vector2 = embeddings.embed_query(text2)

    # Convert to numpy arrays for calculation
    a = np.array(vector1)
    b = np.array(vector2)

    # Calculate and return cosine similarity
    return np.dot(a, b) / (norm(a) * norm(b))
