from langchain_ollama import OllamaEmbeddings

def load_embeddings():
    """
    Load the LLM model for the RAG.
    
    Returns:
    ChatOllama: The LLM model
    """

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings

def return_model_name():
    """
    Returns:
    str: The name of the model used in the RAG
    """
    return "nomic-embed-text"