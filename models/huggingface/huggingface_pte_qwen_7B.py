from sentence_transformers import SentenceTransformer 

def load_llm():
    """
    Load the LLM model for the RAG.
    
    Returns:
    ChatOllama: The LLM model
    """

    model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-7B-instruct", trust_remote_code=True)
    return model

def load_embeddings(text):
    """
    Load the LLM model for the RAG.
    
    Returns:
    ChatOllama: The LLM model
    """

    model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-7B-instruct", trust_remote_code=True)
    embeddings = model.encode(text, show_progress_bar=True)
    return embeddings

def return_model_name():
    """
    Returns:
    str: The name of the model used in the RAG
    """
    return "Alibaba-NLP/gte-Qwen2-7B-instruct"