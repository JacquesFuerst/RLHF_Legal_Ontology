from langchain_ollama import ChatOllama


def load_llm():
    """
    Load the LLM model for the RAG.
    
    Returns:
    ChatOllama: The LLM model
    """

    model = ChatOllama(
        model="deepseek-r1",
        language="en")
    return model

def return_model_name():
    """
    Returns:
    str: The name of the model used in the RAG
    """
    return "deepseek-r1"