from langchain_ollama import ChatOllama

#TODO: figure out whether Ollama is truly the best way to do this... 

def load_llm():
    """
    Load the LLM model for the RAG.
    
    Returns:
    ChatOllama: The LLM model
    """

    model = ChatOllama(
        model="llama3.2",
        language="en")
    return model

def return_model_name():
    """
    Returns:
    str: The name of the model used in the RAG
    """
    return "llama3.2_v1"