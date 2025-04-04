from langchain_chroma import Chroma
from models.huggingface.huggingface_pte_qwen_7B import load_embeddings


#TODO: think about using individual libraries or maybe langchain_community

def store_embeddings(chunks):
    """
    Use an LLM to create embeddings of the chunks and store them in a vector database. 

    Parameters:
    chunks (list): The chunks of text to store

    Returns:
    None
    """
    embeddings =  load_embeddings()

    # Create a Chroma vector store and persist it to disk
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory="./vector_db")