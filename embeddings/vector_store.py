from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

#TODO: think about using individual libraries or maybe langchain_community

def store_embeddings(chunks):
    """
    Use an LLM to create embeddings of the chunks and store them in a vector database. 
    """
    embeddings = OllamaEmbeddings(model="llama3.2") #TODO: use llama to create the embeddings for now, maybe switch language model
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory="./vector_db")