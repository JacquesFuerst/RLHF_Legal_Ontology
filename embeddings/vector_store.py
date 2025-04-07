from langchain_chroma import Chroma
from models.huggingface.embedding_model import EmbeddingModel

import os
import torch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# load the relevant devices available on the server
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("AVAILABLE_DEVICES")
print("Number of GPUs available: ", torch.cuda.device_count())



#TODO: think about using individual libraries or maybe langchain_community

def store_embeddings(chunks):
    """
    Use an LLM to create embeddings of the chunks and store them in a vector database. 

    Parameters:
    chunks (list): The chunks of text to store

    Returns:
    None
    """
    embed_func = EmbeddingModel(os.getenv("EMBEDDING_MODEL_NAME"))  # Load the embedding model name from environment variables

    # Create a Chroma vector store and persist it to disk
    vector_store = Chroma.from_documents(chunks, embed_func, persist_directory="./vector_db")