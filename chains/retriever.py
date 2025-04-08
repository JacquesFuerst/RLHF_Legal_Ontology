from langchain_chroma import Chroma
from models.huggingface.embedding_model import EmbeddingModel
import os
import torch
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# load the relevant devices available on the server
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("AVAILABLE_DEVICES")
# os.environ['TRANSFORMERS_OFFLINE'] = '1'


def retrieve_chunks(query):
    """
    When sending in the prompt, the system should return the most relevant text chunks from the vector database.

    Parameters:
    query (str): The query for the vector store

    Returns:
    list: The most relevant text chunks
    """

    embed_func = EmbeddingModel(os.getenv("EMBEDDING_MODEL_NAME"))  # Load the embedding model name from environment variables
    # Instantiate the OllamaEmbeddings model
    vector_store = Chroma(persist_directory="./vector_db", 
                          embedding_function=embed_func)
    
    # Get the total number of documents in the vector store
    # total_documents = vector_store._collection.count()
    # print(f"Total documents in vector store: {total_documents}")

    # Generate query embedding
    # query_embedding = embedding_function.embed_query(text=query)
    # # print(f"Query embedding: {query_embedding}")

    # Retrieve chunks
    chunks = vector_store.similarity_search_with_score(query=query, k=3)

    #TODO: print similarity scores for documents to get insight into the retrieval process

    return chunks
