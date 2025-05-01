from langchain_chroma import Chroma
import os
import torch
from dotenv import load_dotenv
import fitz


# Load environment variables from .env file
load_dotenv()

# load the relevant devices available on the server
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("AVAILABLE_DEVICES")


def retrieve_chunks(query, embed_func):
    """
    When sending in the prompt, the system should return the most relevant text chunks from the vector database.

    Parameters:
    query (str): The query for the vector store

    Returns:
    list: The most relevant text chunks
    """

    
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
    chunks = vector_store.similarity_search_with_score(query=query, k=5)

    #TODO: print similarity scores for documents to get insight into the retrieval process

    return chunks

def get_whole_doc():
    """
    Retrieve the whole reduced document to reduce memory usage whilst loading the document.
    """

    # Open the PDF document
    doc = fitz.open(os.getenv("RAG_KB_PATH"))

    # Extract and concatenate text from all pages
    all_text = ""
    for page in doc:
        all_text += page.get_text()

    # Optionally, close the document
    doc.close()

    return all_text
