from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


def retrieve_chunks(query):
    """
    When sending in the prompt, the system should return the most relevant text chunks from the vector database.

    Parameters:
    query (str): The query for the vector store

    Returns:
    list: The most relevant text chunks
    """
    # Instantiate the OllamaEmbeddings model
    embedding_function = OllamaEmbeddings(model="llama3.2")
    vector_store = Chroma(persist_directory="./vector_db", 
                          embedding_function=embedding_function)
    
    # Get the total number of documents in the vector store
    # total_documents = vector_store._collection.count()
    # print(f"Total documents in vector store: {total_documents}")

    
    chunks = vector_store.similarity_search(query, k=3) #TODO: how many chunks to retrieve?
    # print(f"""Retrieved {chunks} chunks""")
    return chunks
