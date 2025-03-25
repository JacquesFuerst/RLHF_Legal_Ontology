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
    embedding_function = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = Chroma(persist_directory="./vector_db", 
                          embedding_function=embedding_function)
    
    # Get the total number of documents in the vector store
    total_documents = vector_store._collection.count()
    # print(f"Total documents in vector store: {total_documents}")

    # Generate query embedding
    query_embedding = embedding_function.embed_query(text=query)
    # print(f"Query embedding: {query_embedding}")

    # Retrieve chunks
    chunks = vector_store.similarity_search_with_score(query=query, k=3)
    # print(f"Retrieved {len(chunks)} chunks")
    
    # # Debugging: Print retrieved chunks
    # for i, chunk in enumerate(chunks):
    #     print(f"Chunk {i+1}: {chunk}")

    #TODO: print similarity scores for documents to get insight into the retrieval process

    return chunks
