from embeddings.vector_store import store_embeddings
from embeddings.data_extraction import extract_text

def main():
    # Define the path to your data folder
    folder_path = "C:/Users/furstj/development/RAG/data/text/preconditions.pdf"#"../data/test_document_future_of_AI.pdf"
    
    # Load data from the folder
    docs = extract_text(folder_path)
    
    # Store embeddings in the vector database
    store_embeddings(docs)
    
    # Additional initialization or processing can go here
    print("Embeddings stored successfully!")

if __name__ == "__main__":
    main()


