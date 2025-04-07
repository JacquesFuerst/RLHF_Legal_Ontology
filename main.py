from embeddings.vector_store import store_embeddings
from embeddings.data_extraction import extract_text
import os
import torch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# load the relevant devices available on the server
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("AVAILABLE_DEVICES")

def main():
    # Define the path to your data folder
    folder_path = os.getenv("RAG_KB_PATH")#"../data/test_document_future_of_AI.pdf"
    
    # Load data from the folder
    docs = extract_text(folder_path)
    
    # Store embeddings in the vector database
    store_embeddings(docs)
    
    # Additional initialization or processing can go here
    print("Embeddings stored successfully!")

if __name__ == "__main__":
    main()


