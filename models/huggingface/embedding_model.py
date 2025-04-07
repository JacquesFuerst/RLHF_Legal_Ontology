from sentence_transformers import SentenceTransformer 
from langchain_huggingface import HuggingFaceEmbeddings
import torch

from torch import nn
from device_config import get_device
import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# load the relevant devices available on the server
print("Available devices: ", os.getenv("AVAILABLE_DEVICES"))
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("AVAILABLE_DEVICES")
print("Number of GPUs available: ", torch.cuda.device_count())


class EmbeddingModel:
    """
    A class to handle embedding queries using a SentenceTransformer model.
    """
    def __init__(self, model_name):
        """
        Initialize the EmbeddingFunction with a SentenceTransformer model.
        Parameters:
        model_name (str): The name of the SentenceTransformer model to use.
        """
        self.model_name = model_name
        self.model = nn.DataParallel(SentenceTransformer(self.model_name).eval(), device_ids=[1,2]).to(get_device())
    
    def embed_query(self, query):
        """
        Embed the query using the SentenceTransformer model.
        Parameters:
        query (str): The query to embed.
        Returns:
        torch.Tensor: The embedded query.
        """

        with torch.no_grad():
            return self.model.module.encode(query) # module is needed due to dataparallel
        

    def embed_documents(self, docs):
        """
        Embed a list of documents using the SentenceTransformer model.
        Parameters:
        docs (list): A list of documents to embed.
        Returns:
        list: A list of embedded documents.
        """

        with torch.no_grad():
            return self.model.module.encode(docs) # module is needed due to dataparallel