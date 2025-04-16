from sentence_transformers import SentenceTransformer 
from langchain_huggingface import HuggingFaceEmbeddings
import torch

from torch import nn
from device_config import get_device
import os

import streamlit as st

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# load the relevant devices available on the server
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("AVAILABLE_DEVICES")

# Enable expandable CUDA segments
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# print(f"Using devices: {torch.cuda.device_count()}")

# Clear CUDA memory
import gc

torch.cuda.empty_cache()
gc.collect()

# # initialize process group for nn.DistributedDataParallel

# import torch.distributed as dist

# dist.init_process_group(backend='nccl')  # or 'gloo' for CPU



class EmbeddingModel:
    """
    A class to handle embedding queries using a SentenceTransformer model.
    """
    # @st.cache_resource
    def __init__(self, model_name):
        """
        Initialize the EmbeddingFunction with a SentenceTransformer model.
        Parameters:
        model_name (str): The name of the SentenceTransformer model to use.
        """
        self.model_name = model_name
        self.model = nn.parallel.DataParallel(SentenceTransformer(self.model_name, trust_remote_code=True).eval(), device_ids=[0,1,2,3]).to(get_device())
    
    def embed_query(self, query):
        """
        Embed the query using the SentenceTransformer model.
        Parameters:
        query (str): The query to embed.
        Returns:
        torch.Tensor: The embedded query.
        """

        with torch.no_grad():
            return self.model.module.encode(query, batch_size=4) # module is needed due to dataparallel
        

    def embed_documents(self, docs):
        """
        Embed a list of documents using the SentenceTransformer model.
        Parameters:
        docs (list): A list of documents to embed.
        Returns:
        list: A list of embedded documents.
        """

        with torch.no_grad():
            devices = os.getenv("AVAILABLE_DEVICES")
            print(f"Available devices: {devices}")
            return self.model.module.encode(docs, batch_size=4) # module is needed due to dataparallel