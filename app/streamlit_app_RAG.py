import sys
import torch
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# load the relevant devices available on the server
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("AVAILABLE_DEVICES")

# Enable expandable CUDA segments
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import streamlit as st
from chains.simple_chain import get_rag_response

# Clear CUDA memory
import gc

torch.cuda.empty_cache()
gc.collect()




# creates a simple RAG web interface using streamlit

st.title("RAG system for information extraction")

query = st.text_input("Please enter the act for which you would like to retrieve preconditions here:")

if query:
    prompt_conditions = {'include_examples': True, 'include_chain_of_thought': True}
    response, prompt = get_rag_response(query, prompt_conditions)
    st.write("Input context window:")
    st.write(prompt)
    st.write("Response:")
    st.write(response)







