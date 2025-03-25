import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from chains.simple_chain import get_rag_response_and_collect_feedback


# creates a simple RAG web interface using streamlit

st.title("RAG system for information extraction")

query = st.text_input("Please enter the act for which you would like to retrieve preconditions here:")

if query:
    response, prompt = get_rag_response_and_collect_feedback(query)
    st.write("Input context window:")
    st.write(prompt)
    st.write("Response:")
    st.write(response)







