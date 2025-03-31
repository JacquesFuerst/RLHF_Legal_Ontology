import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from chains.feedback_chain import get_rag_response_and_collect_feedback


# creates a simple RAG web interface using streamlit

st.title("RAG system for information extraction")

query = st.text_input("Please enter the act for which you would like to retrieve preconditions here:")

ground_truth = st.text_input("Please enter the act's ground truth:")

#TODO: maybe add buttn

if query and ground_truth:
    response, context, feedback = get_rag_response_and_collect_feedback(query, ground_truth)
    st.write("Input context window:")
    st.write(context)
    st.write("RAG response:")
    st.write(response)
    st.write("Feedback:")
    st.write(feedback)







