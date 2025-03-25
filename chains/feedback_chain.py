from ollama_model.llama32 import load_llm
from chains.retriever import retrieve_chunks
import streamlit as st
from chains.simple_chain import generate_prompt 

from langchain.chains.sequential import SequentialChain
from chains.feedback_chain import response_chain
from chains.synthetic_feedback import feedback_chain


def response_chain(inputs):
    query = inputs["query"]
    llm = load_llm()
    retrieved_chunks = retrieve_chunks(query)
    
    context = "\n".join([f"Page {chunk.metadata['page_number']}: {chunk.page_content}" for chunk in retrieved_chunks])
    prompt = generate_prompt(query, context)
    response = llm.invoke(prompt)
    
    # Store response and context in session state
    st.session_state.response = response
    st.session_state.context = context
    
    return {"response": response, "context": context}


chain = SequentialChain(
                chains=[response_chain, feedback_chain],
                input_variables=["query", "ground_truth"]
                )


# Example function to get RAG response and collect feedback
def get_rag_response_and_collect_feedback(query, ground_truth):
    inputs = {"query": query, "ground_truth": ground_truth}
    chain(inputs)
    return st.session_state.response, st.session_state.context, st.session_state.feedback