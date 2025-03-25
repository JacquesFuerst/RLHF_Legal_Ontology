from ollama_model.llama32 import load_llm
from chains.retriever import retrieve_chunks

from chains.synthetic_feedback import feedback_chain, store_feedback
from langchain.chains.sequential import SequentialChain

import streamlit as st

from chains.simple_chain import generate_prompt 


############# Future implementation together with feedback chain ###################


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

def response_chain(query):
        """"
        Get the response from the response model based on the query.
        """
        llm = load_llm()
        retrieved_chunks = retrieve_chunks(query)
        
        # extract chunk_page content attribute from each chunk
        context = "\n".join([f"Page {chunk.metadata['page_number']}: {chunk.page_content}" for chunk in retrieved_chunks])

        #TODO: use chain.invoke here later?

        prompt = generate_prompt(query, context)
        response = llm.invoke(prompt)
        return {"response": response, "context": context}


def get_rag_response_and_collect_feedback(query):
        """
        
        """
        # Create the sequential chain
        chain = SequentialChain(chains=[response_chain, feedback_chain])

        # Invoke the chain
        response, context, feedback = chain.invoke(query)

        #store the feedback data
        store_feedback(query, context, response, feedback)
        
        return response, context

#####################################################################################################################################