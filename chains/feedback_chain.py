from ollama_model.llama32 import load_llm
from chains.retriever import retrieve_chunks
import streamlit as st
from chains.simple_chain import generate_prompt 

from langchain.chains.sequential import SequentialChain
from chains.synthetic_feedback import feedback_chain



def response_chain(inputs):
    query = inputs["query"]
    llm = load_llm()
    retrieved_chunks = retrieve_chunks(query)
    
    context = "\n".join([f"Page {chunk[0].metadata['page_number']}: {chunk[0].page_content}" for chunk in retrieved_chunks])
    prompt = generate_prompt(query, context)
    response = llm.invoke(prompt)
    
#     # Store response and context in session state
#     st.session_state.response = response.content
#     st.session_state.context = context

#     print("response: ", response.content)
#     print("context: ", context)
    
    return {"response": response.content, "context": context}


#TODO: cannot use functions as input variables for the chain, need to define them as chains somehow

# chain = SequentialChain(
#                 chains=[response_chain, feedback_chain],
#                 input_variables=["query", "ground_truth"]
#                 )

# could just scrap the sequentialchain approach altogether


# Example function to get RAG response and collect feedback
def get_rag_response_and_collect_feedback(query, ground_truth):

        inputs = {"query": query, "ground_truth": ground_truth}
        inputs.update(response_chain(inputs))

        inputs.update(feedback_chain(inputs))
        return inputs['response'], inputs['context'], inputs['feedback']