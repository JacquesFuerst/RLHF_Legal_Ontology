from ollama_model.llama32 import load_llm
from chains.retriever import retrieve_chunks
import streamlit as st
from chains.simple_chain import generate_prompt 

from chains.synthetic_feedback import feedback_chain, store_feedback



def response_chain(inputs):
    query = inputs["query"]
    llm = load_llm()
    retrieved_chunks = retrieve_chunks(query)
    
    context = "\n".join([f"Page {chunk[0].metadata['page_number']}: {chunk[0].page_content}" for chunk in retrieved_chunks])
    prompt = generate_prompt(query, context)
    response = llm.invoke(prompt)

#     print("response: ", response.content)
#     print("context: ", context)

    return_dict = {"response": response.content, "prompt": prompt}
    return return_dict


# Example function to get RAG response and collect feedback
def get_rag_response_and_collect_feedback(query, ground_truth):

        inputs = {"query": query, "ground_truth": ground_truth}

        # get response and context
        inputs.update(response_chain(inputs))

        # get feedback
        inputs.update(feedback_chain(inputs))

        # store feedback in a JSON file
        store_feedback(query, inputs["prompt"], inputs["response"], inputs["feedback"])

        return inputs['response'], inputs['prompt'], inputs['feedback']