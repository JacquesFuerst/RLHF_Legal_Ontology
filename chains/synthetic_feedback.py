import streamlit as st
from ollama_model.llama32 import load_llm, return_model_name

import re
from datetime import datetime

import csv
import os





def extract_preconditions(response):
    """
    Extract the preconditions from the response.
    
    Parameters:
    response (str): The response to evaluate
    
    Returns:
    preconditions (list): The preconditions extracted from the response
    """
    # Regular expression to match all preconditions
    pattern = r"Precondition:\s*(.*?)(?=\n\n|$)"

    # Extract all preconditions using re.findall --> returns list of strings
    preconditions = re.findall(pattern, response, re.DOTALL)

    return preconditions




def generate_feedback_prompt(response, prompt, ground_truth):
        """
        Generate a feedback prompt for the response based on the context and ground truth. 

        Parameters:
        response (str): The response to evaluate
        prompt (str): The prompt that was given to the original model
        ground_truth (str): The ground truth data

        Returns:
        feedback_prompt (str): The feedback prompt for the response
        """

        #TODO: potentially use preconditions instead of response here

        #TODO: give the original prompt instead of the context only??


        # You are a Dutch Legal expert. 
     
        feedback_prompt = f"""

                    You are given a response to the following query: {prompt} \n \n

                    Response: {response} \n \n
                    
                    Consider whether the semantics of the extracted preconditions are consistent with the ground-truth: \n {ground_truth}.


                    For each response, evaluate how well the answer is aligned with:

                    a) extraction: to which degree the right preconditions were extracted exhaustively and correctly
                    b) detection: to which degree the position of each precondition was provided in the text (which is given in the query)

                    For both extraction and detection, provide a rating of one of the three categories:

                    1. Feedback: Accept
                    2. Feedback: Accept with minor changes
                    3. Feedback: Reject

                    Provide feedback in the following format: \n
                    Feedback_Extraction:  <Answer> \n \n
                    Feedback_Detection:  <Answer> \n \n
                    """
        
        return feedback_prompt




#TODO: potentially use other llm here, set this up after meeting with Mike...

#TODO: somehow give feedback on the page number and line number as well, maybe use the metadata for this

#TODO: is token-level accuracy needed here?

#TODO: have list of preconditions for now, check with FLINT data what the actual precondition format is


#TODO: maybe return feedback for each precondition? No, need to compare proper part of ground truth with proper preconditions





#TODO: need to define this as an actual chain, not a function, HOW?
def feedback_chain(inputs):
    """
    Evaluate the response and provide feedback based on the ground truth.

    Parameters:
    inputs (dict): Dictionary containing the response, context, and ground truth

    Returns:
    dict: Dictionary containing the feedback
    """
    response = inputs["response"]
    prompt = inputs["prompt"]
    ground_truth = inputs["ground_truth"]

    # print("response: ", response)
    # print("context: ", context)

    preconditions = extract_preconditions(response)

    #TODO: implement logic such that reject is automatically there if answer is too bad

    # for precondition in preconditions:
    #     # if jaccard_distance_strings(precondition, ground_truth) < 0.5:
    #     #     feedback = "Feedback: Reject"
    #     #     st.session_state.feedback = feedback
    #     #     return {"feedback": feedback}
    
    if 5 < 4:#jaccard_distance_strings(response, ground_truth) < 0.5: #TODO: implement logic such that reject is automatically there if answer is too bad
        feedback = "Feedback: Reject"
    else:
        feedback_llm = load_llm()
        feedback_prompt = generate_feedback_prompt(response, prompt, ground_truth)
        feedback = feedback_llm.invoke(feedback_prompt)
    
    # # Store feedback in session state
    # st.session_state.feedback = feedback
    
    return {"feedback": feedback.content}

def store_feedback(query, prompt, response, feedback):
    """
    Store the feedback data in a csv file.

    Parameters:
    query (str): The query used to retrieve the chunks
    prompt (str): The context of the response
    response (str): The response to evaluate
    feedback (str): The feedback for the response

    Returns:
    None
    """

    #TODO: change model name functions to return proper model name

    data = {
        "query": query,
        "prompt": prompt,
        "response": response,
        "feedback": feedback,
        "timestamp": datetime.now().isoformat(),
        "model_version": return_model_name(),
        "feedback_model_version": return_model_name()
    }
    
    file_path = "C:/Users/furstj/development/RAG/data/synthetic_feedback/synthetic_feedback.csv"
    
    # Check if the file exists to write headers
    file_exists = os.path.isfile(file_path)
    
    with open(file_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys(), delimiter=';')
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(data)