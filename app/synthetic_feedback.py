import json
from datetime import datetime
from ollama_model.llama32 import load_llm, return_model_name

import re



def jaccard_distance_strings(str1, str2):
    """
    Calculate the Jaccard distance between two strings.
    
    Parameters:
    str1 (str): First string
    str2 (str): Second string
    
    Returns:
    float: Jaccard distance
    """
    set1 = set(str1.split())
    set2 = set(str2.split())
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    jaccard_similarity = intersection / union
    jaccard_distance = 1 - jaccard_similarity
    
    return jaccard_distance


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




def generate_feedback_prompt(response, context, ground_truth):
        """
        Generate a feedback prompt for the response based on the context and ground truth. 

        Parameters:
        response (str): The response to evaluate
        context (str): The context in which the response was generated
        ground_truth (str): The ground truth data

        Returns:
        feedback_prompt (str): The feedback prompt for the response
        """

        #TODO: potentially use preconditions instead of response here
     
        feedback_prompt = f"""
                    Evaluate the following response based on the context provided: \n {context} \n \n 
                    Response: {response} \n \n
                    Consider whether the semantics of the extracted preconditions are consistent with the ground-truth: \n {ground_truth}.
                    Provide feedback in the following format: \n
                    Feedback:  Accept/Reject \n \n
                    """
        
        return feedback_prompt


def feedback_chain(response, context, ground_truth):
        """
        Return a feedback prompt for the response based on the context and ground truth. 
        Give an automatic reject if the response is too far from the ground truth (by Jaccard distance).

        Parameters:
        response (str): The response to evaluate
        context (str): The context in which the response was generated
        ground_truth (str): The ground truth response

        Returns:
        str: The response to evaluate
        str: The context of the response
        str: The feedback for the response
        """

        #TODO: potentially use other llm here, set this up after meeting with Mike...

        #TODO: somehow give feedback on the page number and line number as well, maybe use the metadata for this

        #TODO: is token-level accuracy needed here?

        #TODO: have list of preconditions for now, check with FLINT data what the actual precondition format is
        

        #TODO: maybe return feedback for each precondition? No, need to compare proper part of ground truth with proper preconditions

        precondtions = extract_preconditions(response)

        # if the response is too far away from the ground truth, reject it

        for precondition in precondtions:
            if jaccard_distance_strings(precondition, ground_truth) < 0.5:
                feedback = "Feedback: Reject"
                return feedback
        if jaccard_distance_strings(response, ground_truth) < 0.5:
            feedback = "Feedback: Reject"
        else:
            feedback_llm = load_llm()
            feedback_prompt = generate_feedback_prompt(response, context, ground_truth)
            feedback = feedback_llm.invoke(feedback_prompt)

        return response, context, feedback



def store_feedback(query, context, response, feedback):
    """
    Store the feedback data in a JSON file.

    Parameters:
    query (str): The query used to retrieve the chunks
    context (str): The context of the response
    response (str): The response to evaluate
    feedback (str): The feedback for the response

    Returns:
    None
    """
    data = {
        "query": query,
        "context": context,
        "response": response,
        "feedback": feedback,
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model_version": return_model_name(),
            "feedback_model_version": return_model_name()
        }
    }
    
    with open("C:/Users/furstj/development/RAG/data/feedback/synthetic_feedback.json", "a") as f:
        json.dump(data, f)
        f.write("\n")