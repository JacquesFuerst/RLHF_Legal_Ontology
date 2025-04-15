from chains.simple_chain import get_rag_response
from embeddings.data_extraction import extract_text
from embeddings.vector_store import store_embeddings

import json

def generate_answers(ground_truth_file, knowledge_base_file):
    """
    Function to generate answers from the deployed model. First generate the correct embeddings in the knowledge base for the right file.

    Args:
    ground_truth_file (str): path to ground truth file
    knowledge_base_file (str): path to knowledge base file

    Returns:
    None
    """
    
    # Load data from the folder
    docs = extract_text(knowledge_base_file)
    
    # Store embeddings in the vector database
    store_embeddings(docs)

    # Define prompt conditions for each answer
    prompt_conditions_1 = None
    prompt_conditions_2 = None
    prompt_conditions_3 = None

    with open(ground_truth_file, 'r') as file:
        data = json.load(file)

    for act in data:
        response_1 = get_rag_response(act['text'], prompt_conditions_1)
        response_2 = get_rag_response(act['text'], prompt_conditions_2)
        response_3 = get_rag_response(act['text'], prompt_conditions_3)

        act['responses'] = {prompt_conditions_1: response_1,
                            prompt_conditions_2: response_2,
                            prompt_conditions_3: response_3
                            }
        
    with open(f'acts_and_responses_{knowledge_base_file}', 'w') as file:
        json.dump(data, file, indent=4)

    



    

    