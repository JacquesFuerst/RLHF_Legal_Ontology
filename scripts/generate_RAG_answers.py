import json
import sys

import os
# from embeddings.data_extraction import extract_text
# from embeddings.vector_store import store_embeddings

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chains.simple_chain import get_rag_response
from models.huggingface.generator import Generator

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# load the relevant devices available on the server
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("AVAILABLE_DEVICES")

# Enable expandable CUDA segments
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def generate_answers(ground_truth_file, knowledge_base_file):
    """
    Function to generate answers from the deployed model. First generate the correct embeddings in the knowledge base for the right file.

    Args:
    ground_truth_file (str): path to ground truth file
    knowledge_base_file (str): path to knowledge base file

    Returns:
    None
    """

    # load LLM and tokenizer
    generator = Generator(os.getenv("GENERATION_MODEL_NAME"))
    llm, tokenizer = generator.load_llm_and_tokenizer()
    embed_func = None
    act_bool = False

    model_answers = os.getenv("MODEL_ANSWERS")
    

    # Define prompt conditions for each answer
    prompt_conditions_1 = {'include_examples': True, 'include_chain_of_thought': True}
    prompt_conditions_3 = {'include_examples': False, 'include_chain_of_thought': True}

    with open(ground_truth_file, 'r') as file:
        data = json.load(file)

    for act in data:

        number_preconditions = len(act['precondition_texts']) # act=True, number_preconditions=0, prompt_conditions=None
        print(f"Number of preconditions: {number_preconditions}")
        response_1_1 = get_rag_response(act['text'], llm, tokenizer, embed_func, act=act_bool, number_preconditions=number_preconditions, prompt_conditions=prompt_conditions_1)
        print(f"Response 1_1: {response_1_1}")
        response_3_1 = get_rag_response(act['text'], llm, tokenizer, embed_func, act=act_bool, number_preconditions=number_preconditions, prompt_conditions=prompt_conditions_3)
        print(f"Response 3_1: {response_3_1}")

        # Store the responses in the act dictionary
        act['responses'] = {str((tuple(prompt_conditions_1.items()))): [response_1_1],
                            str((tuple(prompt_conditions_3.items()))): [response_3_1],
                            }
        
    with open(model_answers, 'w') as file:
        json.dump(data, file, indent=4)



ground_truth_file = os.getenv("GROUND_TRUTH_FILE")
knowledge_base_file = os.getenv("RAG_KB_PATH")

generate_answers(ground_truth_file, knowledge_base_file)

    



    

    