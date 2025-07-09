#TODO: add all prompts, ground truths in proper format to use in RL loop

import json
import sys

import os
import fitz
import csv
# from embeddings.data_extraction import extract_text
# from embeddings.vector_store import store_embeddings

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chains.simple_chain import generate_prompt_act, generate_prompt_fact

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# load the relevant devices available on the server
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("AVAILABLE_DEVICES")

# Enable expandable CUDA segments
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

json_file = os.getenv("SYNTHETIC_DATA_FILE")

def get_whole_pdf_text(doc):
    """
    Retrieve the whole reduced document to reduce memory usage whilst loading the document.
    """

    print(doc)

    if 'Participatiewet' in doc:
        pdf = os.getenv("KB_FILE_1")
        print("We use this")
    elif 'Vw' in doc:
        pdf = os.getenv("KB_FILE_2")
    else:
        pdf = os.getenv("KB_FILE_3")
    # Open the PDF document
    pdf_text = fitz.open(pdf)

    # Extract and concatenate text from all pages
    all_text = ""
    for page in pdf_text:
        all_text += page.get_text()

    # Optionally, close the document
    pdf_text.close()

    return all_text

def generate_prompts(file):
    """
    Function to generate answers from the deployed model. First generate the correct embeddings in the knowledge base for the right file.

    Args:
    ground_truth_file (str): path to ground truth file
    knowledge_base_file (str): path to knowledge base file

    Returns:
    None
    """

    prompt_conditions_3 = {'include_examples': False, 'include_chain_of_thought': True}

    with open(file, 'r') as file:
        data = json.load(file)

    for datapoint in data:
        precon_texts = datapoint["precondition_texts"]
        text_positions = datapoint["text_positions"]
        type = datapoint["type"]
        text = datapoint["text"]
        doc = datapoint["file"]
        context = get_whole_pdf_text(doc)
        number_preconditions = len(precon_texts)

        if type == "fact":
            prompt = generate_prompt_fact(text, context, prompt_conditions_3, number_preconditions=number_preconditions)
        else:
            prompt = generate_prompt_act(text, context, prompt_conditions_3, number_preconditions=number_preconditions)

    
        #TODO: write code to set up csv file...

        # Define the CSV file path
        prompt_data_csv = os.getenv('PROMPT_DATASET_CSV')

        # Check if the CSV file already exists
        file_exists = os.path.isfile(prompt_data_csv)

        # Define the field names for the CSV file
        field_names = ['prompt', 
                    'precondition_texts', 
                    'precondition_positions', 
        ]

        row = {
                        'prompt': prompt, 
                        'precondition_texts': precon_texts,
                        'precondition_positions': text_positions
                    }
        
        with open(prompt_data_csv, mode='a', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=field_names, delimiter=';')
                    
                    # Write the header only if the file does not exist
                    if not file_exists:
                        writer.writeheader()
                    
                    # Write the data
                    writer.writerow(row)


generate_prompts(json_file)


