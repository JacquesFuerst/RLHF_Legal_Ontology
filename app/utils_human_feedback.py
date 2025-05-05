import streamlit as st
import os

from fpdf import FPDF
import csv

from datetime import datetime

from weasyprint import HTML, CSS

from dotenv import load_dotenv

import ast


# Load environment variables from .env file
load_dotenv()

# def convert_markdown_to_html(markdown_text):
#     """
#     Convert markdown text to HTML.
#     """
#     html = markdown.markdown(markdown_text)
#     return html

def load_html(file_path, hours, number_of_pairs):
    """
    Load HTML content from a file and replace placeholders with actual values.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
    
    # Replace placeholders with actual values
    html_content = html_content.format(current_date= datetime.now().strftime('%d-%m-%Y'), hours=hours, number_of_pairs=number_of_pairs)
    html_content = html_content.replace("{{number_of_pairs}}", str(number_of_pairs))
    
    return html_content

# Function to handle consent submission
def submit_consent(study_information_text, informed_consent_text, name, informed_consent_pdf_path):
    # set the state for consent being given tot true
    st.session_state.consent_given = True

    # Function to create a PDF
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    full_text = study_information_text + informed_consent_text + f"Name and date: {name}, " + current_time
    
    css = CSS(os.getenv('INFORMED_CONSENT_CSS'))

    # Convert Markdown to HTML
    HTML(string=full_text).write_pdf(informed_consent_pdf_path, stylesheets=[css])




# Function to navigate to the next page
def next_page():
    st.session_state.page += 1

    # Function to navigate to the next page
def next_page_2():
    st.session_state.page_2 += 1


# Function to navigate to the previous page
def prev_page():
    st.session_state.page -= 1

    # Function to navigate to the previous page
def prev_page_2():
    st.session_state.page_2 -= 1



# Function to handle feedback submission
def submit_feedback(feedback_1, feedback_2, data, unique_id, precond_ids, prompt_configs, feedback_additional_content=None):
    """
    

    """
    # Get all relevant indices for storing the feedback
    current_index = st.session_state.current_index
    current_preconditon_index = st.session_state.current_precondition_index
    current_prompt_config_index = st.session_state.current_prompt_config_index
    current_response_index = st.session_state.current_response_index

    current_precond = precond_ids[current_preconditon_index]
    current_prompt_config = prompt_configs[current_prompt_config_index]

    # Convert string to actual tuple
    parsed = ast.literal_eval(current_prompt_config)

    # Get the first value
    config_examples = parsed[0]
    config_chain_of_thought = parsed[1]

    # Store the feedback with the proper precondition id, answer ID and prompt config ID
    # if feedback is given on additional content, store it in the feedback_additional_content field
    # if feedback_additional_content:
    #     data[current_index]['feedback_additional_content'].setfdefault(f"Prompt_config: {current_prompt_config}, answer_id: {current_response_index}", "").append(feedback_additional_content)
    # else:

    # store the feedback as needed
    print(f"keys in data: {data[current_index].keys()}")
    data[current_index]['feedback_extraction'][f"Precond_id: {current_precond}, Prompt_config: {current_prompt_config}, answer_id: {current_response_index}"] = feedback_1 
    data[current_index]['feedback_detection'][f"Precond_id: {current_precond}, Prompt_config: {current_prompt_config}, answer_id: {current_response_index}"] = feedback_2

    # Define the CSV file path
    csv_file_path = os.getenv('HUMAN_FEEDBACK_CSV') + f'_{unique_id}.csv'
    
    # Check if the CSV file already exists
    file_exists = os.path.isfile(csv_file_path)
    
    # Open the CSV file in append mode
    field_names = ['file', 
                   'frame_ID', 
                   'frame_type', 
                   'frame_text', 
                   'precondition_id', 
                   'precondition_text', 
                   'precondition_position', 
                   'response_text', 
                   'prompt_config_examples', 
                   'prompt_config_chain_of_thought', 
                   'feedback_extraction', 
                   'feedback_detection'
    ]
    
    row = {
        'file': data[current_index]['file'], 
        'frame_ID': data[current_index]['ID'], 
        'frame_type': data[current_index]['type'],
        'frame_text': data[current_index]['text'], 
        'precondition_id': current_precond,
        'precondition_text': data[current_index]['precondition_texts'][current_precond],
        'precondition_position': data[current_index]['text_positions'][current_precond],
        'response_text': data[current_index]['responses'][current_prompt_config][current_response_index],
        'prompt_config_examples': config_examples[1],
        'prompt_config_chain_of_thought': config_chain_of_thought[1],
        'feedback_extraction': feedback_1,
        'feedback_detection':feedback_2
    }
    
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=field_names, delimiter=';')
        
        # Write the header only if the file does not exist
        if not file_exists:
            writer.writeheader()
        
        # Write the data
        writer.writerow(row)
    

    # update all the indices in the correct order
    if st.session_state.current_precondition_index < len(precond_ids) - 1:
        print("We ")
        st.session_state.current_precondition_index += 1

    # elif st.session_state.current_response_index < 1 and not st.session_state.additional_content:
    #     st.session_state.additional_content = True
    
    elif st.session_state.current_response_index < 1: # and st.session_state.additional_content:
        # if the precondition index is at the end, reset it to 0
        # and increase the response index
        st.session_state.current_precondition_index = 0
        # st.session_state.additional_content = False
        st.session_state.current_response_index += 1

    elif st.session_state.current_prompt_config_index < len(prompt_configs) - 1:  
        # if the response index is at the end, reset it to 0
        # and increase the prompt config index
        st.session_state.current_response_index = 0
        st.session_state.current_precondition_index = 0
        st.session_state.current_prompt_config_index += 1

    else:
        # if the prompt config index is at the end, reset it to 0
        # and increase the data index
        # if the additional content is given, reset the data index
        st.session_state.current_response_index = 0
        st.session_state.current_precondition_index = 0
        st.session_state.current_prompt_config_index = 0
        st.session_state.current_index += 1
        # st.session_state.additional_content = False
        





# # function to create pdf
# def create_pdf(text):
#     pdf = FPDF()
#     pdf.add_page()
#     pdf.set_font("Arial", size=12)
#     pdf.multi_cell(0, 10, text)
#     return pdf