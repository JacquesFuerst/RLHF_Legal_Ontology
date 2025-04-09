import streamlit as st
import os

from fpdf import FPDF
import csv

from datetime import datetime

from weasyprint import HTML, CSS

from dotenv import load_dotenv


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


# Function to navigate to the previous page
def prev_page():
    st.session_state.page -= 1



# Function to handle feedback submission
def submit_feedback(feedback_1, feedback_2, data, unique_id):
    current_index = st.session_state.current_index
    data[current_index]['feedback_extraction'] = feedback_1
    data[current_index]['feedback_detection'] = feedback_2
    
    # Define the CSV file path
    csv_file_path = os.getenv('HUMAN_FEEDBACK_CSV') + f'_{unique_id}.csv'
    
    # Check if the CSV file already exists
    file_exists = os.path.isfile(csv_file_path)
    
    # Open the CSV file in append mode
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data[current_index].keys(), delimiter=';')
        
        # Write the header only if the file does not exist
        if not file_exists:
            writer.writeheader()
        
        # Write the data
        writer.writerow(data[current_index])
    
    st.session_state.current_index += 1



# # function to create pdf
# def create_pdf(text):
#     pdf = FPDF()
#     pdf.add_page()
#     pdf.set_font("Arial", size=12)
#     pdf.multi_cell(0, 10, text)
#     return pdf