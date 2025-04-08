import streamlit as st
import os

from fpdf import FPDF
import csv

from datetime import datetime

import markdown
import pdfkit

def convert_markdown_to_html(markdown_text):
    """
    Convert markdown text to HTML.
    """
    html = markdown.markdown(markdown_text)
    return html

# Function to handle consent submission
def submit_consent(study_information_text, informed_consent_text, name, informed_consent_pdf_path):
    # set the state for consent being given tot true
    st.session_state.consent_given = True

    # Function to create a PDF
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    full_text = study_information_text + informed_consent_text + f"Name: {name}\n\n\n" + current_time

    # Convert Markdown to HTML
    html_content = convert_markdown_to_html(full_text)

    # Save the HTML content to a PDF
    pdfkit.from_string(html_content, informed_consent_pdf_path)




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
    csv_file_path = f'C:/Users/furstj/development/RAG/data/human_feedback/queries_{unique_id}.csv'
    
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



# function to create pdf
def create_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, text)
    return pdf