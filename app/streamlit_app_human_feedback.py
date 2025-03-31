import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from app.json_handling import read_json
import streamlit as st
import uuid


import csv
import os





############## Helper functions ###################

# Function to handle consent submission
def submit_consent():
    st.session_state.consent_given = True



# Function to handle feedback submission
def submit_feedback(feedback_1, feedback_2):
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


##################################################


# Check if the unique identifier is already in session state
if 'unique_id' not in st.session_state:
    st.session_state.unique_id = str(uuid.uuid4())

# Access the unique identifier
unique_id = st.session_state.unique_id


# Load the JSON file
data = read_json('C:/Users/furstj/development/RAG/data/querys_and_responses/query_data.json')


# Initialize session state
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'consent_given' not in st.session_state:
    st.session_state.consent_given = False

# Display the consent text and button if consent has not been given
if not st.session_state.consent_given:
    st.write("**Consent Form**")
    st.write("""
    Please read the following consent form carefully before providing your feedback.

    By participating in this feedback process, you agree to the collection and use of your feedback for research and development purposes. Your responses will be kept confidential and used to improve our services.
    """)

    # Consent radio button
    consent = st.radio(
        "Do you consent to provide feedback?",
        ("No", "Yes")
    )

    if st.button("Submit Consent"):
        if consent == "Yes":
            submit_consent()
            print("consent given")
            st.rerun()  # Trigger a refresh
        else:
            st.write("Please provide your consent to proceed.")
else:
    # Get the current question and answer
    current_index = st.session_state.current_index
    if current_index < len(data):
        question = data[current_index].get('query')
        answer = data[current_index].get('answer')

        # Display the question and answer
        st.write(f"**Query:** {question}")
        st.write(f"**Answer:** {answer}")

        # Create Likert scales for feedback
        feedback_1 = st.radio(
            "How well were the preconditions extracted?",
            ("Very dissatisfied", "Dissatisfied", "Neutral", "Satisfied", "Very satisfied")
        )

        feedback_2 = st.radio(
            "How clear is the position in the text the model pointed to?",
            ("Very unclear", "Unclear", "Neutral", "Clear", "Very clear")
        )

        # Save the feedback and move to the next question
        if st.button("Submit Feedback"):
            submit_feedback(feedback_1, feedback_2)
            print("feedback given")
            st.rerun()  # Trigger a refresh
    else:
        st.write("You have evaluated all queries. Thank you for your feedback!")