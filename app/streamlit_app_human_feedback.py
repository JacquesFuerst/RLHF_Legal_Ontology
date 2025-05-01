import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from app.json_handling import read_json
import streamlit as st
import uuid

from app.utils_human_feedback import submit_consent, next_page, prev_page, submit_feedback, load_html

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


##################### Streamlit app session states #########################

# Check if the unique identifier is already in session state
if 'unique_id' not in st.session_state:
    st.session_state.unique_id = str(uuid.uuid4())

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 1

# Initialize session state
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'consent_given' not in st.session_state:
    st.session_state.consent_given = False

##############################################################################



############### Variables used on the page #######################

# Access the unique identifier
unique_id = st.session_state.unique_id

# Define the number of prompt-answer pairs the user needs to fill in
number_of_pairs = 150 

# amounbt of hours study will take
hours = 2

# Load the HTML files
definitions = load_html(os.getenv('DEFINITIONS_HTML_PATH'), hours, number_of_pairs)
informed_consent = load_html(os.getenv('INFORMED_CONSENT_HTML_PATH'), hours, number_of_pairs)
study_information = load_html(os.getenv('STUDY_INFORMATION_HTML_PATH'), hours, number_of_pairs)


# Load the JSON file
data = read_json(os.getenv('QUERYS_AND_RESPONSES'))

# informed consent pdf path
informed_consent_pdf_path = os.getenv('INFORMED_CONSENT_STORAGE_PATH') + f'_{unique_id}.pdf'

# the ground truth
ground_truth = "NEEDS TO BE ADDED SOON"

##################################################################################################################################################


# Display the consent text and button if consent has not been given
if not st.session_state.consent_given:

    # Display the current page
    if st.session_state.page == 1:
        
        st.markdown("# Studie Informatie")

        ## Button to navigate to the informed consent page
        if st.button("Informed Consent➡️"):
            next_page()
            st.rerun()

        # Display the study information text
        st.markdown(study_information, unsafe_allow_html=True)
        
        ## Button to navigate to the informed consent page
        if st.button("Informed Consent➡️ "):
            next_page()
            st.rerun()

        

    elif st.session_state.page == 2:

        # Button to navigate back to the study information page
        if st.button("⬅️ Studie Informatie"):
            prev_page()
            st.rerun()

        # Display the informed consent text
        st.markdown(informed_consent, unsafe_allow_html=True)
        
        #field for participant to enter name and surname
        st.markdown("### **Participant Informatie:**")
        name = st.text_area("Voer uw volledige naam in:", value="", max_chars=100, height=100)
        
        # Radio button to give consent

        st.markdown(
            "### Stemt u in met al het bovenstaande?"
        )
        consent = st.radio(

            label="Antwoord:"
            ,
            options=("Ik ga niet akkoord", "Ik ga akkoord")
        )

        if st.button("Submit Consent"):
            if consent == "Ik ga akkoord":
                submit_consent(study_information, informed_consent, name, informed_consent_pdf_path)

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

        #Display the definition of and action and a precondition
        st.markdown(definitions, unsafe_allow_html=True)

        # Display the question and answer
        st.markdown(f"""
                    ### **Query:** 
                     {question}""")
        st.write(f"""
                    ### **Antwoord:** 
                     {answer}""")
        
        st.write(f"""
                    ### **Ground Truth:**
                    {ground_truth}
        """)

        # Create Likert scales for feedback

        #TODO: add proper questions and change radion button labels
        
        st.markdown(
            "### In welke mate is de preconditie door het model geëxtraheerd?"
        )

        feedback_1 = st.radio(
            "",
            ("Volledig fout", "Deels fout", "Grotendeels correct", "Volledig correct")
        )

        st.markdown(
            "### Hoe duidelijk is de positie van de preconditie in de tekst van wat het model heeft weergegeven?"
        )

        feedback_2 = st.radio(
            "",
            ("Helemaal niet duidelijk", "Niet duidelijk", "Duidelijk", "Zeer duidelijk")
        )

        # Save the feedback and move to the next question
        if st.button("Submit Feedback"):
            submit_feedback(feedback_1, feedback_2, data, unique_id)
            print("feedback given")
            st.rerun()  # Trigger a refresh
    else:
        st.write("Je hebt alle vragen geëvalueerd. Hartelijk dank voor je feedback!")