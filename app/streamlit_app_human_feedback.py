import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from app.json_handling import read_json
import streamlit as st
import uuid
from datetime import datetime

from app.utils_human_feedback import submit_consent, next_page, prev_page, submit_feedback, create_pdf


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

# Define current date
current_date = datetime.now().strftime("%d-%m-%Y")

# Define the number of prompt-answer pairs the user needs to fill in
number_of_pairs = 50 #TODO: define as actual number of pairs


# Load the JSON file
data = read_json('C:/Users/furstj/development/RAG/data/querys_and_responses/query_data.json')


informed_consent_text = f"""
            
            ### **Informed Consent:**
                    
            
            By proceeding with this study, I acknowledge that I have read and understood the study information dated {current_date}, 
            or it has been read to me. I have been able to ask questions about the study and my questions have been answered to my satisfaction.

            I consent voluntarily to be a participant in this study and understand that I can refuse to answer questions 
            and I can withdraw from the study at any time, without having to give a reason.

            I understand that taking part in the study involves rating LLM responses to legal queries.

            I understand that information I provide will be used for research purposes only and will be treated confidentially.

            I understand that personal information collected about me that can identify me, such as (e.g. my name or where I live), 
            will not be shared beyond the study team.

            I give permission for the feedback data that I provide to be archived in [name of data repository] 
            so it can be used for future research and learning.
            \n\n\n

            """
definitions = """

            ### **Definitions:**

            Within this study, you will need to evaluate the responses given by a language model 
            for extracting preconditions from a text based on the action that is provided to it. 
            Hence, it is handy to get a clearer idea what is implied by these terms:

            An ***action*** can be performed by an agent within the normative system defined by the legal document.

            A ***precondition*** describes the circumstances under which the act can be performed legally.

            An example action in the context of a library might be 'Person A is lending a book'. 
            Then, corresponding preconditions would be that they are \n

            1. A library member and \n
            2. do not have any open fines.\n

            """

study_information_text = f"""
                    
                ### **General Information:**
                
                **Study Title:** Reinforcement Learning from Human Feedback for legal ontology information extraction

                **Researcher:** Jacques Fürst, KTH - Royal Institute of Technology
                
                **Date**: {current_date}

                ### **Purpose of the Study:** 
                
                You are invited to participate in a research study about Language Model performance in extracting information from Dutch legal documents. 
                Your participation will help to train the language model at hand from your feedback.


                ### **Definitions:**

                Within this study, you will need to evaluate the responses given by a language model 
                for extracting preconditions from a text based on the action that is provided to it. 
                Hence, it is handy to get a clearer idea what is implied by these terms:

                An ***action*** can be performed by an agent within the normative system defined by the legal document.

                A ***precondition*** describes the circumstances under which the act can be performed legally.

                An example action in the context of a library might be 'Person A is lending a book'. 
                Then, corresponding preconditions would be that they are \n

                1. A library member and \n
                2. do not have any open fines.\n

                ### **Procedures:** 
                
                If you agree to participate, you will be shown {number_of_pairs} pairs of an action and its corresponding precondition(s).
                 
                For each of these pairs, the action was given to a language model as part of a prompt 
                and it was asked to return all its corresponding precondition(s) and their respective position(s) in the text. \n

                It is your task to evaluate (on a 4-point Likkert scale) how well the model performed on \n

                a) finding all the relevant preconditions in the text and \n
                b) how clear the position in the text is that the model pointed to. \n

                You will be provided with the document in which you can find the preconditions and the preconditions themselves, 
                but NOT their true position in the text. \n

                For evaluating part a) you can simply compare the preconditions the model named 
                with the ones that were provided to you (which represent the ground truth). \n

                For evaluating part b), it is your task to see whether you can find the precondition in the document with
                the information you got from the prompt and evaluate the language model's performance based on how easy it was for you to find it. \n

                This will take approximately [time required].

                ### **Voluntary Participation:** 
                
                Your participation is completely voluntary. You may choose not to participate or to withdraw at any time without any penalty 
                or having to provide any reason whatsoever.

                ### **Anonymity:** 
                
                Your responses will be completely anonymous. No personal information will be collected, and your responses cannot be traced back to you.

                ### **Risks and Benefits:** 
                
                There are no known risks associated with this study. The benefits include contributing to research that may improve
                the usage of AI in a legal context.

                ### **Contact Information:** 
                
                If you have any questions about this study, please contact jfurst@kth.se."""

informed_consent_pdf_path = f'C:/Users/furstj/development/RAG/data/informed_consent/informed_consent_{unique_id}.pdf'

ground_truth = "NEEDS TO BE ADDED"

##################################################################################################################################################





# Display the consent text and button if consent has not been given

#TODO: add name of repository and add time it takes to complete study

#TODO: insert what a precondition is and what an action is in this context --> make accessible at any moment during the study
if not st.session_state.consent_given:

    # Display the current page
    if st.session_state.page == 1:
        
        st.markdown("# Study Information")

        ## Button to navigate to the informed consent page
        if st.button("Informed Consent➡️"):
            next_page()
            st.rerun()

        # Display the study information text
        st.markdown(study_information_text)
        
        ## Button to navigate to the informed consent page
        if st.button("Informed Consent➡️ "):
            next_page()
            st.rerun()

        

    elif st.session_state.page == 2:

        # Button to navigate back to the study information page
        if st.button("⬅️ Study Information"):
            prev_page()
            st.rerun()

        # Display the informed consent text
        st.markdown(informed_consent_text)
        
        #field for participant to enter name and surname
        st.markdown("### **Participant Information:**")
        name = st.text_area("Please enter your full name:", value="", max_chars=100, height=100)
        
        # Radio button to give consent
        consent = st.radio(

            "Do you consent with all of the above?"
            ,
            ("I do not agree", "I agree")
        )

        if st.button("Submit Consent"):
            if consent == "I agree":
                submit_consent(study_information_text, informed_consent_text, name, informed_consent_pdf_path)

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
        st.markdown(definitions)

        # Display the question and answer
        st.markdown(f"""
                    ### **Query:** 
                     {question}""")
        st.write(f"""
                    ### **Answer:** 
                     {answer}""")
        
        st.write(f"""
                    ### **Ground Truth:**
                    {ground_truth}
        """)

        # Create Likert scales for feedback
        feedback_1 = st.radio(
            "How well were the preconditions extracted?",
            ("Very dissatisfied", "Dissatisfied", "Satisfied", "Very satisfied")
        )

        feedback_2 = st.radio(
            "How clear is the position in the text the model pointed to?",
            ("Very unclear", "Unclear", "Clear", "Very clear")
        )

        # Save the feedback and move to the next question
        if st.button("Submit Feedback"):
            submit_feedback(feedback_1, feedback_2, data, unique_id)
            print("feedback given")
            st.rerun()  # Trigger a refresh
    else:
        st.write("You have evaluated all queries. Thank you for your feedback!")