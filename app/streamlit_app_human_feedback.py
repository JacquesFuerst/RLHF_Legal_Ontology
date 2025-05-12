import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from app.json_handling import read_json
import streamlit as st
import uuid

from app.utils_human_feedback import submit_consent, next_page, prev_page, next_page_2, prev_page_2, submit_feedback, load_html

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

if 'page_2' not in st.session_state:
    st.session_state.page_2 = 2 # be on page 2 by default since this is the feedback page

# Initialize current data index
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0

# Initialize current precondition index
if 'current_precondition_index' not in st.session_state:
    st.session_state.current_precondition_index = 0

#initialize current response index
if 'current_response_index' not in st.session_state:
    st.session_state.current_response_index = 0

if 'current_prompt_config_index' not in st.session_state:
    st.session_state.current_prompt_config_index = 0

if 'additional_content' not in st.session_state:
    st.session_state.additional_content = False

# Check if the consent has been given
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
data = read_json(os.getenv('DATA_FILE_1'))

# informed consent pdf path
informed_consent_pdf_path = os.getenv('INFORMED_CONSENT_STORAGE_PATH') + f'_{unique_id}.pdf'

# the ground truth
# ground_truth = read_json(os.gentenv('GROUND_TRUTH'))


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

                print("consent gegeven")
                st.rerun()  # Trigger a refresh
            else:
                st.write("Geef consent om door te gaan.")

        
    
    
else:

    # Get the current question and answer
    current_index = st.session_state.current_index

    # Get the current precondition index
    current_precondition_index = st.session_state.current_precondition_index

    # Get the current prompt config index
    current_prompt_config_index = st.session_state.current_prompt_config_index

    # Get the current response index
    current_response_index = st.session_state.current_response_index

    if current_index < len(data):

        
        print("Data keys: ", data[current_index].keys())

        #TODO: somehow make sure that you iterate over the answers and for each answer the preconditions and their locaions...
        # need to set this in streamlit session state??
        frame = data[current_index].get('text')
        responses_dict = data[current_index].get('responses')
        precondition_text_dict = data[current_index].get('precondition_texts')
        precondition_position_dict = data[current_index].get('text_positions')

        precond_ids = list(precondition_text_dict.keys())
        precondition_id = precond_ids[current_precondition_index]
        precondition_text = precondition_text_dict[precondition_id]
        precondition_position = precondition_position_dict[precondition_id]

        prompt_configs = list(responses_dict.keys())
        prompt_config_id = prompt_configs[st.session_state.current_prompt_config_index]
        responses = responses_dict[prompt_config_id]

        # TODO: cahnge the way response index is handled here, need to get both answers for each fact/act
        current_response = responses[current_response_index]


        # Inject CSS to prevent scrolling
        st.markdown("""
            <style>
            .no-scroll {
                
                white-space: pre-wrap;  /* Wraps text */
                word-wrap: break-word;  /* Breaks long words */
                overflow-wrap: break-word;   /* Ensures wrapping in all browsers */
                overflow: visible !important;

            }
            </style>
        """, unsafe_allow_html=True)



        if st.session_state.page_2 == 2:
            print("We are on page 2")
            # Button to navigate back to the study information page
            if st.button("⬅️ Definities"):
                prev_page_2()
                st.rerun()


            print(f"current indices: act: {current_index}, precond: {current_precondition_index}, response: {current_response_index}, prompt_config: {st.session_state.current_prompt_config_index}, additional_content: {st.session_state.additional_content}")
            # display act/fact and the answer, also theground truth for the preocndition
            if data[current_index].get('type') == 'act':
                
                st.markdown(f"""
                            ### **Act:** 
                            {frame}""")
                
                st.markdown("### **Model acts en precondities:**")

                answer_text = f"""<div class="no-scroll"; style="white-space: normal; word-wrap: break-word;">
                                        <p>{current_response}</p>
                                    </div>"""

                # replace new line characters with <br> tags for HTML rendering
                html_text = answer_text.replace('\n', '<br>')

                st.markdown(html_text, unsafe_allow_html=True)

                if not st.session_state.additional_content:
                    #display ground truth only if not getting additional content
                    st.markdown("### **Ground Truth preconditie en positie in de tekst:**")

                    ground_truth_text = f"""<div class="no-scroll"; style="white-space: normal; word-wrap: break-word;">
                                        <p>Preconditie tekst: {precondition_text}</p>
                                        <p>Preconditie positie: {precondition_position}</p>
                                    </div>"""
                    
                    # replace new line characters with <br> tags for HTML rendering
                    html_ground_truth = ground_truth_text.replace('\n', '<br>')

                    st.markdown(html_ground_truth, unsafe_allow_html=True)
            else:

                st.markdown(f"""
                            ### **Fact:** 
                            {frame}""")
                
                st.markdown("### **Model facts en subfacts:**")

                answer_text = f"""<div class="no-scroll"; style="white-space: normal; word-wrap: break-word;">
                                        <p>{current_response}</p>
                                    </div>"""

                # replace new line characters with <br> tags for HTML rendering
                html_answer = answer_text.replace('\n', '<br>')

                st.markdown(html_answer, unsafe_allow_html=True)
            
                if not st.session_state.additional_content:
                    #display ground truth only if not getting additional content

                    st.markdown("### **Ground Truth subfact en positie in de tekst:**")

                    ground_truth_text = f"""<div class="no-scroll"; style="white-space: normal; word-wrap: break-word;">
                                        <p>Subfact tekst: {precondition_text}</p>
                                        <p>Subfact positie: {precondition_position}</p>
                                    </div>"""
                    
                    # replace new line characters with <br> tags for HTML rendering
                    html_ground_truth = ground_truth_text.replace('\n', '<br>')

                    st.markdown(html_ground_truth, unsafe_allow_html=True)

                
            

            
                    
            #TODO: somehow collect  additional feedback here, but how to do this? How would participants know what part of the answer is additional?
            # maybe ask about how they would rate the quality of the answer overall (how concise it is and how complete???)

            if st.session_state.additional_content:

                # additional content question for contents that are beyond the precondition that the model was supposed to find 
                # not included in the study in the end

                st.markdown(
                    """
                    
                    #### In welke mate was de extra informatie naast de juiste precondities/subfacten met betrekking tot het vinden van de randvoorwaarden en het nadenken over welke delen van de tekst goede randvoorwaarden zijn nuttig of verwarrend?
                     
                    """
                )

                feedback_additional_content = st.radio(
                    "",
                    ("Zeer nuttig", "een beetje nuttig", "een beetje verwarrend", "Zeer verwarrend")
                )

                # Save the feedback and move to the next question
                if st.button("Submit Feedback"):
                    #TODO: in submit feedback function, should only change current data index if all preconditions have been looked at AND all responses have been looked at, 
                    # maybe get precondition keys ads list and keep index within that list...
                    # also change response index 
                    submit_feedback(None, None, data, unique_id, precond_ids, prompt_configs, feedback_additional_content=feedback_additional_content)
                    print(" extra feedback given")
                    st.rerun()  # Trigger a refresh
            
            else:

                # Create Likert scales for feedback
                st.markdown(
                    "#### In welke mate is de preconditie door het model geëxtraheerd?"
                )

                feedback_1 = st.radio(
                    "",
                    ("Volledig fout", "Deels fout", "Grotendeels correct", "Volledig correct")
                )

                st.markdown(
                    "#### Hoe duidelijk is de positie van de preconditie in de tekst van wat het model heeft weergegeven?"
                )

                feedback_2 = st.radio(
                    "",
                    ("Onbestemde positie in ground truth","Helemaal niet duidelijk", "Niet duidelijk", "Duidelijk", "Zeer duidelijk")
                )

                # Save the feedback and move to the next question
                if st.button("Submit Feedback"):
                    #TODO: in submit feedback function, should only change current data index if all preconditions have been looked at AND all responses have been looked at, 
                    # maybe get precondition keys ads list and keep index within that list...
                    # also change response index 
                    submit_feedback(feedback_1, feedback_2, data, unique_id, precond_ids, prompt_configs)
                    print("feedback given")
                    st.rerun()  # Trigger a refresh
        
        elif st.session_state.page_2 == 1:
            print("We are on page 8")
            # Button to navigate back to the study information page
            if st.button("Feedback geven ➡️"):
                next_page_2()
                st.rerun()

            #Display the definition of and action and a precondition
            st.markdown(definitions, unsafe_allow_html=True)

        
    else:
        st.write("Je hebt alle vragen geëvalueerd. Hartelijk dank voor je feedback!")