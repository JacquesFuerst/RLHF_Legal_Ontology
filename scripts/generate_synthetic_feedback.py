import os
from openai import AzureOpenAI

from app.utils_json import read_json, write_json
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file


# setup variables for the Azure OpenAI API
endpoint = "https://openai-ds-instance-sweden.openai.azure.com/"
model_name = "gpt-4.1"
deployment = "deze-voor-alles"



subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = "2024-12-01-preview"



client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

# System prompt, similar to instructions human participants received

system_prompt = """

                    U bent een expert in het FLINT-framework, een ontologie voor normatieve systemen (voornamelijk weergegeven in de vorm van juridische en beleidsmatige teksten).

                    ---------------------------------------------------------------------------------------

                    In deze studie evalueert u antwoorden van een taalmodel dat precondities of subfacts probeert te extraheren uit een wettelijke tekst, op basis van een gegeven act of fact. Hieronder vindt u een uitleg van de gebruikte termen:

                    - **Act**: Een handeling die kan worden uitgevoerd door een agent binnen het normatieve systeem dat wordt gedefinieerd door het juridische document.
                    - **Preconditie**: De omstandigheden waaronder de handeling wettelijk kan worden uitgevoerd. Als er 'NOT' voor een preconditie staat, betekent dit dat de ontkende versie van de voorwaarde moet gelden om de actie geldig te laten zijn.
                    - **Fact**: Een feit waarvan de aanwezigheid of waarheidswaarde de toestand van het normatieve systeem kenmerkt.
                    - **Subfact**: Een feit dat een deel van een grotere fact is. Er moeten een of merdere subfacts kloppen om de waarheid van de fact te bepalen.

                    ### Voorbeelden

                    - **Act (bibliotheekcontext)**:  
                        _"Persoon A leent een boek."_
                        
                        **Bijbehorende precondities**:
                        
                        - Een lid van de bibliotheek zijn
                        - NOT openstaande boetes hebben
                        
                    - **Fact (Mens-erger-je-niet context)**:  
                        _"Een pion van een speler staat op het bord."_
                        
                        **Bijbehorende subfacts**:
                        
                        1. NOT de pion is in het startvak
                        2. De speler heeft een zes gegooid

                    ---

                    ## Procedure

                    Als u instemt met deelname, krijgt u rond 250 paren van een act/fact en de bijbehorende subfact/preconditie(s) te zien.

                    Voor elk paar werd de act/fact aan een taalmodel gegeven als onderdeel van een prompt, met de opdracht om alle bijbehorende subfact/preconditie(s) en hun respectieve positie(s) in de tekst terug te geven.

                    Uw taak is om per paar te evalueren hoe goed het model presteerde op twee punten (op een 4-punt Likert-schaal):

                    1. **Het vinden van alle relevante precondities in de tekst**
                    2. **Hoe duidelijk de positie in de tekst is die het model aanduidde**

                    Voor het modelantwoord krijgt u het hele antwoord voor een act/fact (die dus meredere precondities/subfacts kan beïnhouden) en moet u voor elke ground truth preconditie afzonderlijk evalueren of deze 
                        a) aanwezig is in het antwoord en 
                        b) of de positie ervan ook goed is aangegeven in het antwoord. 
                        
                    Soms werd de positie niet gegeven in de ground truth, dit kunt u ook aangeven in de feedback.

                    U wordt gevraagd om feedback te geven voor hetzlefde antwoord voor alle ground truth precondities in isolatie, dus na het geven van feedback voor één preconditie verandert het antwoord mogelijk niet als er meerdere precondities/subfacts aanwezig zijn in de set voor de gegeven act/fact.

                    -----------------------------------------------------

                    Geef uw feedback in het volgende formaat:

                    preconditie extractie: Volledig fout/Deels fout/Grotendeels correct/Volledig correct

                    preconditie detectie: Geen positie in ground truth/ Niet goed/ Goed
                    """




# Generate synthetic feedback

data = read_json(os.getenv('DATA_INTERFACE'))

for datapoint in data:
    answer = datapoint['answer']
    ground_truth = datapoint['ground_truth']
    # Assuming ground_truth is a list of preconditions/subfacts
    ground_truth_str = "\n".join(ground_truth)

    # Construct the content string
    # This is a simplified example, adjust according to your actual data structure



    content_string = f"""

                    Model antwoord: {answer}
                    Ground truth precondities/subfacts: {ground_truth_str}
                    """ # answer + ground truth
    
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": content_string,
            }
        ],
        max_tokens=4096,
        temperature=1.0,
        top_p=1.0,
        model=deployment
    )

    # Synthetic feedback added to the datapoint
    datapoint['synthetic_feedback'] = response.choices[0].message.content

    # Store synthetic feedback in csv format

    # Define the CSV file path
    csv_file_path = os.getenv('SYNTHETIC_FEEDBACK_CSV')

    # Check if the CSV file already exists
    file_exists = os.path.isfile(csv_file_path)

    # Define the field names for the CSV file
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
                'feedback_detection', 
                'additional_feedback'
    ]

    # TODO: figure out how to get all these variables or which are even needed when generating synthetic feedback

    row = {
            'file': datapoint['file'], 
            'frame_ID': datapoint['ID'], 
            'frame_type': datapoint['type'],
            'frame_text': datapoint['text'], 
            'precondition_id': current_precond,
            'precondition_text': datapoint['precondition_texts'][current_precond],
            'precondition_position': datapoint['text_positions'][current_precond],
            'response_text': datapoint['responses'][current_prompt_config][current_response_index],
            'prompt_config_examples': config_examples[1],
            'prompt_config_chain_of_thought': config_chain_of_thought[1],
            'feedback_extraction': feedback_1,
            'feedback_detection':feedback_2,
            'additional_feedback': None
        }



















print(response.choices[0].message.content)