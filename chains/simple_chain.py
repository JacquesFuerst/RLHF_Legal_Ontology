
from chains.retriever import retrieve_chunks, get_whole_doc


from transformers import AutoModelForCausalLM, AutoTokenizer

import os
import gc
import torch
from dotenv import load_dotenv

import time

# Load environment variables from .env file
load_dotenv()

# load the relevant devices available on the server
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("AVAILABLE_DEVICES")

torch.manual_seed(42)

def generate_prompt(query, context, prompt_conditions, number_preconditions=0):
        """
        Generate the prompt for the RAG based on the query. For now, this is a format for the toy data.

        Parameters:
        query (str): The query for the RAG
        context (str): The context in which the query is asked
        prompt_conditions (dict): The conditions for the prompt, including whether to include examples and chain of thought
        number_preconditions (int): The number of preconditions to find, default is 0 to prevent always including this

        Returns:
        str: The generated prompt
        """
        # set the boolean variables for what to include in the prompt

        include_examples = prompt_conditions['include_examples']
        include_chain_of_thought = prompt_conditions['include_chain_of_thought']

        

        chain_of_thought_string = """

                                --- Gedachteketen ---

                                1. Zoek alle vermeldingen van de act in de tekst.
                                2. Zoek in de artikelen waarin de act wordt genoemd naar specifieke precondities voor de act. 
                                3. Zoek ook naar specifieke verwijzingen naar andere artikelen waarin mogelijk andere predcondities voor de act worden genoemd.
                                4. Extraheer de precondities en hun positie in de tekst.

                                """
        

        examples_string = """

                                --- Voorbeelden ---

                                Voorbeeld 1: 

                                Act: Persoon A leent een boek.
                                Precondities: 
                                        a) Een lid van de bibliotheek zijn
                                        b) NOT openstaande boetes hebben


                                Voorbeeld 2:

                                Act: Een Tictactoe speel starten.
                                Precondities:
                                        a) Er bestaat een 3x3 rooster.
                                        b) Alle vierkanten van het rooster zijn leeg.
                        """
        

        number_of_preconditions_string = f"""

                                --- Aantal precondities ---
                                
                                Het aantal precondities dat je moet vinden is: {number_preconditions}.
                                """

        prompt = f"""


                --- Definitie ---

                Preconditie: Een preconditie beschrijft de omstandigheden waaronder de handeling wettelijk kan worden uitgevoerd.
                Act: Een act kan worden uitgevoerd door een agent binnen het normatieve systeem dat wordt gedefinieerd door het juridische document.

                {examples_string if include_examples else ""}

                {chain_of_thought_string if include_chain_of_thought else ""}

                {number_of_preconditions_string if number_preconditions > 0 else ""}
                
                --- Opdracht ---

                Gebruik de volgende context: \n {context} \n \n 
                
                Query: Vind alle precondities voor de volgende act in de tekst: {query}. Toon voor elke preconditie de inhoud en de positie ervan in de wetstekst \n \n

                --- Antwoord ---

                Geef het antwoord in het volgende formaat, voor iedere preconditie die je kan vinden: \n \n
                Preconditie: <preconditie> \n \n
                Positie: Artikel <artikelnummer>, Sectie <sectienummer> \n \n
                """
        

        # chain_of_thought_string = """

        #                         --- Chain of thought ---

        #                         1. Based on the event, find the whole sentence it is part of in the retrieved documents.
        #                         2. In the sentence, locate the precondition.
        #                         3. To determine the line number, count the number of newline characters (`\n`) from the beginning of the document until you reach the target text.
        #                         4. Return the precondition and its proper page and line in the document. 
        #                         5. Provide a reason for why this page and line number were chosen for the precondition.

        #                         """
        

        # examples_string = """

        #                         --- Examples ---

        #                         Example 1: 

        #                         Sentence: Because there was no proof that he knew about the World Trade Center and the Pentagon , he was charged in his second trial with accessory to murder in the deaths of those on the planes , rather than in the deaths of everyone killed .
        #                         Event: , he was charged in his second trial with accessory to murder in the deaths of those on the planes ,
        #                         Precondition: killed

        #                         Example 2:

        #                         Sentence: The documents that led to his resignation had been located by Dec. 20 , when the Polish news media began reporting their contents .
        #                         Event: The documents that led to his resignation 
        #                         Precondition: located
        #                 """

        # prompt = f"""


        #         --- Definition ---

        #         Precondition: A precondition is a single verb that precedes and causes the event in time. It is always part of the same sentence as the event, but not of the event itself.
        #         Event: An event is a single verb that describes an action or occurrence.

        #         {examples_string if include_examples else ""}

        #         {chain_of_thought_string if include_chain_of_thought else ""}
                
        #         --- Task ---

        #         Use the following context to answer: \n {context} \n \n 
                
        #         Query: Find the precondition for the following event: {query}. Provide its page and line number in the document. \n \n

        #         --- Answer ---

        #         Provide the answer in the following format: \n \n
        #         According to the document on page <page number>, line <line number>, the precondition is: \n 
        #         Precondition: <precondition> \n \n
        #         Reason: <reason for the page and line number> \n \n

        #         Do not return the prompt, only the answer! \n \n
        #          """
        return prompt


def get_rag_response(query, llm, tokenizer, embed_func, number_preconditions=0, prompt_conditions=None):
        """
        Get the response from the RAG based on the query. 
        Retrieve the chunks, load the LLM, and se the context.

        Parameters:
        query (str): The query for the RAG
        
        Returns:
        str: The response from the RAG
        str: The context in which the response was generated
        """
        
        # retrieved_chunks = retrieve_chunks(query, embed_func)

        # print(f"Retrieved chunks: {retrieved_chunks}")
        
        # extract chunk_page content attribute from each chunk
        
        #, similarity score {chunk[1]}
        # context = "\n".join([f"Pagina {chunk[0].metadata['paginanummer']}: {chunk[0].page_content}" for chunk in retrieved_chunks])

        # get the whole document to reduce memory usage 
        context = get_whole_doc()

        # Generate the prompt using the query and context
        prompt = generate_prompt(query, context, prompt_conditions, number_preconditions=number_preconditions)

        print("Prompt: ", prompt)



        # print(f"Prompt: {prompt}")
        # Ensure the tokenizer has a padding token
        if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token


        # prompt = "What is the captial of France?" 

        # print(llm.device)

        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(llm.device)

        # Generate the response using the LLM --> do sample leads to more creative outputs since we are sampling from prob dist next token
        start = time.time()

        # print("Everything up until here is done")

        #limit number of new tokens to number of preconditions * 150 to prevent hallucinations
        max_new_tokens = number_preconditions * 150
        
        with torch.no_grad():
                generated_ids = llm.generate(**inputs, do_sample=True, temperature=0.7, top_p=0.9, max_new_tokens=max_new_tokens)
        print("⏱ Time taken:", time.time() - start)

        # print("Generated IDs:", generated_ids)


        # Exclude the prompt from the output
        # prompt_length = inputs['input_ids'].shape[1]
        answer = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        # response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # ollama version
        # response = llm.invoke(prompt)

        # print("response: ", answer)

        return answer #retrieved_chunks











################### English version prompt ###########################


# chain_of_thought_string = """

#                                 --- Chain of thought ---

#                                 1. Based on the event, find the whole sentence it is part of in the retrieved documents.
#                                 2. In the sentence, locate the precondition.
#                                 3. To determine the line number, count the number of newline characters (`\n`) from the beginning of the document until you reach the target text.
#                                 4. Return the precondition and its proper page and line in the document. 
#                                 5. Provide a reason for why this page and line number were chosen for the precondition.

#                                 """
        

# examples_string = """

#                         --- Examples ---

#                         Example 1: 

#                         Sentence: Because there was no proof that he knew about the World Trade Center and the Pentagon , he was charged in his second trial with accessory to murder in the deaths of those on the planes , rather than in the deaths of everyone killed .
#                         Event: , he was charged in his second trial with accessory to murder in the deaths of those on the planes ,
#                         Precondition: killed

#                         Example 2:

#                         Sentence: The documents that led to his resignation had been located by Dec. 20 , when the Polish news media began reporting their contents .
#                         Event: The documents that led to his resignation 
#                         Precondition: located
#                 """

# prompt = f"""


#         --- Definition ---

#         Precondition: A precondition is a single verb that precedes and causes the event in time. It is always part of the same sentence as the event, but not of the event itself.
#         Event: An event is a single verb that describes an action or occurrence.

#         {examples_string if include_examples else ""}

#         {chain_of_thought_string if include_chain_of_thought else ""}
        
#         --- Task ---

#         Use the following context to answer: \n {context} \n \n 
        
#         Query: Find the precondition for the following event: {query}. Provide its page and line number in the document. \n \n

#         --- Answer ---

#         Provide the answer in the following format: \n \n
#         According to the document on page <page number>, line <line number>, the precondition is: \n 
#         Precondition: <precondition> \n \n
#         Reason: <reason for the page and line number> \n \n

#         Do not return the prompt, only the answer! \n \n
#         """

##############################################################################################