from ollama_model.llama32 import load_llm
from chains.retriever import retrieve_chunks


def generate_prompt(query, context):
        """
        Generate the prompt for the RAG based on the query. For now, this is a format for the toy data.

        Parameters:
        query (str): The query for the RAG
        context (str): The context in which the query is asked

        Returns:
        str: The generated prompt
        """

        # in the sentence: \n {sentence} 

        # Documents with a lower similarity score are a better match. \n \n

        # Return the sentence in the original document in which the precondition is mentioned. \n \n

        # Sentence: <sentence> \n \n

        # A precondition is a single verb that precedes and causes the event in time. 

        prompt = f"""


                --- Definition ---

                Precondition: A precondition is a single verb that precedes and causes the event in time. It is always part of the same sentence as the event, but not of the event itself.
                Event: An event is a single verb that describes an action or occurrence.
                --- Examples ---

                Example 1: 

                Sentence: Because there was no proof that he knew about the World Trade Center and the Pentagon , he was charged in his second trial with accessory to murder in the deaths of those on the planes , rather than in the deaths of everyone killed .
                Event: , he was charged in his second trial with accessory to murder in the deaths of those on the planes ,
                Precondition: killed

                Example 2:

                Sentence: The documents that led to his resignation had been located by Dec. 20 , when the Polish news media began reporting their contents .
                Event: The documents that led to his resignation 
                Precondition: located


                --- Chain of thought ---

                1. Based on the event, find the whole sentence it is part of in the retrieved documents.
                2. In the sentence, locate the precondition.
                3. To determine the line number, count the number of newline characters (`\n`) from the beginning of the document until you reach the target text.
                4. Return the precondition and its proper page and line in the document. 
                5. Provide a reason for why this page and line number were chosen for the precondition.

                
                --- Task ---

                Use the following context to answer: \n {context} \n \n 
                
                Query: Find the precondition for the following event: {query}. Provide its page and line number in the document. \n \n

                Provide the answer in the following format: \n \n
                According to the document on page <page number>, line <line number>, the precondition is: \n 
                Precondition: <precondition> \n \n
                Reason: <reason for the page and line number> \n \n
                """
        return prompt


def get_rag_response(query):
        """
        Get the response from the RAG based on the query. 
        Retrieve the chunks, load the LLM, and se the context.

        Parameters:
        query (str): The query for the RAG
        
        Returns:
        str: The response from the RAG
        str: The context in which the response was generated
        """
        llm = load_llm()
        retrieved_chunks = retrieve_chunks(query)

        print(f"Retrieved chunks: {retrieved_chunks}")
        
        # extract chunk_page content attribute from each chunk
        
        #, similarity score {chunk[1]}
        context = "\n".join([f"Page {chunk[0].metadata['page_number']}: {chunk[0].page_content}" for chunk in retrieved_chunks])

        #TODO: use chain.invoke here later?



        prompt = generate_prompt(query, context)
        response = llm.invoke(prompt)

        print("response: ", response)
        return response, retrieved_chunks