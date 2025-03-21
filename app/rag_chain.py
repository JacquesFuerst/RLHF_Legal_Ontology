from ollama_model.llama32 import load_llm
from app.retriever import retrieve_chunks

from synthetic_feedback import feedback_chain, store_feedback

from langchain.chains.sequential import SequentialChain


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

        prompt = f"""

                Use the following context to answer: \n {context} \n \n 
                
                Query: Find the precondition (a single verb in the sentence) to the following event {query} 
                \n \n

                Provide the answer in the follwoing format: \n
                According to the document on page <page number>, line <line number>, the precondition is: \n \n
                Precondition: <precondition> \n \n
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
        
        # extract chunk_page content attribute from each chunk
        context = "\n".join([f"Page {chunk.metadata['page_number']}: {chunk.page_content}" for chunk in retrieved_chunks])

        #TODO: use chain.invoke here later?

        prompt = generate_prompt(query, context)
        response = llm.invoke(prompt)
        return response, context










############# Future implementation together with feedback chain ###################

def response_chain(query):
        """"
        Get the response from the response model based on the query.
        """
        llm = load_llm()
        retrieved_chunks = retrieve_chunks(query)
        
        # extract chunk_page content attribute from each chunk
        context = "\n".join([f"Page {chunk.metadata['page_number']}: {chunk.page_content}" for chunk in retrieved_chunks])

        #TODO: use chain.invoke here later?

        prompt = generate_prompt(query, context)
        response = llm.invoke(prompt)
        return response, context


def get_rag_response_and_collect_feedback(query):
        """
        
        """
        # Create the sequential chain
        chain = SequentialChain(chains=[response_chain, feedback_chain])

        # Invoke the chain
        response, context, feedback = chain.invoke(query)

        #store the feedback data
        store_feedback(query, context, response, feedback)
        
        return response, context

#####################################################################################################################################