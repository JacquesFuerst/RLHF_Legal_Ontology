from ollama_model.llama32 import load_llm
from app.retriever import retrieve_chunks


def generate_prompt(query, context):
        """
        Generate the prompt for the RAG based on the query. For now, this is a format for the toy data.
        """

        # in the sentence: \n {sentence} 

        prompt = f"""

                Use the following context to answer: \n {context} \n \n 
                
                Query: Find the precondition (a single verb in the sentence) to the following event {query} 
                \n \n

                Provide the answer in the follwoing format: \n
                According to the document on page <page number>, line <line number>, the precondition is: \n
                Precondition: <precondition> \n \n
                """
        
        # prompt = f"""

        #         Use the follwoing context to answer: \n {context} \n \n 
                
        #         Query: Find the precondition to the following event {event} 
        #         in the sentence: \n {sentence}. \n
        #         Provide the page and line number of the document you retrieved the answer from.\n \n

        #         Provide the answer in the follwoing format: \n
        #         According to the document on page <page number>, line <line number>, the precondition is: \n
        #         Precondition: <precondition> \n \n
        #         """
        return prompt



#TODO: fix llm.run, does not work with llama3.2 
# --> either use different model or somehow make this work with llama3.2 by using correct function call


def get_rag_response(query): #TODO: add prompt tas argument, pass before
        """
        Get the response from the RAG based on the query. 
        Retrieve the chunks, load the LLM, and se the context.
        """
        llm = load_llm()
        retrieved_chunks = retrieve_chunks(query)
        
        # extract chunk_page content attribute from each chunk
        context = "\n".join([f"Page {chunk.metadata['page_number']}: {chunk.page_content}" for chunk in retrieved_chunks])

        #TODO: use chain.invoke here later

        prompt = generate_prompt(query, context)
        response = llm.invoke(prompt)
        return response, retrieved_chunks