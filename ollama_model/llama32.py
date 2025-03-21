from langchain_ollama import ChatOllama

#TODO: figure out whether Ollama is truly the best way to do this... 

def load_llm():
    model = ChatOllama(
        model="llama3.2",
        language="en")
    return model

def return_model_name():
    return "llama3.2_v1"