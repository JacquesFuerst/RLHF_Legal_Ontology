from transformers import AutoModelForCausalLM, AutoTokenizer

def load_llm_and_tokenizer():
    """
    Load the LLM model for the RAG.
    
    Returns:
    ChatOllama: The LLM model
    """

    # device_map=auto allows the model to be loaded on multiple GPUs if available 
    model_name = "Qwen/Qwen2.5-7B-Instruct-1M"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def return_model_name_generator():
    """
    Returns:
    str: The name of the model used in the RAG
    """
    return "Qwen/Qwen2.5-7B-Instruct-1M"