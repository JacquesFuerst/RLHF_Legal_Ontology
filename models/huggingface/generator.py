from transformers import AutoModelForCausalLM, AutoTokenizer

import torch


class Generator:
    """
    A class to handle the generation of text using a language model.
    """

    def __init__(self, model_name):
        """
        Initialize the Generator with a language model.
        Parameters:
        model_name (str): The name of the language model to use.
        """
        self.model_name = model_name

    # @st.cache_resource
    def load_llm_and_tokenizer(self):
        """
        Load the language model and tokenizer from Hugging Face.

        Returns:
        tuple: A tuple containing the loaded model and tokenizer.
        """

        # device_map=auto allows the model to be loaded on multiple GPUs if available 

        print(f"model name: {self.model_name}")
        model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, torch_dtype=torch.float16, device_map="auto")

        model.eval()
        return model, tokenizer