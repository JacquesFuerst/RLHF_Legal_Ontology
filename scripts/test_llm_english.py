from chains.simple_chain import get_rag_response
from embeddings.data_extraction import extract_text
from embeddings.vector_store import store_embeddings
from models.huggingface.generator import Generator
from models.huggingface.embedding_model import EmbeddingModel
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# knowledge_base_file = '/home/jacques.furst/development/RAG/flintfiller-precondition-rl/data/text/participatiewet_txt_converted-15-17.pdf'

prompt_conditions_1 = {'include_examples': True, 'include_chain_of_thought': False}

# load LLM and tokenizer

# generator = Generator(os.getenv("GENERATION_MODEL_NAME"))
# llm, tokenizer = generator.load_llm_and_tokenizer()

# # load embedding model

# embed_func = EmbeddingModel(os.getenv("EMBEDDING_MODEL_NAME"))  # Load the embedding model name from environment variables

# docs = extract_text(knowledge_base_file)

# response_1 = get_rag_response(' to avert disaster on Sunday .', llm, tokenizer, embed_func, prompt_conditions_1)

# print(response_1[0])



model_name = 'Qwen/Qwen2.5-7B-Instruct-1M'
# device = torch.device("cpu")


llm = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

# llm.to(device)
if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

test_prompt = "What is the capital of France?"
inputs = tokenizer(test_prompt, return_tensors="pt").to(llm.device)
generated_ids = llm.generate(**inputs, max_new_tokens=50, temperature=0.8)
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(f"Test response: {response}")