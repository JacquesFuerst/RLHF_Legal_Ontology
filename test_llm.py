from chains.simple_chain import get_rag_response
from embeddings.data_extraction import extract_text
from embeddings.vector_store import store_embeddings
from models.huggingface.generator import Generator
from models.huggingface.embedding_model import EmbeddingModel
import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

knowledge_base_file = '/home/jacques.furst/development/RAG/flintfiller-precondition-rl/data/text/participatiewet_txt_converted-15-17.pdf'

prompt_conditions_1 = {'include_examples': True, 'include_chain_of_thought': False}

# load LLM and tokenizer

generator = Generator(os.getenv("GENERATION_MODEL_NAME"))
llm, tokenizer = generator.load_llm_and_tokenizer()

# load embedding model

embed_func = EmbeddingModel(os.getenv("EMBEDDING_MODEL_NAME"))  # Load the embedding model name from environment variables

# docs = extract_text(knowledge_base_file)

response_1 = get_rag_response('toekennen recht op algemene bijstand college van de gemeente waar de belanghebbende woonplaats heeft', llm, tokenizer, embed_func, prompt_conditions_1)

print(response_1)