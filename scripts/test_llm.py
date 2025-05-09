import os
import sys
# from embeddings.data_extraction import extract_text
# from embeddings.vector_store import store_embeddings

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from chains.simple_chain import get_rag_response
from embeddings.data_extraction import extract_text
from embeddings.vector_store import store_embeddings
from models.huggingface.generator import Generator
from models.huggingface.embedding_model import EmbeddingModel

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# load the relevant devices available on the server
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("AVAILABLE_DEVICES")

# Enable expandable CUDA segments
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# knowledge_base_file = '/home/jacques.furst/development/RAG/flintfiller-precondition-rl/data/text/participatiewet_txt_converted-15-17.pdf'

prompt_conditions_1 = {'include_examples': True, 'include_chain_of_thought': True}

# load LLM and tokenizer

generator = Generator(os.getenv("GENERATION_MODEL_NAME"))
llm, tokenizer = generator.load_llm_and_tokenizer()

# load embedding model

# embed_func = EmbeddingModel(os.getenv("EMBEDDING_MODEL_NAME"))  # Load the embedding model name from environment variables
embed_func = None

# docs = extract_text(knowledge_base_file)

response_1 = get_rag_response('De begrotingsstaat behorende bij de begroting van een niet-departementale begroting eventueel inclusief de agentschappen wordt volgens model 1.21 of model 1.22 opgesteld.', llm, tokenizer, embed_func, act=False, number_preconditions=2, prompt_conditions=prompt_conditions_1)
response_2 = get_rag_response('De begrotingsstaat behorende bij de begroting van een niet-departementale begroting eventueel inclusief de agentschappen wordt volgens model 1.21 of model 1.22 opgesteld.', llm, tokenizer, embed_func, act=False, number_preconditions=2, prompt_conditions=prompt_conditions_1)


print("Answer 1: ", response_1)
print("Answer 2: ", response_2)