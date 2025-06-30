import torch
import os
import sys
from dotenv import load_dotenv
import ast
import evaluate

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

from distutils.util import strtobool
import copy

# from deepeval.models.azure_openai import AzureOpenAIModel

from peft import PeftModel

import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM

from deepeval.models import AzureOpenAIModel
# from deepeval.metrics import AnswerRelevancyMetric

# Load environment variables from .env file
load_dotenv()

# Add the parent directory to the Python path
# __file__


# Adjust this path to point to the directory containing rl_training_new
module_path = os.path.abspath(os.path.join('..')) # or another relative path
if module_path not in sys.path:
    sys.path.append(module_path)

# from rl_training_new.utils import find_best_window


from bert_score import score

# load the relevant devices available on the server
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("AVAILABLE_DEVICES")

# Enable expandable CUDA segments
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# load cuda
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print("CUDA is available. Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")



#### Global variables ####

MODEL = os.getenv("GENERATION_MODEL_NAME")
ALGORITHM = os.getenv("EVAL_MODEL_ALGORITHM")
# RL_TRAINED_ADAPTERS = os.getenv("EVAL_MODEL_FOLDER")
EVAL_ANSWERS_CSV_MODEL_COMP = os.getenv("EVAL_ANSWERS_CSV_MODEL_COMP")
# GENERATE_RESPONSES = str_to_bool(os.getenv("GENERATE_RESPONSES"))
OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
# RL_DATA_PATH = os.getenv("RL_DATA_PATH")
EVAL_FILE = os.getenv("EVAL_FILE")
NUM_RESPONSES_EVAL = int(os.getenv("NUM_RESPONSES_EVAL"))  # Number of responses per model

#TODO: add links for all models to compare in dedicated folder

# Model names and adapter files

MODEL_NAME_1 = os.getenv("EVAL_MODEL_NAME_1")
MODEL_NAME_2 = os.getenv("EVAL_MODEL_NAME_2")
MODEL_NAME_3 = os.getenv("EVAL_MODEL_NAME_3")

RL_TRAINED_ADAPTERS_1 = os.getenv("RL_TRAINED_ADAPTERS_1")
RL_TRAINED_ADAPTERS_2 = os.getenv("RL_TRAINED_ADAPTERS_2")
RL_TRAINED_ADAPTERS_3 = os.getenv("RL_TRAINED_ADAPTERS_3")




# Load the models to compare

#TODO: do copy.deepcopy instead of loading 4 models

# base_model = AutoModelForCausalLM.from_pretrained(MODEL)
base_model = AutoModelForCausalLM.from_pretrained(MODEL,  
                                            #  device_map="auto",  # For GPU/TPU acceleration
                                            device_map={"": "cuda:0"},
                                            torch_dtype=torch.bfloat16,
                                            #  load_in_4bit=True,
                                            quantization_config={
                                                "load_in_4bit": True,
                                                "bnb_4bit_compute_dtype": torch.bfloat16,
                                                "bnb_4bit_use_double_quant": True,
                                                "bnb_4bit_quant_type": "nf4"
                                                }
                                            )   # Optimize precision)

base_model.to("cuda:0")

# new_model_2.to("cuda:2")
# new_model_3.to("cuda:3")

base_model_new = copy.deepcopy(base_model)
base_model_new.to("cuda:1")
base_model_new_2 = copy.deepcopy(base_model)
base_model_new_2.to("cuda:2")
base_model_new_3 = copy.deepcopy(base_model)
base_model_new_3.to("cuda:3")
# base_model_new = AutoModelForCausalLM.from_pretrained(MODEL)
# base_model_new = AutoModelForCausalLM.from_pretrained(MODEL,  
#                                             #  device_map="auto",  # For GPU/TPU acceleration
#                                             device_map={"": "cuda:1"},
#                                             torch_dtype=torch.bfloat16,
#                                             #  load_in_4bit=True,
#                                             quantization_config={
#                                                 "load_in_4bit": True,
#                                                 "bnb_4bit_compute_dtype": torch.bfloat16,
#                                                 "bnb_4bit_use_double_quant": True,
#                                                 "bnb_4bit_quant_type": "nf4"
#                                                 }
#                                             )   # Optimize precision)

# base_model_new_2 = AutoModelForCausalLM.from_pretrained(MODEL,  
#                                             #  device_map="auto",  # For GPU/TPU acceleration
#                                             device_map={"": "cuda:2"},
#                                             torch_dtype=torch.bfloat16,
#                                             #  load_in_4bit=True,
#                                             quantization_config={
#                                                 "load_in_4bit": True,
#                                                 "bnb_4bit_compute_dtype": torch.bfloat16,
#                                                 "bnb_4bit_use_double_quant": True,
#                                                 "bnb_4bit_quant_type": "nf4"
#                                                 }
#                                             )   # Optimize precision)

# base_model_new_3 = AutoModelForCausalLM.from_pretrained(MODEL,  
#                                             #  device_map="auto",  # For GPU/TPU acceleration
#                                             device_map={"": "cuda:3"},
#                                             torch_dtype=torch.bfloat16,
#                                             #  load_in_4bit=True,
#                                             quantization_config={
#                                                 "load_in_4bit": True,
#                                                 "bnb_4bit_compute_dtype": torch.bfloat16,
#                                                 "bnb_4bit_use_double_quant": True,
#                                                 "bnb_4bit_quant_type": "nf4"
#                                                 }
#                                             )   # Optimize precision)


new_model_1 = PeftModel.from_pretrained(base_model_new, RL_TRAINED_ADAPTERS_1)
new_model_2 = PeftModel.from_pretrained(base_model_new_2, RL_TRAINED_ADAPTERS_2)
new_model_3 = PeftModel.from_pretrained(base_model_new_3, RL_TRAINED_ADAPTERS_3)

tokenizer = AutoTokenizer.from_pretrained(MODEL)

base_model.eval()
new_model_1.eval()
new_model_2.eval()
new_model_3.eval()







# get evaluation dataset

df = pd.read_csv(EVAL_FILE, sep=';')


# Function to generate response
def generate_response(prompt, tokenizer, model, max_length=1024):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = inputs['input_ids'].shape[1]
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=input_length + max_length, do_sample=True, top_k=50)
        generated_ids = outputs[0][input_length:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)




# generate the responses

#TODO: change if statements to check for proper file
if not os.path.exists(EVAL_ANSWERS_CSV_MODEL_COMP):
    # Generate multiple responses for each prompt
    for i in range(NUM_RESPONSES_EVAL):
        df[f'response_base_model_{i+1}'] = df['prompt'].apply(lambda x: generate_response(x, tokenizer, base_model))
        df[f'response_model_{MODEL_NAME_1}_{i+1}'] = df['prompt'].apply(lambda x: generate_response(x, tokenizer, new_model_1))
        df[f'response_new_model__{MODEL_NAME_2}_{i+1}'] = df['prompt'].apply(lambda x: generate_response(x, tokenizer, new_model_2))
        df[f'response_new_model__{MODEL_NAME_3}_{i+1}'] = df['prompt'].apply(lambda x: generate_response(x, tokenizer, new_model_3))

    # Show result: displaying first response columns for brevity
    response_cols = [f'response_new_model_{i+1}' for i in range(NUM_RESPONSES_EVAL)] + \
                    [f'response_base_model_{i+1}' for i in range(NUM_RESPONSES_EVAL)]
    print(df[['prompt'] + response_cols])

    # Store in CSV
    df.to_csv(EVAL_ANSWERS_CSV_MODEL_COMP, index=False, sep=';')

else:
    df = pd.read_csv(EVAL_ANSWERS_CSV_MODEL_COMP, sep=';')




### Process ground truth for evaluation ####


# Create list of column names
new_cols = [f'response_new_model_{j+1}' for j in range(NUM_RESPONSES_EVAL)]
base_cols = [f'response_base_model_{j+1}' for j in range(NUM_RESPONSES_EVAL)]

# Select columns and convert to list of lists (rows)
candidates_new = df[new_cols].values.tolist()
candidates_base = df[base_cols].values.tolist()

precon_text_list = df['precondition_texts'].to_list()
precon_pos_list = df["precondition_positions"].to_list()

references = []
for dict1, dict2 in zip(precon_text_list, precon_pos_list):
    dict1 = ast.literal_eval(dict1)
    dict2 = ast.literal_eval(dict2)
    combined = []
    for key in dict1.keys():  # or use sorted(dict1.keys()) if key order isn't guaranteed
        combined.append(str(dict1[key]) + '\n')
        combined.append(str(dict2[key]) + '\n\n')
    references.append(''.join(combined))

print(references)


# Flatten the candidate lists
candidates_new_flat = [resp for row in candidates_new for resp in row]
candidates_base_flat = [resp for row in candidates_base for resp in row]

# Repeat each reference NUM_RESPONSES times to match the flattened predictions
references_flat = [ref for ref in references for _ in range(NUM_RESPONSES_EVAL)]




###### ROUGE score ##########

rouge = evaluate.load('rouge')


results_new = rouge.compute(predictions=candidates_new_flat, references=references_flat)
results_base = rouge.compute(predictions=candidates_base_flat, references=references_flat)


print(f"Results ROUGE-L new: {results_new}")
print(f"Results ROUGE-L base: {results_base}")


###### BERTScore ##########

# Maybe add BERTScore --> semantic similarity based on sentencetransformer
P_new, R_new, F1_new = score(
    candidates_new_flat, 
    references_flat, 
    model_type='answerdotai/ModernBERT-base', 
    num_layers=22,
    lang='nl')

P_base, R_base, F1_base = score(
    candidates_base_flat, 
    references_flat, 
    model_type='answerdotai/ModernBERT-base', 
    num_layers=22,
    lang='nl')

print(f"BERT Score metrics new: {P_new, R_new, F1_new}")
print(f"BERT Score metrics base: {P_base, R_base, F1_base}")

print(f"F1 base: {F1_base.mean()}, F1 new: {F1_new.mean()}")