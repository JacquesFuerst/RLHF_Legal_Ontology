import torch
import warnings
warnings.filterwarnings('ignore')
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
# from trl import PPOConfig, create_reference_model, AutoModelForCausalLMWithValueHead
from trl import GRPOTrainer, GRPOConfig
# from datasets import DatasetDict

from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training

import os
from dotenv import load_dotenv
from utils import CustomRewardFunction
# from ppo_trainer_custom import CustomPPOTrainer
import pandas as pd
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin
import json

# from types import MethodType
# import json
# import sys
# import wandb

# # Add the parent directory to the Python path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables from .env file
load_dotenv()

# load the relevant devices available on the server
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("AVAILABLE_DEVICES")

# Enable expandable CUDA segments
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

print(os.environ["CUDA_VISIBLE_DEVICES"])

# import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())
print(torch.cuda.is_available())

# load cuda
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print("CUDA is available. Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")



# Global variables
MODEL = os.getenv("GENERATION_MODEL_NAME")
ALGORITHM = os.getenv("RL_ALGORITHM")
REWARD_MODEL = os.getenv("REWARD_MODEL_NAME")
REWARD_MODEL_EXTRACTION_LORA = os.getenv("REWARD_MODEL_EXTRACTION_LORA")
REWARD_MODEL_DETECTION_LORA = os.getenv("REWARD_MODEL_DETECTION_LORA")
RL_TOKENIZATION = os.getenv("RL_TOKENIZATION")
MAX_LENGTH = int(os.getenv("RL_MAX_LENGTH"))
STRIDE = int(os.getenv("RL_STRIDE"))
PROMPT_DATASET_TRAIN = os.getenv("PROMPT_DATASET_TRAIN")
PROMPT_DATASET_EVAL = os.getenv("PROMPT_DATASET_EVAL")
DETECTION_DIFFERENCE = int(os.getenv("DETECTION_DIFFERENCE"))
WEIGHT_EXTRACTION = float(os.getenv("WEIGHT_EXTRACTION"))
WEIGHT_DETECTION = float(os.getenv("WEIGHT_DETECTION"))
WEIGHT_LENGTH_PENALTY = float(os.getenv("WEIGHT_LENGTH_PENALTY"))
RL_TRAINING_FILES = os.getenv("RL_TRAINING_FILES") + "_" + ALGORITHM


# Load DeepSpeed config

with open("/home/jacques.furst/development/RAG/flintfiller-precondition-rl/rl_training_new/deepspeed_config.json", "r") as f:
    ds_config = json.load(f)

ds_plugin = DeepSpeedPlugin(
    hf_ds_config=ds_config
)
accelerator = Accelerator(
    mixed_precision="bf16", 
    deepspeed_plugin=ds_plugin  # Optional if you loaded config from file
)



#### Load prompt train dataset ####

prompt_df_train = pd.read_csv(PROMPT_DATASET_TRAIN, sep=";")
prompt_df_eval = pd.read_csv(PROMPT_DATASET_EVAL, sep=";")
dataset_train = Dataset.from_pandas(prompt_df_train)
dataset_eval = Dataset.from_pandas(prompt_df_eval)



base_model = AutoModelForCausalLM.from_pretrained(MODEL,  
                                            #  device_map="auto",  # For GPU/TPU acceleration
                                            device_map=None,
                                            torch_dtype=torch.bfloat16,
                                            #  load_in_4bit=True,
                                            quantization_config={
                                                "load_in_4bit": True,
                                                "bnb_4bit_compute_dtype": torch.bfloat16,
                                                "bnb_4bit_use_double_quant": True,
                                                "bnb_4bit_quant_type": "nf4"
                                                }
                                            )   # Optimize precision)


tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

qlora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # depends on the model architecture
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

base_model.gradient_checkpointing_enable()

# Prepare for QLoRA fine-tuning
base_model = prepare_model_for_kbit_training(base_model)

# Apply QLoRA
policy_model = get_peft_model(base_model, qlora_config)

policy_model.print_trainable_parameters()




######## Model Lodading ##########

# Load reward model feedback extraction
reward_model_extraction = AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL, num_labels=1)
reward_model_detection = AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL, num_labels=1)
reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL)

extraction_model = PeftModel.from_pretrained(reward_model_extraction, REWARD_MODEL_EXTRACTION_LORA).to(device)
# extraction_model = extraction_model.merge_and_unload()

detection_model = PeftModel.from_pretrained(reward_model_detection, REWARD_MODEL_DETECTION_LORA).to(device)
# detection_model = detection_model.merge_and_unload()

# response_text = """
#                         Inhoud: <inhoud>
#                         <details>
#                         <summary>parsering</summary>
#                         <pre>
#                         <code>
#                 subfact
#                         </code>
#                         </pre>
#                         </details>
        
#                         Resultaat:


#                         Subfact: vreemdeling dfsf
        
#                         Positie: Artikel 8 IN Verordening vreemdelingenattributen
        
#                         Inhoud: de vreemdeling heeft in Nederland uitsluitend rechtmatig verblijf:
#                         <details>
#                         <summary>parsering</summary>
#                         <pre>
#                         <code>
#                 de vreemdeling
#                         </code>
#                         </pre>
#                         </details>
        
#                         Subfact: vreemdeling 
        
#                         Positie: Artikel 8 IN Verordening vreemdelingenattributen
        
#                         Inhoud: het verblijf van een vreemdeling in Nederland op grond van deze wet anders dan op de 
#                         gronden bedoeld in de artikelen 29 en 34
#                         <details>
#                         <summary>parsering</summary>
#                         <pre>
#                         <code>
#                 het verblijf van een vreemdeling
#                         </code>
#                         </pre>
#                         </details>
        
#                         Subfact: vreemdeling 
        
#                         Positie: Artikel 8, onder a IN Verordening vreemdelingenattributen
        
#                         Inhoud: op grond van een verblijfsvergunning voor bepaalde tijd als bedoeld in artikel 14;
#                         <details>
#                         <summary>parsering</summary>
#                         <pre>
#                         <code>
#                 op grond van een verblijfsvergunning voor bepaalde tijd
#                         </code>
#                         </pre>
#                         </details>
#                         """

# precon_text = "NOT ieder die op grond van een wettelijke bepaling als Nederlander moet worden behandeld"

# with torch.no_grad():
#     inputs = reward_tokenizer(precon_text + " " + response_text, return_tensors='pt', truncation=True, padding="max_length").to(device)
#     outputs = extraction_model(**inputs)
#     prediction = outputs.logits.item()
#     print(f"Sample {1}: Predicted Rating: {prediction}")

#         #################


# Create the custom reward function
reward_function = CustomRewardFunction(extraction_model, 
                                       detection_model, 
                                       reward_tokenizer, 
                                       MAX_LENGTH, STRIDE, 
                                       RL_TOKENIZATION, 
                                       device, 
                                       weight_extraction=WEIGHT_EXTRACTION, 
                                       weight_detection=WEIGHT_DETECTION,
                                       weight_length_penalty=WEIGHT_LENGTH_PENALTY, 
                                       detection_difference=DETECTION_DIFFERENCE)


########## GRPO setup #############

if ALGORITHM == "GRPO":

    training_args = GRPOConfig(
        output_dir=RL_TRAINING_FILES, 
        per_device_train_batch_size=1,
        per_device_eval_batch_size=6,
        logging_steps=1, 
        gradient_checkpointing=True,
        learning_rate=5e-5,
        num_train_epochs=10,
        # weight_decay=0.01,
        # warmup_steps=17, # TODO:check if this makes any sense at all
        logging_dir="logs",
        # save_steps=1,
        save_total_limit=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        # eval_steps=1,
        # batch_size=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        gradient_accumulation_steps=3, #TODO: think about whether this is truly necessary
        report_to="wandb",
        max_completion_length=2048,
        max_prompt_length=3000,
        optim="adamw_8bit",
        bf16=True,
        ddp_find_unused_parameters=False,
        num_generations=9, # Number of generations per prompt
        )

    # Initialize GRPO trainer
    trainer = GRPOTrainer(
        model=policy_model,
        reward_funcs=reward_function,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        args=training_args,
        # peft_config=lora_config
    )


trainer.train()



