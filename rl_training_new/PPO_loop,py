import torch
import warnings
warnings.filterwarnings('ignore')
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from trl import PPOConfig, create_reference_model, AutoModelForCausalLMWithValueHead
from trl import GRPOTrainer, GRPOConfig
from datasets import DatasetDict

from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training

import os
from dotenv import load_dotenv
from utils import CustomRewardFunction, LabelPreservingCollator, CustomRewardFunctionPPOTrainer, tokenize_and_keep_original, create_label
from ppo_trainer_custom import CustomPPOTrainer
import pandas as pd
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin

from types import MethodType
import json
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
RL_TOKENIZATION = "best_window"
MAX_LENGTH = int(os.getenv("RL_MAX_LENGTH"))
STRIDE = int(os.getenv("RL_STRIDE"))
PROMPT_DATASET_TRAIN = os.getenv("PROMPT_DATASET_TRAIN")
PROMPT_DATASET_EVAL = os.getenv("PROMPT_DATASET_EVAL")
DETECTION_DIFFERENCE = int(os.getenv("DETECTION_DIFFERENCE"))
WEIGHT_EXTRACTION = float(os.getenv("WEIGHT_EXTRACTION"))
WEIGHT_DETECTION = float(os.getenv("WEIGHT_DETECTION"))
RL_TRAINING_FILES = os.getenv("RL_TRAINING_FILES") + "_" + ALGORITHM


# Load DeepSpeed config
if ALGORITHM == "PPO":
    ds_plugin = DeepSpeedPlugin(
        hf_ds_config="deepspeed_config.json"
    )
    accelerator = Accelerator(
        mixed_precision="fp16", 
        deepspeed_plugin=ds_plugin  # Optional if you loaded config from file
    )
else: 
    accelerator = Accelerator()


#### Load prompt train dataset ####

prompt_df_train = pd.read_csv(PROMPT_DATASET_TRAIN, sep=";")
prompt_df_eval = pd.read_csv(PROMPT_DATASET_EVAL, sep=";")
dataset_train = Dataset.from_pandas(prompt_df_train)
dataset_eval = Dataset.from_pandas(prompt_df_eval)



base_model = AutoModelForCausalLM.from_pretrained(MODEL,  
                                             device_map={"": accelerator.process_index},  # For GPU/TPU acceleration
                                             torch_dtype=torch.bfloat16,
                                            #  load_in_4bit=True,
                                             quantization_config={
                                                "load_in_4bit": True,
                                                "bnb_4bit_compute_dtype": torch.bfloat16,
                                                "bnb_4bit_use_double_quant": True,
                                                "bnb_4bit_quant_type": "nf4"
                                                }
                                            )   # Optimize precision)


tokenizer = AutoTokenizer.from_pretrained(MODEL, truncation=False, padding=False)
tokenizer.pad_token = tokenizer.eos_token

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



######## Model Lodading ##########

# Load reward model feedback extraction
reward_model = AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL, num_labels=1)
reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL)

extraction_model = PeftModel.from_pretrained(reward_model, REWARD_MODEL_EXTRACTION_LORA).to(device)
# extraction_model = extraction_model.merge_and_unload()

detection_model = PeftModel.from_pretrained(reward_model, REWARD_MODEL_DETECTION_LORA).to(device)
# detection_model = detection_model.merge_and_unload()


# Create the custom reward function
reward_function = CustomRewardFunction(extraction_model, 
                                       detection_model, 
                                       reward_tokenizer, 
                                       MAX_LENGTH, STRIDE, 
                                       RL_TOKENIZATION, 
                                       device, 
                                       weight_extraction=WEIGHT_EXTRACTION, 
                                       weight_detection=WEIGHT_DETECTION, 
                                       detection_difference=DETECTION_DIFFERENCE)


###### LoRA setup #####

#TODO: is this truly needed --> already QLora on policy model --> too much overhead 

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type='CAUSAL_LM',  
)








######### PPO training ##############



#TODO: Prepare dataset here
# Need to tokenize to use for PPO



# Apply the function to the dataset
tokenized_dataset_train = dataset_train.map(tokenize_and_keep_original, 
                                            fn_kwargs={"tokenizer": tokenizer},
                                            batched=True)
tokenized_dataset_eval = dataset_train.map(tokenize_and_keep_original, 
                                           fn_kwargs={"tokenizer": tokenizer},
                                           batched=True)



tokenized_dataset_train = tokenized_dataset_train.map(create_label)
tokenized_dataset_train = tokenized_dataset_train.remove_columns(["prompt", "precondition_texts", "precondition_positions"])

tokenized_dataset_eval = tokenized_dataset_eval.map(create_label)
tokenized_dataset_eval = tokenized_dataset_eval.remove_columns(["prompt", "precondition_texts", "precondition_positions"])

print(tokenized_dataset_train[0]['additional_entries'][2])

# use own data collator that does not pad label column
data_collator = LabelPreservingCollator(tokenizer)


if ALGORITHM == "PPO":
    #TODO: use create reference model function here instead...

    ref_model = create_reference_model(policy_model)
    # ref_model.to(model.device)
    # load the value model with same peft setup as the policy model
    
    # can add value head to policy model here
    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(policy_model, 
                                                                    peft_config=qlora_config, 
                                                                    device_map={"": accelerator.process_index},  # For GPU/TPU acceleration
                                                                    )
    policy_model.base_model_prefix = "pretrained_model"

    def score(self, hidden_states):
        return self.v_head(hidden_states).squeeze(-1)

    policy_model.score = MethodType(score, policy_model)

    #TODO: use accelerator.process_index here maybe

    reward_function_PPO = CustomRewardFunctionPPOTrainer(extraction_model, 
                                                         detection_model, 
                                                         reward_tokenizer, 
                                                         MAX_LENGTH, 
                                                         STRIDE, 
                                                         RL_TOKENIZATION, 
                                                         device, 
                                                         weight_extraction=WEIGHT_EXTRACTION, 
                                                         weight_detection=WEIGHT_DETECTION, 
                                                         detection_difference=DETECTION_DIFFERENCE)
    

if ALGORITHM == "PPO":

    

    training_args_PPO = PPOConfig(
        output_dir=RL_TRAINING_FILES, 
        logging_steps=10, 
        gradient_checkpointing=True,
        learning_rate=1e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        warmup_steps=100,
        logging_dir="logs",
        save_steps=1000,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="wandb",
        response_length=512, # max length of the model responses generated --> get 512
        local_rollout_forward_batch_size=4, # generate two responses per query --> get 2
        per_device_train_batch_size=4, # get two queries --> get 2 --> maybe change later
        per_device_eval_batch_size=4,
        )

    # Initialize GRPO trainer
    trainer = CustomPPOTrainer(
        model=policy_model,
        reward_func=reward_function_PPO,
        # collator_max_length=2000,
        train_dataset=tokenized_dataset_train,
        eval_dataset=tokenized_dataset_eval,
        args=training_args_PPO,
        ref_model=ref_model,
        value_model=policy_model,
        # **grpo_config
        peft_config=lora_config,
        processing_class=tokenizer,
        data_collator=data_collator
    )


# Train
policy_model = accelerator.prepare(policy_model)

trainer.train()


