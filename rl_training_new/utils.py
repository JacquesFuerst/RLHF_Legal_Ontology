import torch

import os
from dotenv import load_dotenv
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error

from sentence_transformers import SentenceTransformer
from torch.nn import functional as F
from transformers import Trainer, DataCollatorWithPadding, TrainerCallback

import ast
import wandb

# Load environment variables from .env file
load_dotenv()


# if labels are not integers, convert them to integers
def convert_label_to_int(data):
    data["label"] = int(data["label"])
    return data

def parse_ratings(rating):
    rating_dict = {
        "Volledig fout": int(os.getenv("EXTRACTION_FEEDBACK_0")),
        "Deels fout":  int(os.getenv("EXTRACTION_FEEDBACK_1")),
        "Grotendeels correct":  int(os.getenv("EXTRACTION_FEEDBACK_2")),
        "Volledig correct":  int(os.getenv("EXTRACTION_FEEDBACK_3")),
        "Geen positie in ground truth":  int(os.getenv("DETECTION_FEEDBACK_NONEXISTENT")),
        "Niet goed":  int(os.getenv("DETECTION_FEEDBACK_1")),
        "Goed":  int(os.getenv("DETECTION_FEEDBACK_0")),
        "Duidelijk":  int(os.getenv("DETECTION_FEEDBACK_0")),
        "Helemaal niet duidelijk":  int(os.getenv("DETECTION_FEEDBACK_1")),
        "Onbestemde positie in ground truth":  int(os.getenv("DETECTION_FEEDBACK_NONEXISTENT")),
        "Niet duidelijk":  int(os.getenv("DETECTION_FEEDBACK_1")),
        "Zeer duidelijk":  int(os.getenv("DETECTION_FEEDBACK_0")),
    }
    return rating_dict.get(rating, None)


def count_categories(tensor, categories):
    counts = []
    for row in tensor:
        row_counts = [torch.sum(row == torch.tensor(int(category))).item() for category in categories]
        counts.append(row_counts)
    return torch.tensor(counts)


def find_best_window(long_text, ground_truth, device, tokenizer, window_size=512, stride=256, similarity_model=None):
    """
    Find the best sliding window of text that is most similar to the ground truth.
    Uses a semantic similarity model to evaluate the windows.
    """
    if similarity_model is None:
        similarity_model = SentenceTransformer('NetherlandsForensicInstitute/robbert-2022-dutch-sentence-transformers')
        similarity_model.to(device)
        print("Loading the model in the function")
    
    # Tokenize the full text
    tokens = tokenizer.tokenize(long_text)
    
    best_score = -1
    best_window = None

    
    # Handle special cases in sequence length
    if len(tokens) < window_size:
        # Handle short sequences (optional: pad or skip)
        start_indices = [0]
    elif len(tokens) - window_size + 1 < stride:
        # Not enough room for a second stride step
        start_indices = [0, len(tokens) - window_size]
    else:
        # Standard sliding window
        start_indices = list(range(0, len(tokens) - window_size + 1, stride))

    
    # Create sliding windows
    for start in start_indices:
        window_tokens = tokens[start:start + window_size]
        window_text = tokenizer.convert_tokens_to_string(window_tokens)
        
        # Calculate cosine similarity with ground truth
        embeddings = similarity_model.encode([window_text, ground_truth])
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        
        if similarity > best_score:
            best_score = similarity
            best_window = window_tokens
    
    return tokenizer.convert_tokens_to_string(best_window)


# tokenize queries and answers together to provide proper context to reward model
def tokenize_fn_with_best_window(examples, feedback_train, tokenizer, max_length, stride, device, rl_training=False, similarity_model=None):

    """
    Tokenization function choosing the best window in the answer using similarity score with ground truth.
    """
    # Ensure max_length is not greater than 512
    max_length = min(max_length, 512)


    # print("response text", examples["response_text"])
    response_tokens = len(tokenizer(examples["response_text"], truncation=False)[0])
    
    precond_tokens = len(tokenizer(examples["precondition_text"], truncation=False)[0])
    # print(f"Precondition tokens: {precond_tokens}")
    pos_tokens = len(tokenizer(examples["precondition_position"], truncation=False)[0])
    # print(f"precond text: {examples['precondition_text']}")
    # print(f"Pos tokens: {pos_tokens}")

    precon_text = examples["precondition_text"]
    precon_pos = examples["precondition_position"]
    response = examples["response_text"]

    # combine the text for each separate use case
    if feedback_train == "feedback_extraction":
        # print("Tokenizing for feedback extraction")
        # choose the best sequence if the number of tokens is more than 512
        num_tokens = response_tokens + precond_tokens
        if num_tokens > max_length:
            window_size = max_length - precond_tokens
            stride = window_size // 2
            ground_truth = precon_text
            # print(f"Window size: {window_size}")
            assert 0 <= window_size <= max_length, "The window size must be between 0 and max_length."

            text = find_best_window(response, ground_truth, device, tokenizer, window_size=window_size, stride=stride, similarity_model=similarity_model)
            # print(text)
            # combined_texts = [f"{t} {r}" for t, r in zip(examples["precondition_text"], text)] --> batched version
            combined_texts = f"{precon_text} {text}"
            # combined_tokens = len(tokenizer(examples["precondition_position"], truncation=True, padding="max_length", max_length=max_length)[0])

        else:
            # combined_texts = [f"{t} {r}" for t, r in zip(examples["precondition_text"], examples["response_text"])] --> batched version
            combined_texts = f"{precon_text} {response}"
    
    
    
    
    elif feedback_train == "feedback_detection":
        # print("Tokenizing for feedback detection")
        num_tokens = response_tokens + precond_tokens + pos_tokens
        # print(num_tokens)
        # print(max_length)
        # print(examples["response_text"])
        if num_tokens > max_length:
            # print("We get here at all")
            # window_size = max_length - (precond_tokens + pos_tokens)
            window_size = max(max_length - (precond_tokens + pos_tokens), 1)
            stride = window_size // 2
            ground_truth = precon_text + " " + precon_pos
            assert 0 <= window_size <= max_length, "The window size must be between 0 and max_length."
            text = find_best_window(response, ground_truth, device, tokenizer, window_size=window_size, stride=stride, similarity_model=similarity_model)

            #combined_texts = [f"{t} {p} {r}" for t, p, r in zip(examples["precondition_text"], examples["precondition_position"], text)] --> batched version
            
            combined_texts = f"{precon_text} {precon_pos} {text}"
        else:
            #combined_texts = [f"{t} {p} {r}" for t, p, r in zip(examples["precondition_text"], examples["precondition_position"], examples["response_text"])] --> batched version
            combined_texts = f"{precon_text} {precon_pos} {response}"

    #Only return pytorch tensors if doing RL training loop
    if rl_training:
        tokenized = tokenizer(combined_texts, return_tensors='pt', truncation=True, padding="max_length", max_length=max_length)
        if len(tokenized['input_ids'][0]) != 512:
            print(f"Tokenized length: {len(tokenized['input_ids'][0])}")

        assert len(tokenized['input_ids'][0] <= max_length), "Tokenized input exceeds max_length."
    else:
        tokenized = tokenizer(combined_texts, truncation=True, padding="max_length", max_length=max_length)
        if len(tokenized['input_ids']) != 512:
            print(f"Tokenized length: {len(tokenized['input_ids'])}")

    return tokenized


def tokenize_fn_basic_batched(examples, feedback_train, tokenizer, rl_training=False):
    """
    Basic tokenization with standard cutoff after 512 tokens. Made to be applied to dataset batches.
    """
    if feedback_train == "feedback_extraction":
        combined_texts = [f"{c} {r}" for c, r in zip(examples["precondition_text"], examples["response_text"])]

    elif feedback_train == "feedback_detection":
        combined_texts = [f"{c} {p} {r}" for c, p, r in zip(examples["precondition_text"], examples["precondition_position"], examples["response_text"])]
    
    if rl_training:
        tokenized = tokenizer(combined_texts, return_tensors='pt', truncation=True, padding="max_length")
    else:
        tokenized = tokenizer(combined_texts, truncation=True, padding="max_length")

    return tokenized


def tokenize_fn_basic(examples, feedback_train, tokenizer, rl_training=False):
    """
    Basic tokenization with standard cutoff after 512 tokens.
    """

    precon_text = examples["precondition_text"]
    precon_pos = examples["precondition_position"]
    response = examples["response_text"]

    if feedback_train == "feedback_extraction":
        combined_texts = f"{precon_text} {response}"

    elif feedback_train == "feedback_detection":
        combined_texts = f"{precon_text} {precon_pos} {response}"

    if rl_training:
        tokenized = tokenizer(combined_texts, return_tensors='pt', truncation=True, padding="max_length")
    else:
        tokenized = tokenizer(combined_texts, truncation=True, padding="max_length")
    
    return tokenized





""" 
TF-IDF based tokenization, use if needed.
"""

# TODO: implement relevant sequence extraction with TFIDF if you have extra time...

# from sklearn.feature_extraction.text import TfidfVectorizer
# from transformers import AutoTokenizer
# import re

# def extract_relevant_sentences(long_text, ground_truth, tokenizer, max_tokens=MAX_LENGTH):
#     # Extract keywords from ground truth using TF-IDF
#     vectorizer = TfidfVectorizer(stop_words='english', max_features=20)

#     gt_tokens = len(tokenizer.tokenize(ground_truth))
#     sequence_length = int(gt_tokens * (1 + threshold_ratio))
#     sentences = re.split(r'[.!?]+', long_text)
    
#     # Score sentences based on ground truth keywords
#     all_texts = sentences + [ground_truth]
#     tfidf_matrix = vectorizer.fit_transform(all_texts)
    
#     # Get ground truth vector (last item)
#     gt_vector = tfidf_matrix[-1]
    
#     # Calculate similarity scores for each sentence
#     sentence_scores = []
#     for i, sentence in enumerate(sentences):
#         if sentence.strip():
#             similarity = (tfidf_matrix[i] * gt_vector.T).toarray()[0, 0]
#             sentence_scores.append((sentence, similarity))
    
#     # Sort by relevance and select top sentences
#     sentence_scores.sort(key=lambda x: x[1], reverse=True)
    
#     selected_text = ""
#     for sentence, score in sentence_scores:
#         test_text = selected_text + " " + sentence
#         if len(tokenizer.tokenize(test_text)) <= max_tokens:
#             selected_text = test_text
#         else:
#             break
    
#     return selected_text.strip()









########## Reward trainer ###################

class CustomRewardTrainer(Trainer):
    def __init__(self, *args, loss_type="mse", weight_strategy=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type  # "mse", "huber", or custom
        self.weight_strategy = weight_strategy  # "linear", "inverse", or None

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        # Extract labels (ratings) and optional sample weights
        labels = inputs.pop("labels").float()  # Shape: (batch_size)
        
        # Optional: Compute sample weights dynamically
        weights = self._get_sample_weights(labels) if self.weight_strategy else None
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits.squeeze()  # Shape: (batch_size) --> logits are the predicted rewards in this case
        
        # Custom loss calculation
        #TODO: take cross entropy loss here
        loss = self._compute_custom_loss(logits, labels, weights)
        
        return (loss, outputs) if return_outputs else loss

    def _compute_custom_loss(self, logits, labels, weights=None):
        if self.loss_type == "mse":
            loss = F.mse_loss(logits, labels, reduction="none") # --> MSE provides precise regression BUT sensitive to outliers
        elif self.loss_type == "huber":
            loss = F.huber_loss(logits, labels, reduction="none", delta=1.0) #--> balances between MSE and MAE for data that has outliers/ noise
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

        # Apply sample weights if provided
        if weights is not None:
            loss = loss * weights
            loss = loss.mean()  # Normalize by mean if weights are unnormalized
        else:
            loss = loss.mean()
        
        return loss

    def _get_sample_weights(self, labels):
        """
        Generate sample weights based on rating values.
        
        
        """
        if self.weight_strategy == "linear":
            # Linear weighting (e.g., emphasize extremes)
            weights = torch.abs(labels - labels.mean()) + 1.0
        elif self.weight_strategy == "inverse":
            # Inverse frequency weighting (if ratings are skewed)
            unique, counts = torch.unique(labels, return_counts=True)
            freq = counts.float() / len(labels)
            weight_map = 1.0 / (freq + 1e-6)  # Avoid division by zero
            weights = torch.tensor([weight_map[(unique == lbl).nonzero().item()] for lbl in labels])
        else:
            weights = None
        
        return weights.to(labels.device) if weights is not None else None



    def compute_metrics(self, eval_preds):
        predictions, labels = eval_preds
        predictions = predictions.squeeze()
        
        # Regression metrics
        mse = mean_squared_error(labels, predictions)
        pearson = pearsonr(labels, predictions)[0] # Pearson correlation coefficient
        
        # Threshold accuracy --> 
        tolerance_acc = (np.abs(predictions - labels) <= 0.5).mean()
        
        return {"mse": mse, "pearson": pearson, "tolerance_acc": tolerance_acc}
    


    
    #TODO: evaluate whether the plotting should be done or whether it is redundant to add them

    # def evaluation_loop(self, *args, **kwargs):
    #     output = super().evaluation_loop(*args, **kwargs)
    #     predictions = output.predictions.squeeze()
    #     labels = output.label_ids
        
    #     # Generate plots (saved to disk or logged to W&B)
    #     plot_distributions(predictions, labels, self.state.epoch)
    #     plot_calibration(predictions, labels)
        
    #     return output











"""
Other useful training things, look at later if needed.
"""


#TODO: debug if this is truly needed...

# add distributioncallback to trianing to evaluate 

# class DistributionCallback(TrainerCallback):
#     def on_evaluate(self, args, state, control, metrics, **kwargs):
#         # Get predictions and labels from the trainer's eval loop
#         eval_results = trainer.evaluate()
#         predictions = eval_results["eval_predictions"]
#         labels = eval_results["eval_labels"]
        
#         # Log histogram to W&B
#         wandb.log({
#             "reward_histogram": wandb.Histogram(predictions),
#             "true_ratings_histogram": wandb.Histogram(labels),
#         })


# def plot_distributions(predictions, labels, epoch):
#     plt.figure(figsize=(10, 4))
#     plt.subplot(1, 2, 1)
#     plt.hist(predictions, bins=20, alpha=0.7, label="Predicted")
#     plt.title("Predicted Rewards")
    
#     plt.subplot(1, 2, 2)
#     plt.hist(labels, bins=20, alpha=0.7, label="True Ratings", color="orange")
#     plt.title("True Ratings")
    
#     plt.savefig(f"distributions_epoch_{epoch}.png")
#     plt.close()

# class PlotCallback(TrainerCallback):
#     def on_evaluate(self, args, state, control, **kwargs):
#         predictions = trainer.predict(test_dataset).predictions.squeeze()
#         labels = test_dataset["ratings"]
#         plot_distributions(predictions, labels, state.epoch)




# def plot_calibration(predictions, labels):
#     """
#     Function to check if the sdfs

    

#     """
#     bin_means = np.linspace(1, 5, num=5)  # For 1-5 ratings
#     bin_centers = []
#     empirical_means = []
    
#     for i in range(len(bin_means) - 1):
#         mask = (labels >= bin_means[i]) & (labels < bin_means[i+1])
#         if mask.sum() > 0:
#             bin_centers.append((bin_means[i] + bin_means[i+1]) / 2)
#             empirical_means.append(predictions[mask].mean())
    
#     plt.plot(bin_centers, empirical_means, marker="o")
#     plt.plot([1, 5], [1, 5], linestyle="--", color="gray")  # Ideal line
#     plt.xlabel("True Rating")
#     plt.ylabel("Predicted Reward")
#     plt.savefig("calibration_plot.png")




# from transformers import DefaultDataCollator

# #TODO: only do this if the labels fix does not work for some reason

# class RewardDataCollator(DefaultDataCollator):
#     def __call__(self, features):

#         ratings = [f.pop("rating") for f in features]  # Removes rating from features temporarily
#         batch = super().__call__(features)
#         # Explicitly ensure rating is included
#         print(features)
#         # Re-inject ratings into the batch
#         batch["rating"] = torch.tensor(ratings, dtype=torch.float32)
#         return batch


#################### Reward function RL training #################################

class CustomRewardFunction:
    def __init__(self, reward_model_extraction, reward_model_detection, reward_tokenizer, max_length, stride, rl_tokenization, device, similarity_model, weight_extraction=1.0, weight_detection=1.0, weight_length_penalty=0.01, detection_difference=5, custom_logger=None):
        """
        Custom reward function that calculates rewards based on the extraction and detection of preconditions in responses.
        Args:
            reward_model_extraction (AutoModelForSequenceClassification): Model for extracting preconditions.
            reward_model_detection (AutoModelForSequenceClassification): Model for detecting preconditions.
            reward_tokenizer (AutoTokenizer): Tokenizer for processing input text.
            weight_extraction (float): Weight for the extraction reward.
            weight_detection (float): Weight for the detection reward.
        """
        self.weight_extraction = weight_extraction
        self.weight_detection = weight_detection
        self.weight_length_penalty = weight_length_penalty
        self.reward_model_extraction = reward_model_extraction
        self.reward_model_detection = reward_model_detection
        self.reward_tokenizer = reward_tokenizer
        self.max_length = max_length
        self.stride = stride
        self.device = device
        self.rl_tokenization = rl_tokenization
        self.detection_difference = detection_difference
        self.custom_logger = custom_logger
        self.similarity_model = similarity_model
        

        self.__name__ = "precondition_reward_function"

    def __call__(self, prompts, completions, completion_ids, **kwargs):
        """
        This function was written assuming that the precondition positions and precondition texts are passed aas a dictionary that is stored for the respective prompt in the prompt dataset TODO: --> double-check!!!
        """
        #TODO: for several batched samples and prompts here...
        # print(kwargs.keys())
        precondition_texts_list = kwargs["precondition_texts"]
        precondition_positions_list = kwargs["precondition_positions"]

        # Make sure that reward models and tokenizers are compatible
        self.reward_model_extraction.resize_token_embeddings(len(self.reward_tokenizer))
        self.reward_model_detection.resize_token_embeddings(len(self.reward_tokenizer))

        

        # print(precondition_texts_list)

        response_rewards = []

        with torch.no_grad():
            total_reward_extraction = 0
            total_reward_detection = 0

            for prompt, response, precondition_texts, precondition_positions in zip(prompts, completions, precondition_texts_list, precondition_positions_list):

                precondition_texts_dict = ast.literal_eval(precondition_texts)
                precondition_positions_dict = ast.literal_eval(precondition_positions)
                num_preconditions = len(precondition_texts_dict)
                response_reward = 0
                total_reward_detection = 0
                total_reward_extraction = 0
                all_precons_and_positions_text = ""

                # print(f"precondition texts keys: {precondition_positions_dict.keys()}")

                # iterate over all preconditions and give reward for response cumulatively
                for condition_id in precondition_texts_dict.keys():

                    # extraction_text = precondition_texts_dict[condition_id] + " " + response
                    # detection_text = precondition_texts_dict[condition_id] + " " + precondition_positions_dict[condition_id] + " " + response
                    inputs = {}
                    inputs["precondition_text"] = precondition_texts_dict[condition_id]
                    inputs["precondition_position"] = precondition_positions_dict[condition_id]
                    inputs["response_text"] = response

                    #gather toatl precondition text and position length
                    all_precons_and_positions_text += inputs["precondition_text"] + inputs["precondition_position"]

                    # tokenize response
                    if self.rl_tokenization == "basic":
                        inputs_extraction = tokenize_fn_basic(inputs, "feedback_extraction", self.reward_tokenizer, rl_training=True).to(self.device)
                        inputs_detection = tokenize_fn_basic(inputs, "feedback_detection", self.reward_tokenizer, rl_training=True).to(self.device)
                    elif self.rl_tokenization == "best_window":
                        inputs_extraction = tokenize_fn_with_best_window(inputs, "feedback_extraction", self.reward_tokenizer, self.max_length, self.stride, self.device, rl_training=True, similarity_model=self.similarity_model).to(self.device)
                        inputs_detection = tokenize_fn_with_best_window(inputs, "feedback_detection", self.reward_tokenizer, self.max_length, self.stride, self.device, rl_training=True, similarity_model=self.similarity_model).to(self.device)

                    

                    
                    # get reward model valuation
                    # print(f"Inputs extraction type: {type(inputs_extraction)}")
                    # print(f"Inputs detection type: {type(inputs_detection)}")
                    # print(f"Inputs extraction shape: {inputs_extraction['input_ids'].shape}")
                    # print(f"Inputs detection shape: {inputs_detection['input_ids'].shape}")
                    with torch.no_grad():
                        outputs_extraction = self.reward_model_extraction(**inputs_extraction)
                        outputs_detection = self.reward_model_detection(**inputs_detection)
                    
                    # add up rewads for all preconditions
                    reward_extraction = outputs_extraction.logits.item()
                    reward_detection = outputs_detection.logits.item()
                    total_reward_extraction += reward_extraction
                    total_reward_detection += (reward_detection - self.detection_difference) # subtracting detection_difference here to get to the proper detection difference

                    # get number fo response words for length penalty
                    response_words = len(response.split())


                # splitting prompt to get number of words
                essential_words = len(all_precons_and_positions_text.split())
                length_penalty = response_words - essential_words + 10 * len(precondition_texts_dict)

                # add total prompt reward to list of prompt rewards
                response_reward = (self.weight_extraction * total_reward_extraction + self.weight_detection * total_reward_detection  - self.weight_length_penalty * length_penalty) / num_preconditions

                # # add length penalty to logged variabels
                # weighted_length_penalty = self.weight_length_penalty * length_penalty
                # if self.custom_logger is not None:
                #     self.custom_logger.accumulate(weighted_length_penalty)

                #add reward to reward list
                response_rewards.append(response_reward)

            return torch.tensor(response_rewards)
        




class CustomMetricLogger(TrainerCallback):

    def __init__(self):
        self.custom_length_penalty_sum = 0.0
        self.custom_length_penalty_count = 0
        self.last_logged_step = -1

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Only log if this is a new step (prevents duplicate logs)
        if state.global_step == self.last_logged_step:
            return

        if self.custom_length_penalty_count > 0:
            mean_custom_length_penalty = self.custom_length_penalty_sum / self.custom_length_penalty_count

            if wandb.run is not None:
                wandb.log({"train/length_penalty/mean": mean_custom_length_penalty})
                run_id = wandb.run.id

                with open(f"/home/jacques.furst/development/RAG/flintfiller-precondition-rl/custom_metrics/custom_metrics_{run_id}.log", "a") as f:
                    f.write(f"{state.global_step}, length penalty: {mean_custom_length_penalty}\n")

            self.custom_length_penalty_sum = 0.0
            self.custom_length_penalty_count = 0
            self.last_logged_step = state.global_step

    def accumulate(self, length_penalty):
        self.custom_length_penalty_sum += length_penalty
        self.custom_length_penalty_count += 1













######################################### PPO Classes ###################################################################
        


################################################# Copied from TRL trainer.utils ######################################################

def first_true_indices(bools: torch.Tensor, dtype=torch.long):
    """
    Takes an N-dimensional bool tensor and returns an (N-1)-dimensional tensor of integers giving
    the position of the first True in each "row".

    Returns the length of the rows (bools.size(-1)) if no element is True in a given row.

    Args:
        bools (`torch.Tensor`):
            An N-dimensional boolean tensor.
        dtype (`torch.dtype`, optional):
            The desired data type of the output tensor. Defaults to `torch.long`.

    Returns:
        `torch.Tensor`:
            An (N-1)-dimensional tensor of integers indicating the position of the first True
            in each row. If no True value is found in a row, returns the length of the row.
    """
    row_len = bools.size(-1)
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)
    return torch.min(zero_or_index, dim=-1).values

##############################################################################################################################################




class CustomRewardFunctionPPOTrainer:
    def __init__(self, reward_model_extraction, reward_model_detection, reward_tokenizer, max_length, stride, rl_tokenization, device, weight_extraction=1.0, weight_detection=1.0, detection_difference=5):
        """
        Custom reward function that calculates rewards based on the extraction and detection of preconditions in responses.
        Args:
            reward_model_extraction (AutoModelForSequenceClassification): Model for extracting preconditions.
            reward_model_detection (AutoModelForSequenceClassification): Model for detecting preconditions.
            reward_tokenizer (AutoTokenizer): Tokenizer for processing input text.
            weight_extraction (float): Weight for the extraction reward.
            weight_detection (float): Weight for the detection reward.
        """
        self.weight_extraction = weight_extraction
        self.weight_detection = weight_detection
        self.reward_model_extraction = reward_model_extraction
        self.reward_model_detection = reward_model_detection
        self.reward_tokenizer = reward_tokenizer
        self.max_length = max_length
        self.stride = stride
        self.device = device
        self.rl_tokenization = rl_tokenization
        self.detection_difference = detection_difference

        self.__name__ = "precondition_reward_function"

    def __call__(self, query, response, tokenizer, response_tokenized, precondition_texts_list, precondition_positions_list) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        This function was written assuming that the precondition positions and precondition texts are passed aas a dictionary that is stored for the respective prompt in the prompt dataset TODO: --> double-check!!!
        """
        #TODO: figure out format of the query_responses 

        # print(f"{query_responses}")

        prompt_rewards = []

        with torch.no_grad():
            total_reward_extraction = 0
            total_reward_detection = 0

            for response_tokenized, response, precondition_texts, precondition_positions in zip(response_tokenized, response, precondition_texts_list, precondition_positions_list):

                precondition_texts_dict = ast.literal_eval(precondition_texts)
                precondition_positions_dict = ast.literal_eval(precondition_positions)
                prompt_reward = 0
                # print(prompt_reward)
                total_reward_detection = 0
                total_reward_extraction = 0

                # print(f"precondition texts keys: {precondition_positions_dict.keys()}")

                # iterate over all preconditions and give reward for response cumulatively
                for condition_id in precondition_texts_dict.keys():

                    # extraction_text = precondition_texts_dict[condition_id] + " " + response
                    # detection_text = precondition_texts_dict[condition_id] + " " + precondition_positions_dict[condition_id] + " " + response
                    inputs = {}
                    inputs["precondition_text"] = precondition_texts_dict[condition_id]
                    inputs["precondition_position"] = precondition_positions_dict[condition_id]
                    inputs["response_text"] = response


                    if self.rl_tokenization == "basic":
                        inputs_extraction = tokenize_fn_basic(inputs, "feedback_extraction", self.reward_tokenizer, rl_training=True).to(self.device)
                        inputs_detection = tokenize_fn_basic(inputs, "feedback_detection", self.reward_tokenizer, rl_training=True).to(self.device)
                    elif self.rl_tokenization == "best_window":
                        inputs_extraction = tokenize_fn_with_best_window(inputs, "feedback_extraction", self.reward_tokenizer, self.max_length, self.stride, self.device, rl_training=True).to(self.device)
                        inputs_detection = tokenize_fn_with_best_window(inputs, "feedback_detection", self.reward_tokenizer, self.max_length, self.stride, self.device, rl_training=True).to(self.device)

                    # pass tensors into reward models
                    outputs_extraction = self.reward_model_extraction(**inputs_extraction)
                    outputs_detection = self.reward_model_detection(**inputs_detection)
                    
                    # add up rewads for all preconditions
                    total_reward_extraction += outputs_extraction.logits.item()
                    total_reward_detection += (outputs_detection.logits.item() - self.detection_difference) # subtracting detection_difference here to get to the proper detection difference
                
                # add total prompt reward to list of prompt rewards
                prompt_reward = self.weight_extraction * total_reward_extraction + self.weight_detection * total_reward_detection  # Divide by 100 to maybe have more stable training
                
                # # add large negative reward if model just generates a lot of eos tokens
                # eos_count = (response_tokenized == tokenizer.eos_token_id).sum().item()
                # if eos_count > 1:
                #     prompt_reward -= (eos_count - 1)
                
                prompt_rewards.append(prompt_reward)

            
            # subtract mean and divide by standard deviation for the rewards
            mean = np.mean(prompt_rewards)
            std = np.std(prompt_rewards)   

            normalized_prompt_rewards = [(r - mean) / std for r in prompt_rewards]

            # print(f"prompt rewards: {prompt_rewards}")
            reward_logits = torch.tensor(normalized_prompt_rewards)

            #TODO: figure out how to retunr the rewards in proper order here...
            # No need to order rewards since I give one reward per sequence, not a token-specific reward...
            # sequence_lengths = first_true_indices(query_responses[:, context_length:] == pad_token_id) - 1 + context_length

            return (
                None,
                reward_logits,
                None
                    )


##############################################################
# Custom data collator that does not pad label column

class LabelPreservingCollator(DataCollatorWithPadding):
    def __call__(self, features):
        # Extract labels (remove from features temporarily)
        labels = [f.pop("additional_entries") for f in features]

        # Use parent collator for everything else (tokenizer padding etc)
        # print(f"Data collaotr features {features}")
        batch = super().__call__(features)

        # Re-attach original labels without modification
        batch["additional_entries"] = labels

        return batch
##############################################################



def tokenize_and_keep_original(example, tokenizer):
    # Tokenize the prompt column
    tokenized = tokenizer(example["prompt"], truncation=False, padding=False, max_length=2000)
    # Keep the original text
    # tokenized["original_text"] = example["prompt"]
    return tokenized

# Create label column for this to be handled properly in PPO Trainer
def create_label(example):
    return {"additional_entries": (example["prompt"], example["precondition_texts"], example["precondition_positions"])}

# #######################

# # Value head model for PPO

# class ValueHeadModel(nn.Module):
#     def __init__(self, base_model_name):
#         super().__init__()
#         self.model = AutoModelForCausalLM.from_pretrained(base_model_name, output_hidden_states=True)
#         self.value_head = nn.Linear(self.model.config.hidden_size, 1)

#     def forward(self, input_ids, attention_mask=None, **kwargs):
#         output = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
#         hidden_states = output.hidden_states[-1]
#         values = self.value_head(hidden_states).squeeze(-1)
#         return output, values

#     def score(self, hidden_states):
#         return self.value_head(hidden_states).squeeze(-1)







