import json
import random

import os

from dotenv import load_dotenv

load_dotenv()

# List your JSON file paths
# file_1 = os.getenv()


json_files_first_doc = ['/home/jacques.furst/development/RAG/flintfiller-precondition-rl/data/model_answers/model_answers_acts_Participatiewet_most_recent_public.json',
                        '/home/jacques.furst/development/RAG/flintfiller-precondition-rl/data/model_answers/acts_rijksbegroting_model_answers.json',
              ]

json_files_2nd_doc = [  '/home/jacques.furst/development/RAG/flintfiller-precondition-rl/data/model_answers/acts_besluit_Vw_model_answers.json',
                        '/home/jacques.furst/development/RAG/flintfiller-precondition-rl/data/model_answers/model_answers_facts_rijksbegrotingscyclus.json',
              ]

merged_data_1 = []
merged_data_2 = []

# Load and merge all JSON data
for file in json_files_first_doc:
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, list):
            merged_data_1.extend(data)
        else:
            merged_data_1.append(data)  # If it's a dict or single object

# Load and merge all JSON data
for file in json_files_2nd_doc:
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, list):
            merged_data_2.extend(data)
        else:
            merged_data_2.append(data)  # If it's a dict or single object

# Shuffle the combined list
# random.shuffle(merged_data)

#TODO: add facts Vw to file before writing to first data file
shared_file = '/home/jacques.furst/development/RAG/flintfiller-precondition-rl/data/model_answers/facts_besluit_Vw_model_answers.json'

with open(shared_file, 'r', encoding='utf-8') as f:
        shared_data = json.load(f)

file_1_data = shared_data + merged_data_1

# Optionally, write to first data file
with open('/home/jacques.furst/development/RAG/flintfiller-precondition-rl/data/participant_data/study_file_1_participatie_en_Vw_unshuffled.json', 'w', encoding='utf-8') as f:
    json.dump(file_1_data, f, ensure_ascii=False, indent=2)


file_2_data = shared_data + merged_data_2

# Optionally, write to first data file
with open('/home/jacques.furst/development/RAG/flintfiller-precondition-rl/data/participant_data/study_file_2_rijksbegroting_acts_Vw_unshuffled.json', 'w', encoding='utf-8') as f:
    json.dump(file_2_data, f, ensure_ascii=False, indent=2)

# # Reverse the list
# if isinstance(merged_data, list):
#     merged_data.reverse()

# # add shared data (to ensure inter-annotator agreement)
# file_2_data = shared_data + merged_data

# # Save to second data file
# with open('/home/jacques.furst/development/RAG/flintfiller-precondition-rl/data/participant_data/data_file_2_unshuffled.json', 'w', encoding='utf-8') as f:
#     json.dump(file_2_data, f, ensure_ascii=False, indent=2)