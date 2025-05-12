import json
import random

# List your JSON file paths
json_files = ['/home/jacques.furst/development/RAG/flintfiller-precondition-rl/data/model_answers/model_answers_acts_Interpretatie_Vw_over_besluiten_op_aanvragen_voor_een_verblijfsvergunning_regulier_bepaalde_tijd.json', 
              '/home/jacques.furst/development/RAG/flintfiller-precondition-rl/data/model_answers/model_answers_acts_Participatiewet_most_recent_public.json', 
              '/home/jacques.furst/development/RAG/flintfiller-precondition-rl/data/model_answers/model_answers_acts_rijksbegrotingsccyclus.json', 
              '/home/jacques.furst/development/RAG/flintfiller-precondition-rl/data/model_answers/model_answers_facts_rijksbegrotingscyclus.json'
              ]

merged_data = []

# Load and merge all JSON data
for file in json_files:
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, list):
            merged_data.extend(data)
        else:
            merged_data.append(data)  # If it's a dict or single object

# Shuffle the combined list
# random.shuffle(merged_data)

#TODO: add facts Vw to file before writing to first data file
shared_file = '/home/jacques.furst/development/RAG/flintfiller-precondition-rl/data/model_answers/model_answers_facts_Interpretatie_Vw_over_besluiten_op_aanvragen_voor_een_verblijfsvergunning_regulier_bepaalde_tijd.json'

with open(shared_file, 'r', encoding='utf-8') as f:
        shared_data = json.load(f)

file_1_data = shared_data + merged_data

# Optionally, write to first data file
with open('/home/jacques.furst/development/RAG/flintfiller-precondition-rl/data/participant_data/data_file_1_unshuffled.json', 'w', encoding='utf-8') as f:
    json.dump(file_1_data, f, ensure_ascii=False, indent=2)

# Reverse the list
if isinstance(merged_data, list):
    merged_data.reverse()

# add shared data (to ensure inter-annotator agreement)
file_2_data = shared_data + merged_data

# Save to second data file
with open('/home/jacques.furst/development/RAG/flintfiller-precondition-rl/data/participant_data/data_file_2_unshuffled.json', 'w', encoding='utf-8') as f:
    json.dump(file_2_data, f, ensure_ascii=False, indent=2)