import json

json_files_first_doc = ['/home/jacques.furst/development/RAG/flintfiller-precondition-rl/data/model_answers/model_answers_acts_Participatiewet_most_recent_public.json',
                        '/home/jacques.furst/development/RAG/flintfiller-precondition-rl/data/model_answers/acts_rijksbegroting_model_answers.json',
                        '/home/jacques.furst/development/RAG/flintfiller-precondition-rl/data/model_answers/acts_besluit_Vw_model_answers.json',
                        '/home/jacques.furst/development/RAG/flintfiller-precondition-rl/data/model_answers/model_answers_facts_rijksbegrotingscyclus.json',
                        '/home/jacques.furst/development/RAG/flintfiller-precondition-rl/data/model_answers/facts_besluit_Vw_model_answers.json'
              ]

merged_data = []

# Load and merge all JSON data
for file in json_files_first_doc:
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, list):
            merged_data.extend(data)
        else:
            merged_data.append(data)  # If it's a dict or single object


# Optionally, write to first data file
with open('/home/jacques.furst/development/RAG/flintfiller-precondition-rl/data/participant_data/synthetic_feedback_dataset.json', 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=2)