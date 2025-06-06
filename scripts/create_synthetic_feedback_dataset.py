import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

json_files_doc = [os.getenv("MODEL_ANSWERS__DATA_1"), os.getenv("MODEL_ANSWERS__DATA_2"), os.getenv("MODEL_ANSWERS__DATA_3"), os.getenv("MODEL_ANSWERS__DATA_4"), os.getenv("MODEL_ANSWERS__DATA_5")]

merged_data = []

# Load and merge all JSON data
for file in json_files_doc:
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, list):
            merged_data.extend(data)
        else:
            merged_data.append(data)  # If it's a dict or single object


# Optionally, write to first data file
with open(os.getenv("SYNTHETIC_DATA_FILE"), 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=2)