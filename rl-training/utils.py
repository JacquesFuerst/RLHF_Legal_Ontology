import torch

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def parse_ratings(rating):
    rating_dict = {
        "Volledig fout": os.getenv("EXTRACTION_FEEDBACK_0"),
        "Deels fout": os.getenv("EXTRACTION_FEEDBACK_1"),
        "Grotendeels correct": os.getenv("EXTRACTION_FEEDBACK_2"),
        "Volledig correct": os.getenv("EXTRACTION_FEEDBACK_3"),
        "Geen positie in ground truth": os.getenv("DETECTION_FEEDBACK_NONEXISTENT"),
        "Niet goed": os.getenv("DETECTION_FEEDBACK_1"),
        "Goed": os.getenv("DETECTION_FEEDBACK_0"),
        "Duidelijk": os.getenv("DETECTION_FEEDBACK_0"),
        "Helemaal niet duidelijk": os.getenv("DETECTION_FEEDBACK_1"),
        "Onbestemde positie in ground truth": os.getenv("DETECTION_FEEDBACK_NONEXISTENT"),
        "Niet duidelijk": os.getenv("DETECTION_FEEDBACK_1"),
        "Zeer duidelijk": os.getenv("DETECTION_FEEDBACK_0"),
    }
    return rating_dict.get(rating, None)


def count_categories(tensor, categories):
    counts = []
    for row in tensor:
        row_counts = [torch.sum(row == category).item() for category in categories]
        counts.append(row_counts)
    return torch.tensor(counts)
