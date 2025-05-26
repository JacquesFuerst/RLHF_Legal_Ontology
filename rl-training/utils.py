import torch

def parse_ratings(rating):
    rating_dict = {
        "Volledig fout": 0,
        "Deels fout": 1,
        "Grotendeels correct": 2,
        "Volledig correct": 3,
        "Geen positie in ground truth": 5, #TODO: change
        "Niet goed": 6,
        "Goed": 4,
        "Duidelijk": 5,
        "Helemaal niet duidelijk": 6,
        "Onbestemde positie in ground truth": 5,
        "Niet duidelijk": 5,
        "Zeer duidelijk": 4,
    }
    return rating_dict.get(rating, None)


def count_categories(tensor, categories):
    counts = []
    for row in tensor:
        row_counts = [torch.sum(row == category).item() for category in categories]
        counts.append(row_counts)
    return torch.tensor(counts)
