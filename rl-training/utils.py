def parse_ratings(rating):
    rating_dict = {
        "Volledig fout": 0,
        "Deels fout": 1,
        "Grotendeels correct": 2,
        "Volledig correct": 3,
        "Geen positie in ground truth": 0, #TODO: change
        "Niet Goed": 1,
        "Goed": 2,
    }
    return rating_dict.get(rating, None)