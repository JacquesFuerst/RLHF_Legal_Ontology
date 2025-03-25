import json


def read_json(file_path):
    """
    Function to read data from a JSON file.

    Parameters:
    file_path (str): The path to the JSON file

    Returns:
    list: The data from the JSON file
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        data = []
    return data


def write_json(file_path, data):
    """
    Function to write data to a JSON file.

    Parameters:
    file_path (str): The path to the JSON file
    data (list): The data to write to the JSON file

    Returns:
    None
    """
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


def add_to_json(file_path, new_entry):
    """
    Function to add a new entry to the JSON file.

    Parameters:
    file_path (str): The path to the JSON file
    new_entry (dict): The new entry to add to the JSON file

    Returns:
    None
    """
    data = read_json(file_path)
    data.append(new_entry)
    write_json(file_path, data)
