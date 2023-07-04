import json
import os


def read_file(file_to_save: str) -> dict:
    """
    Read function able to load from file the related json data.
    @param file_to_save: Filename where loading json data.
    @return: Loaded file in json format.
    """
    if file_to_save is not None:
        if os.path.exists(file_to_save):
            with open(file_to_save) as file:
                return json.load(file)


def save_record_in_file(file_to_save: str, strategy_key: str, strategy_value: dict):
    """
    Function able to save a record in a json file.
    @param file_to_save: Filename of the file to update.
    @param strategy_key: Strategy name.
    @param strategy_value: Strategy results.
    """
    results = read_file(file_to_save)
    if results is None:
        results = {}
    results[strategy_key] = strategy_value
    with open(file_to_save, 'w') as file:
        json.dump(results, file)
