import json


def read_file(file_to_save):
    if file_to_save is not None:
        if os.path.exists(file_to_save):
            with open(file_to_save) as file:
                return json.load(file)


def save_record_in_file(file_to_save, strategy_key, strategy_value):
    results = read_file(file_to_save)
    if results is None:
        results = {}
    results[strategy_key] = strategy_value
    with open(file_to_save, 'w') as file:
        json.dump(results, file)
