import json
from pathlib import Path

def load_json_files(results_dir):
    """Load all JSON files from the specified directory."""
    json_files = Path(results_dir).glob("*.json")
    data = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data.extend(json.load(f))
    return data

def validate_schema(data, schema):
    """Validate if each item in data conforms to the given schema."""
    for item in data:
        for key in schema:
            if key not in item:
                print(f"Missing key '{key}' in item: {item}")
                return False
            if not isinstance(item[key], type(schema[key])):
                print(f"Incorrect type for key '{key}' in item: {item}")
                return False
    return True

def main():
    results_dir = "results"
    schema = {
        "user_id": str,
        "task_type": str,
        "prompt": str,
        "split": str,
        "target": dict,
        "user_profile": dict
    }

    # Load all JSON data
    data = load_json_files(results_dir)

    # Validate schema
    if validate_schema(data, schema):
        print("All data conforms to the schema.")
    else:
        print("Some data does not conform to the schema.")

if __name__ == "__main__":
    main()
