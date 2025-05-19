import json

# Load the JSON data from the file
with open('evaluation/merged_wvs_gss.json', 'r') as file:
    data = json.load(file)

# Filter out elements with 'answer_value' of null
filtered_data = [
    {key: value for key, value in row.items() if value is not None}
    for row in data
]

# Save the filtered data back to the file or another file if needed
with open('filtered_merged_wvs_jss.json', 'w') as file:
    json.dump(filtered_data, file, indent=4)
