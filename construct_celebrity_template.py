import json
from tqdm import tqdm

# File paths
input_file_path = '/egr/research-dselab/renjie3/renjie/NeurIPS24_concept_removal/concept_removal/data/50k-metadata.jsonl'
output_file_path = './data/filtered_person.jsonl'

# Open the input file and the output file
with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
    # Iterate over each line in the input file
    for line in tqdm(input_file):
        # Convert the JSON string to a dictionary
        data = json.loads(line.strip())
        
        # Check if "person" is one of the values in the dictionary
        if (data['text'].count("person") == 1 and "personal" not in data['text']) or (data['text'].count("Person") == 1 and "Personal" not in data['text']):
            # If yes, write the original JSON string to the output file
            data['text'] = data['text'].replace("person", "{}").replace("Person", "{}")
            data.pop('file_name')
            output_file.write(json.dumps(data) + '\n')

print(f'Filtered data saved to {output_file_path}.')
