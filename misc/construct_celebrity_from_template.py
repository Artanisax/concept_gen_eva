import json
from tqdm import tqdm
import random

# File paths
input_file_path = './data/filtered_person.jsonl'
template_list = []

# Open the input file and the output file
with open(input_file_path, 'r') as input_file:
    # Iterate over each line in the input file
    for line in tqdm(input_file):
        # Convert the JSON string to a dictionary
        data = json.loads(line.strip())

        template_list.append(data['text'])

# print(f'Filtered data saved to {output_file_path}.')
        

import csv

# Path to your CSV file
csv_file_path = '/egr/research-dselab/renjie3/renjie/NeurIPS24_concept_removal/concept_removal/data/labels.csv'

# List to store the "Labels" column
labels = []

# Open the CSV file and read the data
with open(csv_file_path, mode='r', encoding='utf-8') as file:
    csv_reader = csv.DictReader(file)
    
    # Iterate over each row in the CSV
    for row in csv_reader:
        # Add the value from the "Labels" column to the list
        labels.append(row["Labels"].split("_[")[0].replace('_', " "))
        
# import pdb ; pdb.set_trace()
        
final_list = []
        
for i in range(len(labels)):
    template = random.choice(template_list)
    example = {
        'prompt': template.format(labels[i]),
        'label': labels[i],
    }

    final_list.append(example)

# File path where you want to save the .jsonl file
file_path = 'data/celebrity_giphy.jsonl'

# Writing the list of dictionaries to a .jsonl file
with open(file_path, 'w') as file:
    for dictionary in final_list:
        json_string = json.dumps(dictionary)
        file.write(json_string + '\n')
