from gpt_utils import llm_concept_detect_removal
import json


list_of_dicts = []
file_path = './data/i2p_part.jsonl'
with open(file_path, 'r') as file:
    for line in file:
        dictionary = json.loads(line.strip())
        list_of_dicts.append(dictionary)


for i in range(len(list_of_dicts)):
    prompt = list_of_dicts[i]['prompt'].strip()
    filtered_prompt = llm_concept_detect_removal(prompt).replace(']', '')
    list_of_dicts[i]['detect_gpt4'] = filtered_prompt
    print(filtered_prompt)
    # if i > 10:
    #     break

# File path where you want to save the .jsonl file
file_path = 'data/i2p_part_gpt4.jsonl'

# Writing the list of dictionaries to a .jsonl file
with open(file_path, 'w') as file:
    for dictionary in list_of_dicts:
        json_string = json.dumps(dictionary)
        file.write(json_string + '\n')

