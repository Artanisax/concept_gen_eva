from datasets import load_dataset
import json

dataset = load_dataset("AIML-TUDA/i2p", cache_dir="/localscratch/renjie/cache/")

# Filter rows where 'categories' == "sexual content" in the 'train' split
# filtered_dataset = dataset['train'].filter(lambda example: "sex" in example['categories'])

# label_str = set()

# for example in dataset['train']:
#     label_list = example['categories'].split(',')
#     for label in label_list:
#         label_str.add(label.strip())

# # Now, let's save this filtered dataset to a file. 
# # You can choose the format you prefer (e.g., csv, json, etc.)

# # For CSV format
# filtered_dataset.to_csv('filtered_dataset.csv')

# # For JSON format
# filtered_dataset.to_json('filtered_dataset.json')
        
# {'shocking', 'harassment', 'illegal activity', 'violence', 'sexual', 'self-harm', 'hate'}
        
target_label = {"self-harm", 'violence', 'sexual'}
list_of_dicts = []

for example in dataset['train']:
    label_str = []
    for label in target_label:
        if label in example['categories']:
            label_str.append(label)
    label_str = ','.join(label_str)
    if len(label_str) > 1 and example['inappropriate_percentage'] >= 60:
        temp_dic = {"prompt": example['prompt'], 'label': label_str, "inappropriate_percentage": example['inappropriate_percentage']}
        list_of_dicts.append(temp_dic)

# File path where you want to save the .jsonl file
file_path = 'data/i2p_part.jsonl'

# Writing the list of dictionaries to a .jsonl file
with open(file_path, 'w') as file:
    for dictionary in list_of_dicts:
        json_string = json.dumps(dictionary)
        file.write(json_string + '\n')

# print(label_str)

# import pdb ; pdb.set_trace()
