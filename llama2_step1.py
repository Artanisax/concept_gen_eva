import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Process some integers.")

# Add arguments
parser.add_argument("--model_name", type=str, default="CompVis/stable-diffusion-v1-4", help="an integer to be processed")
parser.add_argument("--local", type=str, default='', help="The scale of noise offset.")
parser.add_argument("--prompt", type=str, default="a photo of an astronaut riding a horse on mars", help="The scale of noise offset.")
parser.add_argument("--job_id", type=str, default='local', help="The scale of noise offset.")
parser.add_argument("--output_name", type=str, default='local', help="The scale of noise offset.")
parser.add_argument("--counter_exit", default=10, type=int)
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--num_inference_steps", default=50, type=int)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--icl_model", default='tinyllama', type=str)
parser.add_argument("--template", default='llama2_templatev0_3', type=str)

# Parse the arguments
args = parser.parse_args()

import os
os.environ['HF_HOME'] = '/localscratch/renjie/cache'
os.environ['TRANSFORMERS_CACHE'] = '/localscratch/renjie/cache'
if args.local != '':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.local

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
from icl_template import *
import json

access_token = "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"
# model = "meta-llama/Llama-2-7b-chat-hf"
# model = "/localscratch/renjie/cache/Llama-2-7b-chat-hf"
# model = "/egr/research-dselab/renjie3/.cache/llama2/Llama-2-7b-chat-hf"
model = "/egr/research-dselab/renjie3/.cache/llama2/Llama-2-13b-hf"

# User prompt: American star Kobe Bryant is playing on the court. He leaps up and dunks.
# Unwanted concept: Kobe Bryant

tokenizer = AutoTokenizer.from_pretrained(model, token=access_token)
# print(tokenizer("]"))
# import pdb ; pdb.set_trace()

# model = AutoModelForCausalLM.from_pretrained(
#     model,
#     token=access_token,
#     cache_dir="/localscratch/renjie/cache"
# )

# messages = [
#     {'role': 'system', 'content': 'This is a system prompt.'},
#     {'role': 'user', 'content': 'This is the first user input.'},
#     {'role': 'assistant', 'content': 'This is the first assistant response.'},
#     {'role': 'user', 'content': 'This is the second user input.'},
# ]
# print('###### Default (yet Correct) Chat Template ######')
# print(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
# print('###### Corrected Chat Template ######')
# chat_template = open('./chat_templates/llama-2-chat.jinja').read()
# chat_template = chat_template.replace('    ', '').replace('\n', '')
# tokenizer.chat_template = chat_template
# print(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

# full_input = tokenizer.apply_chat_template(messages, tokenize=False)
# input_encoded = tokenizer(full_input, return_tensors="pt", padding=True).to(
#     model.device
# )
# output_ids = model.generate(
#     **input_encoded,
#     max_new_tokens=256,
#     do_sample=False,
#     pad_token_id=tokenizer.pad_token_id,
# )[0]
# reply_ids = output_ids[input_encoded["input_ids"].shape[-1] :]
# decoded = tokenizer.decode(reply_ids, skip_special_tokens=True).strip()

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    token=access_token,
    torch_dtype=torch.float16,
    device_map="auto",
    # cache_dir="/localscratch/renjie/cache"
)

list_of_dicts = []
file_path = './data/celebrity_giphy.jsonl'
with open(file_path, 'r') as file:
    for line in file:
        dictionary = json.loads(line.strip())
        list_of_dicts.append(dictionary)

# File path where you want to save the .jsonl file
file_path = f'data/{args.output_name}_{args.template}.jsonl'

with open(file_path, 'w') as file:
    for i in range(len(list_of_dicts)):
        prompt = list_of_dicts[i]['prompt'].strip()
        # messages = [
        #     {'role': 'system', 'content': llama2_sys_prompt2},
        #     {'role': 'user', 'content': 'Please provide the prompts.'},
        #     {'role': 'user', 'content': llama2_icl_prompt2.format(prompt)},
        # ]
        # llama2_templatev0_3
        # llama2_templatev0_1
        if args.template == "llama2_templatev0_4":
            template = llama2_templatev0_4
        elif args.template == "llama2_templatev0_3":
            template = llama2_templatev0_3
        elif args.template == "llama2_templatev0_5":
            template = llama2_templatev0_5
        sequences = pipeline(template.format(prompt), max_new_tokens=128, do_sample=False, eos_token_id=[tokenizer.eos_token_id, 4514, 5586, 29962],)
        # print(sequences[0].get("generated_text").replace(llama2_templatev0_1.format(prompt), ""))
        output = sequences[0].get("generated_text").split("User prompt:")[-1]
        print("User prompt:" + output)
        list_of_dicts[i]['detect_llama2_13b'] = "User prompt:" + output

        json_string = json.dumps(list_of_dicts[i])
        file.write(json_string + '\n')

        # if i > 3:
        #     break
