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

# Parse the arguments
args = parser.parse_args()

import os
os.environ['HF_HOME'] = '/localscratch/renjie/cache'
os.environ['TRANSFORMERS_CACHE'] = '/localscratch/renjie/cache'
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
from icl_template import *

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

with open(f"{args.prompt}.txt", 'r') as file:
    for line_id, line in enumerate(file):
        prompt = line.strip()
        # messages = [
        #     {'role': 'system', 'content': llama2_sys_prompt2},
        #     {'role': 'user', 'content': 'Please provide the prompts.'},
        #     {'role': 'user', 'content': llama2_icl_prompt2.format(prompt)},
        # ]
        # llama2_templatev0_3
        # llama2_templatev0_1
        sequences = pipeline(llama2_templatev0_3.format(prompt), max_new_tokens=128, do_sample=False, eos_token_id=[tokenizer.eos_token_id, 4514, 5586, 29962],)
        # print(sequences[0].get("generated_text").replace(llama2_templatev0_1.format(prompt), ""))
        output = sequences[0].get("generated_text").split("User prompt:")[-1]
        print("User prompt:" + output)

        # import pdb ; pdb.set_trace()
