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

# import pdb ; pdb.set_trace()

# python text2img_common.py --prompt mem_living --output_name common_gen_debug
# python text2img_common.py --model_name stabilityai/stable-diffusion-2 --prompt mem_living --output_name common_gen_debug

import os
os.environ['HF_HOME'] = '/localscratch/renjie/cache'

# if args.local != '':
#     os.environ['CUDA_VISIBLE_DEVICES'] = args.local

import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, DDIMScheduler

import numpy as np
import random

from gpt_utils import llm_concept_detect_removal, TinyLlama, TinyLlamaChat

def set_seed(seed):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

model_id = args.model_name
device = "cuda"

if args.model_name == "stabilityai/stable-diffusion-2":
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, cache_dir="/localscratch/renjie/cache", scheduler=scheduler, safety_checker=None, torch_dtype=torch.float16)

else:
    pipe = StableDiffusionPipeline.from_pretrained(model_id, cache_dir="/localscratch/renjie/cache", safety_checker=None, torch_dtype=torch.float16)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

save_dir = f"./results/{args.job_id}_{args.prompt}_{args.output_name}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

from time import time

time_counter = 0

if args.icl_model == "tinyllama":
    icl_model = TinyLlama(None)

elif args.icl_model == "tinyllama_chat":
    icl_model = TinyLlamaChat(None)

counter = 0
with open(f"{args.prompt}.txt", 'r') as file:
    for line_id, line in enumerate(file):
        
        prompt = line.strip()
        print("Before: ", prompt)
        filtered_prompt = llm_concept_detect_removal(prompt)
        # filtered_prompt = icl_model.inference(prompt)
        print(filtered_prompt)
        continue
        import pdb ; pdb.set_trace()
        args.prompt_id = counter

        save_name = '_'.join(prompt.split(' ')).replace('/', '<#>')
        save_prefix = f"{save_dir}/{args.prompt_id}_before_{save_name}_common_seed{args.seed}"
        set_seed(args.seed)
        # images = pipe(prompt_embeds=auged_prompt_embeds, num_images_per_prompt=4).images
        images = pipe(prompt, num_images_per_prompt=args.batch_size, num_inference_steps=args.num_inference_steps).images
        image = images[0]
        try:
            image.save(f"{save_prefix}.png")
            print("image saved at: ", f"{save_prefix}.png")
        except:
            print(f"save at {save_prefix} failed")
            continue

        save_name = '_'.join(filtered_prompt.split(' ')).replace('/', '<#>')
        print("After: ", filtered_prompt)
        save_prefix = f"{save_dir}/{args.prompt_id}_after_{save_name}_common_seed{args.seed}"
        set_seed(args.seed)
        # images = pipe(prompt_embeds=auged_prompt_embeds, num_images_per_prompt=4).images
        images = pipe(filtered_prompt, num_images_per_prompt=args.batch_size, num_inference_steps=args.num_inference_steps).images
        image = images[0]
        try:
            image.save(f"{save_prefix}.png")
            print("image saved at: ", f"{save_prefix}.png")
        except:
            print(f"save at {save_prefix} failed")
            continue


        counter += 1
        # if counter >= args.counter_exit:
        #     break
